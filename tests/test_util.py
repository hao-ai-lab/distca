from dataclasses import dataclass
import os
import socket
from typing import Optional

from megatron.core import parallel_state as mpu
import ray
import torch

from d2.runtime.attn_kernels.ops import nvshmem_get_unique_id, nvshmem_alloc_empty_unique_id, DispatcherWrapper
from d2.runtime.inplace_metadata import Metadata


######## Workers
@dataclass
class ParallelConfig:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: Optional[int] = None


class MegatronBaseWorker:
    """Worker base class to init communication groups (megatron and nvshmem)."""
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.nvshmem_initialized = False
        self.nvshmem_pe = None

    #### General init functions
    def get_node_ip_port(self):
        host_ipv4 = os.getenv("MY_HOST_IP", None)
        host_ipv6 = os.getenv("MY_HOST_IPV6", None)
        host_ip_by_env = host_ipv4 or host_ipv6
        host_ip_by_sdk = ray._private.services.get_node_ip_address()

        host_ip = host_ip_by_env or host_ip_by_sdk

        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return host_ip, str(port)

    def set_master_addr_port(self, master_addr: str, master_port: str):
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

    def init_comm(self, stride_q: int, stride_kv: int, max_tokens_query: int, max_tokens_key_value: int,
                  parallel_config: ParallelConfig):
        # Init megatron communication.
        if not torch.distributed.is_initialized():
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", rank=self.rank, world_size=self.world_size)
        # NOTE: do not set to local_rank here because the cuda visible device is set by ray.

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=parallel_config.virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank=None,
            use_sharp=False,
            context_parallel_size=parallel_config.context_parallel_size,
            expert_model_parallel_size=parallel_config.expert_model_parallel_size,
            expert_tensor_parallel_size=parallel_config.expert_tensor_parallel_size,
            nccl_communicator_config_path=None,
        )
        # Init nvshmem.
        if self.rank == 0:
            uid = nvshmem_get_unique_id()
        else:
            uid = nvshmem_alloc_empty_unique_id()
        torch.distributed.broadcast(uid, src=0)

        DispatcherWrapper.init(
            q_stride=stride_q,
            kv_stride=stride_kv,
            max_tokens_query=max_tokens_query,
            max_tokens_key_value=max_tokens_key_value,
            rank=self.rank,
            world_size=self.world_size,
            uid=uid,
        )


######## Data construction
def test_local_proj_metadata(world_size: int, seq_len: int, offset: int):
    """
    Test tool generating the easiest case: Each rank holds a sequence,
    and is sent to a rank with an offset. (0 means sending to itself)
    """
    num_recv_tokens = torch.zeros((world_size, world_size + 1), dtype=torch.int64)
    num_recv_tokens[:, -1] = seq_len
    i_diag = torch.arange(world_size, dtype=torch.int64)
    j_diag = (i_diag + offset) % world_size
    # the diagonal of num_recv_tokens is the seq_len
    fwd_recv_tokens = num_recv_tokens.clone()
    fwd_recv_tokens[j_diag, i_diag] = seq_len

    bwd_j_diag = (i_diag - offset) % world_size
    bwd_recv_tokens = num_recv_tokens.clone()
    bwd_recv_tokens[bwd_j_diag, i_diag] = seq_len

    mlp_to_attn_metadata = Metadata(
        dst_rank=torch.tensor(
            [(i + offset) % world_size for i in range(world_size)]
        ).reshape(world_size, 1),
        dst_offset=torch.tensor(
            [0] * world_size
        ).reshape(world_size, 1),
        seq_len=torch.tensor(
            [seq_len] * world_size
        ).reshape(world_size, 1),
        num_recv_tokens=fwd_recv_tokens,
        num_seqs=torch.tensor([1] * world_size).reshape(world_size, 1),
        world_size=world_size,
        num_total_recv_tokens=[seq_len] * world_size,
    )
    attn_to_mlp_metadata = Metadata(
        dst_rank=torch.tensor(
            [(i - offset) % world_size for i in range(world_size)]
        ).reshape(world_size, 1),
        dst_offset=mlp_to_attn_metadata.dst_offset.clone(),
        seq_len=mlp_to_attn_metadata.seq_len.clone(),
        num_recv_tokens=bwd_recv_tokens,
        num_seqs=mlp_to_attn_metadata.num_seqs.clone(),
        world_size=world_size,
        num_total_recv_tokens=[seq_len] * world_size,
    )
    mlp_to_attn_kv_metadata = Metadata(
        dst_rank=mlp_to_attn_metadata.dst_rank.clone().unsqueeze(-1),
        dst_offset=mlp_to_attn_metadata.dst_offset.clone().unsqueeze(-1),
        seq_len=mlp_to_attn_metadata.seq_len.clone(),
        num_recv_tokens=fwd_recv_tokens.clone(),
        num_seqs=mlp_to_attn_metadata.num_seqs.clone(),
        world_size=world_size,
        num_total_recv_tokens=[seq_len] * world_size,
    )
    mlp_to_attn_kv_grad_metadata = Metadata(
        dst_rank=attn_to_mlp_metadata.dst_rank.clone(),
        dst_offset=attn_to_mlp_metadata.dst_offset.clone(),
        seq_len=attn_to_mlp_metadata.seq_len.clone(),
        num_recv_tokens=bwd_recv_tokens.clone(),
        num_seqs=attn_to_mlp_metadata.num_seqs.clone(),
        world_size=world_size,
        seq_recv_mask=torch.ones(world_size, 1, 1),
        recv_seq_lens=mlp_to_attn_metadata.seq_len.clone(),
        num_total_recv_tokens=[seq_len] * world_size,
    )
    return mlp_to_attn_metadata, attn_to_mlp_metadata, mlp_to_attn_kv_metadata, mlp_to_attn_kv_grad_metadata


def gen_seq_lens(world_size: int, num_seqs: int, total_len: int) -> torch.Tensor:
    ratio = torch.rand((world_size, num_seqs)) + 0.25 / num_seqs   # Use a min value to guarantee that the sequence is not too short (0 after rounding)
    ratio = ratio / ratio.sum(dim=1, keepdim=True)
    seq_len = (ratio * total_len).round().int()
    seq_len_total = seq_len.sum(dim=1)
    seq_len_total_error = seq_len_total - total_len
    seq_len[:, -1] -= seq_len_total_error
    return seq_len
