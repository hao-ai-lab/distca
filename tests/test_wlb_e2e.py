"""
Debug example:
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 torchrun --nnodes 1 --nproc_per_node 2 test_megatron_e2e_pipeline.py --num-gpus-per-node 2 --pp-size 2 --num-microbatch 2
"""
import argparse
from functools import partial
import os
import time

import megatron.core.parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
import torch
from transformers import AutoConfig

from d2.runtime.compute_metadata import get_attn_metadata
from d2.runtime.megatron_patch.packed_seq_params import arg_to_cuda, PingPangSingleStepPackedSeqParams, PingPangPackedSeqParams
from d2.runtime.megatron_patch.forward_backward_func import forward_backward_pipelining_without_interleaving as forward_backward_func

from test_util import ParallelConfig, init_worker_torch_distributed, create_qkv_dispatch_pipeline_tick
from test_megatron_e2e import MegatronE2eWorker as BaseMegatronE2eWorker, set_random_seed
from megatron_test_utils import (
    gptmodel_forward, make_batch_generator, unwrap_model,
)
from typing import Optional

import d2.planner.wlb_planner
import wlbllm
import wlbllm.utils
import wlbllm.registry
import wlbllm.megatron_patch
import wlbllm.megatron_patch.dot_product_attention
import wlbllm.megatron_patch.backends
import wlbllm.megatron_patch.pp_schedules
from dataclasses import dataclass

@dataclass
class WLBPackedSeqParams():
    qkv_format: str = "thd"
    cu_seqlens_q: 'Tensor' = None
    cu_seqlens_kv: 'Tensor' = None
    cu_seqlens_q_padded: 'Tensor' = None
    cu_seqlens_kv_padded: 'Tensor' = None
    max_seqlen_q: 'Tensor' = None
    max_seqlen_kv: 'Tensor' = None
    wlb_metadata: dict = None
    pass


class MegatronE2eWorker(BaseMegatronE2eWorker):
    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        local_rank = int(os.getenv("LOCAL_RANK"))
        torch.cuda.set_device(local_rank)
        torch.set_default_device(torch.device("cuda", local_rank))

    def init_comm(self, parallel_config: ParallelConfig):
        # Init megatron communication.
        self.init_torch_distributed()
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
        self.as_world_size = None
        self.as_rank = None

    def forward_backward_batch(self, microbatches: list[dict], forward_only: bool=False,
                               mode: str="ping_pong", with_dummy: bool=True):

        microbatches = [{
            k: arg_to_cuda(v) for k, v in microbatch.items()
        } for microbatch in microbatches]
        if "orig" in mode:
            for mb in microbatches:
                psp = mb["packed_seq_params"]
                if isinstance(psp, PingPangSingleStepPackedSeqParams):
                    mb["packed_seq_params"] = mb["packed_seq_params"].mlp_packed_seq_params
                else:
                    assert isinstance(psp, (PackedSeqParams, WLBPackedSeqParams))

        # forward_backward_func = get_forward_backward_func()
        pp_size = self.tf_config.pipeline_model_parallel_size
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        n_micro_batch = len(microbatches) - pp_size + 1
        # thd layout
        total_seqlen = microbatches[0]['input_ids'].shape[0]

        def loss_func(output):
            # NOTE: this is a dummy loss function.
            loss = output.mean()
            return loss, {'loss': loss}

        def forward_step(batch_iter, model):
            print(f"🟡 forward_step")
            # import traceback
            # traceback.print_stack()


            batch = next(batch_iter)
            torch.cuda.nvtx.range_push("forward_step")
            input_ids = batch['input_ids']
            position_ids = batch['position_ids']
            attention_mask = None
            packed_seq_params = batch['packed_seq_params']
            # returns "hidden_states" if not model.post_process (not the last layer)
            # returns "logits" when label is None.
            output = gptmodel_forward(
                model, input_ids, attention_mask, position_ids, self.tf_config.sequence_parallel,
                packed_seq_params, 
                # returns "hidden_states" if not model.post_process (not the last layer)
                # returns "logits" when label is None.
                labels=None, 
                # labels=input_ids.unsqueeze(0),
            )
            torch.cuda.nvtx.range_pop()
            return output, loss_func

        def dummy_backward_step(model, dummy_bwd_iter, skip: bool):
            next_iter_args = next(dummy_bwd_iter)
            if skip:
                return
            unwrap_model(model).dummy_backward(next_iter_args)

        assert len(self.train_module) == 1, "only support one module"

        # shift bwd metadata since the order it runs is different from the
        # corresponding dummy forward's.
        dummy_bwd_packed_seq_params = [
            microbatch['packed_seq_params'] for microbatch in
            (microbatches[-pp_size + pp_rank + 1:][:pp_size - pp_rank - 1] + microbatches[:pp_rank])
        ]
        dummy_bwd_packed_seq_params = dummy_bwd_packed_seq_params[pp_rank:] + dummy_bwd_packed_seq_params[:pp_rank]

        assert mode in ["ping_pong", "orig_reimpl", "single_sided"]

        for module in self.train_module:
            debug = (mode != "ping_pong")
            debug_fwd_impl = mode if debug else None
            unwrap_model(module).set_debug(debug=debug, debug_fwd_impl=debug_fwd_impl)
            unwrap_model(module).train()
        assert len(self.train_module) == 1, "only support one module"

        dummy_bwd_packed_seq_params_iter = iter(dummy_bwd_packed_seq_params)
        batch_generator = make_batch_generator(
            microbatches if with_dummy else microbatches[pp_rank:],
            vpp_size=len(self.train_module)
        )
        # if mpu.get_pipeline_model_parallel_world_size() > 1:

        torch.cuda.synchronize()
        from d2.runtime.attn_kernels.ops import nvshmem_barrier_all
        nvshmem_barrier_all()
        if with_dummy:
            raise NotImplementedError("Dummy backward is not implemented for WLBLLM")
            
        # orig_fwd_backward_func = get_forward_backward_func()
        import wlbllm.megatron_patch.pp_schedules
        orig_fwd_backward_func = wlbllm.megatron_patch.pp_schedules.forward_backward_pipelining_without_interleaving
        print(f"🟡 orig_fwd_backward_func: {orig_fwd_backward_func}")
        print(f"🟡 orig_fwd_backward_func location: {orig_fwd_backward_func.__module__}.{orig_fwd_backward_func.__name__}")

        losses_reduced = orig_fwd_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.train_module,
            num_microbatches=n_micro_batch,
            seq_length=total_seqlen,  # no use, since variable_seq_lengths=True
            micro_batch_size=1,  # no use when input_shapes was set
            forward_only=forward_only,
        )
        grad_sample = unwrap_model(self.train_module[0]).decoder.layers[-1].self_attention.linear_proj.weight.main_grad.clone()

        # when testing numerical correctness, instead of running optimizer step, reset grads.
        for tm in self.train_module:
            for param in unwrap_model(tm).parameters():
                param.main_grad.zero_()
        return losses_reduced, grad_sample


def init_megatron_e2e_test(
    world_size: int, cp_size: int, tp_size: int, pp_size: int, dp_size: int, worker_cls=MegatronE2eWorker
):
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        context_parallel_size=cp_size,
    )
    rank = int(os.environ.get("RANK"))
    worker = worker_cls(rank, world_size)
    worker.init_comm(parallel_config)

    dp_rank = mpu.get_data_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    cp_rank = mpu.get_context_parallel_rank()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    print(f"[Rank {rank}] WLBLLM Communication groups initialized: {dp_rank=}, {pp_rank=}, {cp_rank=}, {tp_rank=}, {worker.as_rank=}, {worker.as_world_size=}")
    return worker


from global_batch_provider import setup_global_batch, get_next_batch
from dataclasses import dataclass

def flatten(a):
    return [y for x in a for y in x]

def pad_doc_lens(doc_lens: list[int], cp_size: int, dp_size: int):
    if len(doc_lens) < dp_size:
        # Pad the doc_lens to dp_size
        doc_lens += [512] * (dp_size - len(doc_lens))
        pass
    # TODO(HACK): This is a hack to ensure the doc_lens is divisible by cp_size*2.
    if sum(doc_lens) % (cp_size * 2 * 8) != 0:
        sum_of_doc_lens = sum(doc_lens)
        doc_lens[-1] += (cp_size * 2 * 8) - sum_of_doc_lens % (cp_size * 2 * 8)
        # assert doc_lens[-1] > 0
        pass
    assert sum(doc_lens) % (cp_size * 2 * 8) == 0, f"sum(doc_lens)={sum(doc_lens)} must be divisible by {cp_size * 2 * 8}"
    assert sum(doc_lens) % (cp_size * 2) == 0, f"sum(doc_lens)={sum(doc_lens)} must be divisible by {cp_size * 2}"
    return doc_lens




def test(args):
    os.environ["WLBLLM_MODE"] = "1"
    seed = args.seed

    batch_size = args.num_batches
    num_microbatch = args.num_microbatch
    
    # Setup model configs
    model_path = args.model_path
    num_layers = args.num_layers
    dtype = torch.bfloat16
    element_size = dtype.itemsize
    # TODO: (Refactor) This is a hack to set the number of layers. It should be properly set in the HuggingFace config, not here
    if num_layers is not None:
        # See `megatron_test_utils.py` for more details.
        os.environ["NUM_LAYERS"] = str(num_layers)
    
    # Setup testing scales
    num_tokens = args.num_tokens   
    total_seq_len = args.num_tokens
    cp_size = args.cp_size
    tp_size = args.tp_size
    pp_size = args.pp_size
    world_size = args.num_nodes * args.num_gpus_per_node
    assert world_size % (tp_size * pp_size * cp_size) == 0

    assert world_size == int(os.environ.get("WORLD_SIZE")), f"world_size: {world_size} != WORLD_SIZE: {os.environ.get('WORLD_SIZE')}"
    dp_size = world_size // (tp_size * pp_size * cp_size)

    # Setup the get batch logic
    # - each batch will contain num_tokens tokens.
    # - if cp is specified, then this num_tokens will be on one CP group.
    setup_global_batch(
        num_tokens
    )

    # Setup the model and configuration
    hf_config = AutoConfig.from_pretrained(model_path)
    hidden_size_q = hf_config.hidden_size

    hidden_size_kv = hidden_size_q
    if hasattr(hf_config, "num_key_value_heads"):
        hidden_size_kv = (hidden_size_kv * hf_config.num_key_value_heads //
                          hf_config.num_attention_heads)

    wlbllm.megatron_patch.dot_product_attention.monkey_patch()
    wlbllm.megatron_patch.backends.monkey_patch()

    worker: MegatronE2eWorker = init_megatron_e2e_test(
        world_size, cp_size, tp_size, pp_size, dp_size, 
        MegatronE2eWorker,
    )
    worker.set_config(dtype=dtype)
    worker.init(model_path, seed=seed)
    # set again to potentially adapt to the ray launch case.
    set_random_seed(seed, set_megatron=False)

    # Check rank correctness
    dp_rank = mpu.get_data_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    cp_rank = mpu.get_context_parallel_rank()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    cp_group = mpu.get_context_parallel_group()

    # Setup the WLBLLM registry
    wlbllm.registry.clear()
    wlbllm.registry.set("cp_group", cp_group)
    wlbllm.registry.set("cp_stream", torch.cuda.current_stream())
    wlbllm.registry.set("global_tensor_length", (num_tokens * cp_size * 2))
    wlbllm.registry.set("num_microbatch", num_microbatch)
    wlbllm.registry.set("forward_cnt", 0)
    wlbllm.registry.set("backward_cnt", 0)
    
    def swap_metadata_fn(counter: int):
        wlb_metadata = wlbllm.registry.get(counter)
        for key, value in wlb_metadata.items():
            print(f"🟡 swap_metadata_fn[{counter}]: swapping {key}")
            wlbllm.registry.set(key, value)
        return wlb_metadata

        
    wlbllm.registry.set("swap_metadata_fn", swap_metadata_fn)

    
    num_microbatches_including_dummy = num_microbatch + (pp_size - 1) # num_warmup_microbatches_including_dummy

    def get_next_batch_including_dummy(batch_size: int, mb_idx: int):
        if mb_idx >= num_microbatch:
            return [128] * batch_size   
        return get_next_batch(batch_size * 2)


    microbatches = []
    for mb_idx in range(num_microbatch): # for mb_idx in range(num_microbatch):        
        _seq_lens: list[list[int]] = get_next_batch_including_dummy(batch_size, mb_idx)
        print(f"🟡 get_next_batch_including_dummy[{mb_idx}]: _seq_lens: {_seq_lens}")
        seq_lens, new_batch = d2.planner.wlb_planner.balance_data_for_wlbllm(
            dp_size, dp_rank, total_seq_len, batch_size, _seq_lens, 
            ENABLE_BALANCED_FLOS_NO_DEFER=True,
            # TODO: (Refactor) This is a hack to pass the model config to the WLBLLM planner.
            model_config=hf_config, 
        )
        doc_lens = flatten(seq_lens)
        print(f"🟡 balance_data_for_wlbllm[{mb_idx}]: doc_lens: {doc_lens}")
        context_length = sum(doc_lens) # maximum possible context length is just the num_tokens
        num_tokens_this_rank = local_context_length = context_length // cp_size
        doc_shards = wlbllm.utils.compute_per_doc_cp_shard_doc_len(
            doc_lens, context_length, cp_size
        )
        (
            cu_seqlens_q_list, cu_seqlens_k_list, 
            max_seqlen_q_list, max_seqlen_k_list, 
            kv_idx_list,
        ) = wlbllm.utils.compute_per_doc_metadate_combined__metadata_only(    
            context_length, 
            doc_lens, 
            doc_shards,
            cp_size, 
            cp_rank, 
            device=torch.cuda.current_device()
        )

        # Only take the number of tokens in this rank
        # seq_lens is already the dp_rank's doc_lens. 
        # so here we only take the cp-shard of it.
        input_ids = torch.randint(10, 1000, (num_tokens_this_rank,))
        position_ids = torch.arange(num_tokens_this_rank)
        
        wlb_metadata = dict(
            doc_lens=doc_lens,
            doc_shards=doc_shards,
            kv_idx_list=kv_idx_list,
            cu_seqlens_q_list=cu_seqlens_q_list,
            cu_seqlens_kv_list=cu_seqlens_k_list,
            max_seqlen_q_list=max_seqlen_q_list,
            max_seqlen_kv_list=max_seqlen_k_list,
            # global_tensor_length: 
            global_tensor_length=(num_tokens * cp_size * 2),
        )
        packed_seq_params = WLBPackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_q_list[-1],
            cu_seqlens_kv=cu_seqlens_k_list[-1],
            max_seqlen_q=max_seqlen_q_list[-1].item(),
            max_seqlen_kv=max_seqlen_k_list[-1].item(),
        )
        packed_seq_params.wlb_metadata = wlb_metadata
        microbatch = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "packed_seq_params": packed_seq_params,
        }
        microbatches.append(microbatch)
        wlbllm.registry.set(mb_idx, wlb_metadata)
        print(f"🟡 wlbllm.registry.set mb_idx: {mb_idx}, wlb_metadata: {wlb_metadata}")

    
    set_random_seed(seed, set_megatron=True)

    time.sleep(2)
    
    print("Prepare to run wlbllm")
    loss, grad = worker.forward_backward_batch(
        microbatches=microbatches,
        forward_only=False,
        mode="orig_reimpl", # actually wlbllm
        with_dummy=False,
        # with_dummy=True,
    )
    
    torch.cuda.synchronize()
    torch.distributed.barrier()
    print("finish wlbllm")
    

    print("=" * 20 + "forward_backward_batch attention server, done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--num-microbatch", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus-per-node", type=int, default=4)
    parser.add_argument("--cp-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=4)
    
    parser.add_argument("--model-path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="./logs/")

    args = parser.parse_args()
    test(args)