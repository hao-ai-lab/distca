import torch
import torch.distributed as dist
import os

def setup():
    
    local_rank = os.environ.get("LOCAL_RANK", 0)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend="nccl", 
        device_id=device,
    )
    torch.distributed.all_reduce(torch.rand(1, device=device))     
    torch.cuda.synchronize()
    torch.distributed.barrier()
    print("Done initializing distributed")


    
    # Setup CP communication group.
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_per_node = 8
    local_rank = rank % device_per_node
    torch.cuda.set_device(device)
    nnodes = world_size // device_per_node

    # Create CP communication group.
    _, s = divmod(rank, device_per_node)
    ranks = [s + device_per_node * nid for nid in range(nnodes)]
    print(f"Ranks: {ranks}")
    cp_group = dist.new_group(ranks=ranks)
    return cp_group, device, nnodes, rank


def main():
    
    # Test all gather communication latency.
    cp_group, device, nnodes, rank = setup()
    cp_world_size = dist.get_world_size(cp_group)
    print(f"CP world size: {cp_world_size}")
    print(f"CP group: {cp_group}")
    
    # Create tensor for all gather test
    M = 1024 * 1024  # 1MB tensor
    base_size = 1 * M

    # Open output file on rank 0
    output_file = None
    if rank == 0:
        output_file = open(f"all_gather.N{nnodes}.txt", "w")

    for factor in [1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        print(f"Factor: {factor}")
        dtype = torch.float16
        tensor_size = base_size * factor
        tensor_size_mb = tensor_size * dtype.itemsize / 1024 / 1024
        torch.cuda.nvtx.range_push(f"all_gather.{factor}M.N{nnodes}")
        local_tensor = torch.randn(tensor_size, device=device, dtype=dtype)
        
        # Prepare output tensor list for all gather
        output_tensors = [torch.zeros_like(local_tensor) for _ in range(cp_world_size)]
        
        # Warmup
        for _ in range(10):
            # print(f"Warmup {factor}M.N{nnodes}.actual")
            dist.all_gather(output_tensors, local_tensor, group=cp_group)
            torch.cuda.synchronize()
        
        # Measure latency
        num_iterations = 10
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

    
        torch.cuda.synchronize()
        # dist.barrier()
        start_time.record()
        for _ in range(num_iterations):
            with torch.cuda.nvtx.range(f"all_gather.{factor}M.N{nnodes}.actual"):
                # print(f"All-gather {factor}M.N{nnodes}.actual")
                dist.all_gather(output_tensors, local_tensor, group=cp_group)
        end_time.record()
        torch.cuda.synchronize()

        
        elapsed_time = start_time.elapsed_time(end_time)
        avg_latency = elapsed_time / num_iterations
        torch.cuda.nvtx.range_pop()
        
        
        
        if rank == 0:
            print(f"Tensor size: {tensor_size_mb:.2f} MB")
            print(f"All-gather latency: {avg_latency:.2f} ms")
            print(f"World size: {cp_world_size}")
            
            # Write to file
            if output_file:
                output_file.write(f"Tensor size: {tensor_size_mb:.2f} MB\n")
                output_file.write(f"All-gather latency: {avg_latency:.2f} ms\n")
                output_file.write(f"World size: {cp_world_size}\n")
                output_file.flush()
        
        pass
    
    # Close output file
    if rank == 0 and output_file:
        output_file.close()

if __name__ == "__main__":
    
    main()