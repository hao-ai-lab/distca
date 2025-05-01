import torch
import torch.distributed as dist
import multiprocessing as mp
import cloudpickle
from rich import print

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def print_master(rank, msg):
    if rank == 0:
        print(msg)

def run(rank, size, queue):
    """ Function to run on each process. """
    # Establish the world communication ring
    world_group = dist.new_group(ranks=list(range(size)))

    # Create all possible communication groups
    for i in range(1 << size):
        ranks = [j for j in range(size) if (i & (1 << j))]
        if len(ranks) > 1:  # Only create groups with more than one rank
            dist.new_group(ranks=ranks)
            print_master(rank, f"Created group {ranks}")

    # Master process logic
    if rank == 0:
        # Example: Send a cloudpickle serialized function to be executed
        func = lambda: dist.all_reduce(torch.tensor([1.0]), op=dist.ReduceOp.SUM, group=world_group)
        serialized_func = cloudpickle.dumps(func)
        queue.put(serialized_func)

    # Worker process logic
    while True:
        if not queue.empty():
            serialized_func = queue.get()
            func = cloudpickle.loads(serialized_func)
            func()

def main():
    size = 4  # Number of processes
    processes = []
    queues = [mp.Queue() for _ in range(size)]

    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run, queues[rank]))
        p.start()
        processes.append(p)

    print("All processes started. Waiting for user input.")
    while True:
        try:
            input()
            item = None
            for rank in range(size):
                queues[rank].put(item)
            
        except KeyboardInterrupt:
            break
        except EOFError:
            break

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
