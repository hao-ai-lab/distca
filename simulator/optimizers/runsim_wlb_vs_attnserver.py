# %%

import wlbllm as wlbllm
import attnserver as attnserver

from attnserver import AttnServerSolver
from wlbllm import WlbLlmSolver


# %%
K = 1024
# batch = [64, 32, 32, 32, 32, 10]
batch = [64, 64, 32, 32, 1, 1, 2, 2, 3, 3,]
batch =  [i * K for i in batch]

num_workers = 4
num_total_devices = 16

print("Qwen3-235B")

# %%
solver = WlbLlmSolver()
solution = solver.solve(
    batch, 
    max_length=sum(batch), 
    num_workers=num_workers, 
    parallel_plan=(8, 2),
)
solution.print_solution()

# %%
solver = AttnServerSolver()
solution = solver.solve(
    batch, 
    num_workers=num_total_devices, 
    num_total_devices=num_total_devices,
)
solution.print_solution()


# %%
