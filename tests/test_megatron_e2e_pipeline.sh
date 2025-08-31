

# 🟢 Passed (D2)
# DP1PP2CP1TP4 BS4
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 torchrun --nnodes 1 --nproc_per_node 8 test_megatron_e2e_pipeline.py --num-gpus-per-node 8 --pp-size 2 --tp-size 4 --num-tokens 16384 --num-microbatch 4 --use-planner
# DP1PP1CP1TP8 BS1
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 torchrun --nnodes 1 --nproc_per_node 8 test_megatron_e2e_pipeline.py --num-gpus-per-node 8 --pp-size 1 --tp-size 8 --num-tokens 16384 --num-microbatch 1 --use-planner
# DP1PP1CP1TP8 BS4
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 torchrun --nnodes 1 --nproc_per_node 8 test_megatron_e2e_pipeline.py --num-gpus-per-node 8 --pp-size 1 --tp-size 8 --num-tokens 16384 --num-microbatch 1 --use-planner


# 🟢 Passed (WLBLLM)
# DP2PP1CP1TP4 BS4
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 torchrun --nnodes 1 --nproc_per_node 8 test_megatron_e2e_pipeline.py --num-gpus-per-node 8 --pp-size 1 --tp-size 4 --num-tokens 16384 --num-microbatch 2 --use-planner --mode wlbllm

# DP2PP2CP1TP2 BS2
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 torchrun --nnodes 1 --nproc_per_node 8 test_megatron_e2e_pipeline.py --num-gpus-per-node 8 --pp-size 2 --tp-size 2 --num-tokens 16384 --num-microbatch 4 --use-planner --mode wlbllm


# 🟡 Running
# ⚪ Ready


