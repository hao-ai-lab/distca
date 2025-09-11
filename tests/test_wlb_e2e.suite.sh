


export JOBID=710588
export ENABLE_NSYS=0
export EXPERIMENT_LOG_MEMORY_USAGE=1


# Define test configurations
declare -a configs=(
    # NNODES  NUM_MB  TP  PP  CP  LAYERS  TOKENS
    # ✅ 
    "  1      2       2   2   2   8       16384"  # ✅ 
    "  2      4       2   4   2   8       16384"  # ✅ 
    "  8      4       8   4   2   8       16384"  # ✅ 
    "  8      4       8   4   2   32      16384"  # ✅ 
    "  16     4       8   4   2   48      16384"  # ✅ 
    "   4     4       1   4   2   8       65536"  # ✅ - labels=None is required
    "  32     4       8   4   2   48      65536"  # ✅
    "  32     4       8   4   8   48      65536"  # ✅
    "  32     8       8   4   8   48      262144"  # ✅
)

# Run tests for each configuration
for config in "${configs[@]}"; do
    read -r nodes mb tp pp cp nlayers num_tokens<<< "$config"
    echo "Running test with: NNODES=$nodes NUM_MICROBATCH=$mb TP_SIZE=$tp PP_SIZE=$pp CP_SIZE=$cp NUM_LAYERS=$nlayers NUM_TOKENS=$num_tokens"
    NNODES=$nodes NUM_MICROBATCH=$mb TP_SIZE=$tp PP_SIZE=$pp CP_SIZE=$cp NUM_LAYERS=$nlayers NUM_TOKENS=$num_tokens bash test_wlb_e2e.sh
    echo "Finished running test with: NNODES=$nodes NUM_MICROBATCH=$mb TP_SIZE=$tp PP_SIZE=$pp CP_SIZE=$cp NUM_LAYERS=$nlayers NUM_TOKENS=$num_tokens"
done