set -e

# -----------------------------
# Command Line Argument Parsing
# -----------------------------
LIST_ONLY=0
LIST_RUNS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --ls)
            LIST_ONLY=1
            shift
            ;;
        --ls-runs)
            LIST_RUNS=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--ls] [--ls-runs] [--dry-run]"
            echo "  --ls        List configs that would be run and exit"
            echo "  --ls-runs   List configs and show existing run folders, then exit"
            echo "  --dry-run   Show what would be run but don't execute"
            exit 1
            ;;
    esac
done

# export JOBID=${JOBID:-710588}
TS=$(TZ=America/Los_Angeles date +%m%d_%H%M%S)_PST
export OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v1-sweep-pp-34b

export STATUS_FILE_PATH=${OUTPUT_DIR_PREFIX}/job-status.txt
mkdir -p ${OUTPUT_DIR_PREFIX}
touch ${STATUS_FILE_PATH}

# Function to analyze the status of a run directory
# Returns: SUCCESS|PARTIAL|STARTED|OOM|UNKNOWN
analyze_run_status() {
    local run_dir=$1
    
    # Check for success indicator
    if [ -f "$run_dir/benchmark.json" ]; then
        echo "SUCCESS"
        return
    fi
    
    # Check for partial success indicator
    if [ -f "$run_dir/benchmark.raw.json" ]; then
        echo "PARTIAL"
        return
    fi
    
    # Check for OOM error in logs
    if [ -d "$run_dir/logs" ]; then
        if grep -r -q "OutOfMemoryError" "$run_dir/logs/"*.log 2>/dev/null; then
            echo "OOM"
            return
        fi
    fi
    
    # Check if run started (has README.md)
    if [ -f "$run_dir/README.md" ]; then
        echo "STARTED"
        return
    fi
    
    echo "UNKNOWN"
}

# Function to get status emoji and description
get_status_display() {
    local status=$1
    case $status in
        "SUCCESS")
            echo "✅ SUCCESS (benchmark.json)"
            ;;
        "PARTIAL")
            echo "🟡 PARTIAL (benchmark.raw.json)"
            ;;
        "STARTED")
            echo "🟠 STARTED (README.md)"
            ;;
        "OOM")
            echo "🔴 OOM (OutOfMemoryError in logs)"
            ;;
        "UNKNOWN")
            echo "❓ UNKNOWN"
            ;;
    esac
}

# Function to find existing run folders for a given config
# Usage: find_existing_runs mode nnodes num_tokens batch_size microbatch_size cp_size tp_size pp_size model_path_normalized
find_existing_runs() {
    local mode=$1
    local nnodes=$2
    local num_tokens=$3
    local batch_size=$4
    local microbatch_size=$5
    local cp_size=$6
    local tp_size=$7
    local pp_size=$8
    local model_path_normalized=$9
    
    # Construct the search pattern (ignoring datetime part)
    # Pattern: <datetime>.<mode>-n<node>-t<tokens>-b<batchsize>-mb<microbatchsize>-cp<cp_size>tp<tpsize>pp<pp_size>-<modelname>
    local search_pattern="*.${mode}-n${nnodes}-t${num_tokens}-b${batch_size}-mb${microbatch_size}-cp${cp_size}tp${tp_size}pp${pp_size}-${model_path_normalized}"
    
    # Search for matching directories
    if [ -d "$OUTPUT_DIR_PREFIX" ]; then
        find "$OUTPUT_DIR_PREFIX" -maxdepth 1 -type d -name "$search_pattern" 2>/dev/null | sort
    fi
}

# -----------------------------
# TorchRun Final Logging Flag (for cleaner console)
# -----------------------------
# Quiet C++/glog + c10d
export TORCH_CPP_LOG_LEVEL=ERROR        # hides INFO/WARNING from libtorch/c10d
export GLOG_minloglevel=2               # 0=INFO,1=WARNING,2=ERROR,3=FATAL
export TORCH_DISTRIBUTED_DEBUG=OFF      # don't spam states
export NCCL_DEBUG=ERROR                 # only errors from NCCL
unset TORCH_SHOW_CPP_STACKTRACES        # disable those C++ frames in tracebacks

# -----------------------------
# Debugging Flags
# -----------------------------
# export ENABLE_NSYS=1
# export EXPERIMENT_PYTHON_DEBUG_TRACE_CALLS=1
export EXPERIMENT_LOG_MEMORY_USAGE=0
export EXPERIMENT_SKIP_FA2A_OP=0 # (DebugOnly: Ensure not stuck at fast a2a op)
export EXPERIMENT_SKIP_OPTIMIZER_STEP=1
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=4
export SHOULD_ADD_DEBUG_CASES=0


# ------------------------
# 
# ------------------------
# export NCCL_CUMEM_ENABLE=0
# export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6"
# export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:512,expandable_segments:True"
# export PYTORCH_CUDA_ALLOC_CONF="verbose_policy:1"

set -e
export EXPERIMENT_FA2A_SPLIT_SENDRECV=1
export EXPERIMENT_SHOULD_LOG_MEMORY_DURING_WARMUP=1
export EXPERIMENT_0TH_SAMPLE_WARMUP_TIMES=1
export EXPERIMENT_WARMUP_TIMES=0
export EXPERIMENT_REPEAT_TIMES=1
# export EXPERIMENT_ADD_SELECTIVE_CKPT=1
# export PYTORCH_CUDA_ALLOC_CONF="pinned_use_background_threads:True"
# export EXPERIMENT_SHOULD_LOG_MEMORY_DURING_WARMUP=1
export MAX_SAMPLE_ID=1
export ENABLE_NSYS=0
export EXPERIMENT_ADD_SELECTIVE_CKPT=1
# export EXPERIMENT_ADD_SELECTIVE_CKPT=0

model_config="codellama/CodeLlama-34b-hf 131072 48"
param_configs_cases=()


# -------------------------------
#  Sweep Memory without OOM
# -------------------------------
# memory sweep
config_sweep_memory=()
export MAX_SAMPLE_ID=1
export ENABLE_NSYS=1
# Get current directory of this script
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Read config file line by line and append to array
while IFS= read -r line; do
    echo "$line"
    if [[ "$line" == *"Stop here"* ]]; then
        break
    fi
    # Skip empty lines (including whitespace-only) and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*$ || "$line" =~ ^[[:space:]]*#.*$ ]] && continue
    # Skip the line that contains "Stop here"
    # sleep 1 
    # if there is `#` comment after the line, trim the comment
    if [[ "$line" == *"#"* ]]; then
        line=$(echo "$line" | sed 's/#.*$//')
    fi
    
    # Skip lines that have EXPERIMENT__STATUS=PASS or EXPERIMENT__STATUS=FAIL (filtering logic moved up)
    if [[ "$line" == *"EXPERIMENT__STATUS=PASS"* ]] || [[ "$line" == *"EXPERIMENT__STATUS=FAIL"* ]]; then
        echo "🟢 Skipping config line due to EXPERIMENT__STATUS flag: $line"
        continue
    fi
    
    config_sweep_memory+=("$line")
done < "$CURDIR/config_sweep_memory.config.sh"

for config in "${config_sweep_memory[@]}"; do
    cases+=("$model_config $config")
done


# for config in "${param_configs_cases[@]}"; do
#     cases+=("$model_config $config")
# done


dists=(
    "wlbllm 0.0"
    # "prolong 0.3"
)

function write_status_log() {
    echo "$1" >> $STATUS_FILE_PATH
}


# -------------------------------
# -------------------------------

wlbllm_cases=()
d2_cases=()
# Define format string once
format_str="%-15s  %5s  %6s  %4s  %4s  %4s  %6s  %7s  %4s  %4s  %4s  %4s %10s\n"

echo "🟡 All cases:"
printf "$format_str" \
    "model_path" "atb" "nlayers" "N" "bs" "mb" "toks" "mode" "cp" "pp" "tp" "comment" "env_var"
for case in "${cases[@]}"; do
    read -r model_path attn_linear_breakpoint num_layers nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size comment <<< "$case"
    # Trim model path to 15 chars
    model_path_short="${model_path:0:15}"
    
    # Create model_path_normalized for folder search
    model_path_normalized=$(echo $model_path | sed 's/\//_/g')
    
    if [ "$mode" == "wlbllm" ]; then
        wlbllm_cases+=("$case")
        printf "$format_str" \
            "$model_path_short" "$attn_linear_breakpoint" "$num_layers" "$nnodes" "$batch_size" "$microbatch_size" "$num_tokens" "$mode" "$cp_size" "$pp_size" "$tp_size" "$comment" "$env_var"
        
        # Show existing runs if --ls-runs is specified
        if [ "$LIST_RUNS" == "1" ]; then
            existing_runs=$(find_existing_runs "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized")
            if [ ! -z "$existing_runs" ]; then
                echo "    📁 Existing runs:"
                while IFS= read -r run_dir; do
                    run_name=$(basename "$run_dir")
                    status=$(analyze_run_status "$run_dir")
                    status_display=$(get_status_display "$status")
                    echo "       $run_name - $status_display"
                done <<< "$existing_runs"
            else
                echo "    📁 No existing runs found"
            fi
            echo ""
        fi
    fi
    if [ "$mode" == "d2" ]; then
        d2_cases+=("$case")
        printf "$format_str" \
            "$model_path_short" "$attn_linear_breakpoint" "$num_layers" "$nnodes" "$batch_size" "$microbatch_size" "$num_tokens" "$mode" "$cp_size" "$pp_size" "$tp_size" "$comment" "$env_var"
        
        # Show existing runs if --ls-runs is specified
        if [ "$LIST_RUNS" == "1" ]; then
            existing_runs=$(find_existing_runs "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized")
            if [ ! -z "$existing_runs" ]; then
                echo "    📁 Existing runs:"
                while IFS= read -r run_dir; do
                    run_name=$(basename "$run_dir")
                    status=$(analyze_run_status "$run_dir")
                    status_display=$(get_status_display "$status")
                    echo "       $run_name - $status_display"
                done <<< "$existing_runs"
            else
                echo "    📁 No existing runs found"
            fi
            echo ""
        fi
    fi
done
echo "🟡 Total wlbllm cases: ${#wlbllm_cases[@]}"
echo "🟡 Total d2 cases: ${#d2_cases[@]}"
echo ""

# Exit if only listing configs
if [ "$LIST_ONLY" == "1" ]; then
    echo "📋 LIST_ONLY mode: Showing configs that would be run and exiting."
    exit 0
fi

# Exit if only listing configs with existing runs
if [ "$LIST_RUNS" == "1" ]; then
    echo "📋 LIST_RUNS mode: Showing configs with existing run folders and exiting."
    exit 0
fi

sleep 3



max_cases=10000
echo "🏁 Start regression sweep. Only running $max_cases cases."
cases_index=0

set -x

for sample_config in "${dists[@]}"; do
for config in "${cases[@]}"; do
    echo config="$config" ";" sample_config="$sample_config" ";"
    if ! read -r model_path attn_linear_breakpoint num_layers nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size comment env_var <<< "$config"; then
        echo "⚠️ Failed to parse config: $config"
        continue
    fi
    if ! read -r sample_name change_long_doc_ratio <<< "$sample_config"; then
        echo "⚠️ Failed to parse sample config: $sample_config" 
        continue
    fi
    
    export MODE=$mode
    export BATCH_SIZE=$batch_size
    export NNODES=$nnodes
    export NUM_TOKENS=$num_tokens
    export NUM_MICROBATCH=$microbatch_size
    export CP_SIZE=$cp_size
    export PP_SIZE=$pp_size
    export TP_SIZE=$tp_size
    export ATTN_LINEAR_BREAKPOINT=$attn_linear_breakpoint
    export NUM_LAYERS=$num_layers
    export MODEL_PATH=$model_path
    export MODEL_PATH_normalized=$(echo $model_path | sed 's/\//_/g')
    export SAMPLE_NAME=$sample_name
    export CHANGE_LONG_DOC_RATIO=$change_long_doc_ratio

    export RATIO=$(echo "8 / ($NNODES / $BATCH_SIZE)" | bc -l)
    export OUTPUT_DIR_SUFFIX_ADDON="-${MODEL_PATH_normalized}"


    export ELONGATE_FACTOR=$(($NUM_TOKENS / 65536))
    if [ $ELONGATE_FACTOR -lt 1 ]; then
        ELONGATE_FACTOR=1
    fi
    
    
    # ----------------------------------
    #  Memory Sweep Pass
    # ----------------------------------
    # Override export environment variables if env_var is not empty
    declare -A old_env_vars  # Associative array to store old values
    env_vars_to_restore=()  # Array to track which variables we modified
    
    if [ ! -z "$env_var" ]; then
        # Remove surrounding quotes if present
        env_var_clean="${env_var//\'/}"  # Remove single quotes
        env_var_clean="${env_var_clean//\"/}"  # Remove double quotes
        
        IFS=',' read -ra ENV_VARS <<< "$env_var_clean"
        for var in "${ENV_VARS[@]}"; do
            IFS='=' read -r key value <<< "$var"
            
            # Save the old value if the variable exists
            if [[ -v "$key" ]]; then
                old_env_vars["$key"]="${!key}"
                echo "⚪ Saving old environment variable: $key=${!key}"
            else
                old_env_vars["$key"]="__UNSET__"
                # echo "⚪ Variable $key was not previously set"
            fi
            env_vars_to_restore+=("$key")
            
            # set -x
            export "$key"="$value"
            # set +x
            echo "🟡 Exporting environment variable: $key=$value"
        done
    fi
    
    
    

    # D2 normal
    echo "🟡 Running config: MODE=$MODE, NNODES=$NNODES, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, NUM_MICROBATCH=$NUM_MICROBATCH, CP_SIZE=$CP_SIZE, PP_SIZE=$PP_SIZE, TP_SIZE=$TP_SIZE, MODE=$MODE, MODEL_PATH=$MODEL_PATH, MODEL_PATH_normalized=$MODEL_PATH_normalized, NUM_LAYERS=$NUM_LAYERS, ATTN_LINEAR_BREAKPOINT=$ATTN_LINEAR_BREAKPOINT, COMMENT=$COMMENT, RATIO=$RATIO"

    if [ "$DRY_RUN" == "1" ]; then
        echo "🔵 DRY_RUN=1, skipping execution"
        continue
    fi

    

    write_status_log "Running: $sample_config $config"
    
    
    if [ $MODE == "d2" ]; then
        bash test_megatron_e2e_pipeline_with_cp.sh
    fi

    if [ $MODE == "wlbllm" ]; then
        bash test_wlb_e2e.sh
    fi

    write_status_log "Finished: $sample_config $config"

    echo "🟡 Finished running config: MODE=$MODE, NNODES=$NNODES, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, NUM_MICROBATCH=$NUM_MICROBATCH, CP_SIZE=$CP_SIZE, PP_SIZE=$PP_SIZE, TP_SIZE=$TP_SIZE, MODE=$MODE, MODEL_PATH=$MODEL_PATH, MODEL_PATH_normalized=$MODEL_PATH_normalized, NUM_LAYERS=$NUM_LAYERS, ATTN_LINEAR_BREAKPOINT=$ATTN_LINEAR_BREAKPOINT, COMMENT=$COMMENT, RATIO=$RATIO"
    
    # Restore original environment variables
    if [ ${#env_vars_to_restore[@]} -gt 0 ]; then
        # echo "⚪ Restoring original environment variables..."
        for key in "${env_vars_to_restore[@]}"; do
            if [ "${old_env_vars[$key]}" == "__UNSET__" ]; then
                unset "$key"
                # echo "⚪ Unsetting environment variable: $key (was not previously set)"
            else
                export "$key"="${old_env_vars[$key]}"
                # echo "⚪ Restored environment variable: $key=${old_env_vars[$key]}"
            fi
        done
    fi


    # ----------------------------------
    #  Performance Sweep Pass
    # ----------------------------------
    # TODO: Add logic to check if the performance run is completed. 
    # if [ "${EXPERIMENT__STATUS:-}" == "FAIL" ]; then
    #     echo "🔴 Skipping performance sweep due to EXPERIMENT__STATUS=FAIL flag"
    #     continue
    # fi

    # OLD_ENABLE_NSYS=$ENABLE_NSYS
    # OLD_MAX_SAMPLE_ID=$MAX_SAMPLE_ID
    # export MAX_SAMPLE_ID=10
    # export ENABLE_NSYS=0

    # if [ $MODE == "d2" ]; then
    #     bash test_megatron_e2e_pipeline_with_cp.sh
    # fi

    # if [ $MODE == "wlbllm" ]; then
    #     bash test_wlb_e2e.sh
    # fi

    # export MAX_SAMPLE_ID=$OLD_MAX_SAMPLE_ID
    # export ENABLE_NSYS=$OLD_ENABLE_NSYS
    

    # ----------------------------------
    #  Epilogue
    # ----------------------------------
    
    cases_index=$((cases_index + 1))
    if [ $cases_index -gt $max_cases ]; then
        break
    fi

done
done

write_status_log "Finished running all cases"

for i in {1..10}; do
    echo '\a'
    sleep 1
done