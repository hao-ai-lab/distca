#!/bin/bash

set -e

# Get current directory of this script
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create backup filename with timestamp
# /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/config_sweep_memory.config.v3_0926_0237PST.sh
TIMESTAMP=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)_PST
ORIGINAL_CONFIG="$CURDIR/config_sweep_memory.config.v4_0926_1600PST.sh"
BACKUP_CONFIG="$CURDIR/config_sweep_memory.config.v4_0926_1600PST.sh.backup_${TIMESTAMP}"
UPDATED_CONFIG="$CURDIR/config_sweep_memory.config.v4_0926_1600PST.sh.updated_${TIMESTAMP}"

echo "🟡 Config Status Updater"
echo "📁 Original config: $ORIGINAL_CONFIG"
echo "💾 Backup will be: $BACKUP_CONFIG"
echo "✨ Updated config will be: $UPDATED_CONFIG"
echo ""

# Copy original to backup
cp "$ORIGINAL_CONFIG" "$BACKUP_CONFIG"
echo "✅ Created backup: $BACKUP_CONFIG"

# Source the functions we need from the main script
export OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v6-sweep-pp-34b

# Function to analyze the status of a run directory (copied from main script)
analyze_run_status() {
    local run_dir=$1
    local line_count=""
    
    # Check for success indicator
    if [ -f "$run_dir/benchmark.json" ]; then
        # Count lines in benchmark.raw.jsonl if it exists
        if [ -f "$run_dir/benchmark.raw.jsonl" ]; then
            line_count=$(wc -l < "$run_dir/benchmark.raw.jsonl" 2>/dev/null || echo "0")
            echo "SUCCESS:$line_count"
        else
            echo "SUCCESS:0"
        fi
        return
    fi
    
    # Check for partial success indicator
    if [ -f "$run_dir/benchmark.raw.jsonl" ]; then
        line_count=$(wc -l < "$run_dir/benchmark.raw.jsonl" 2>/dev/null || echo "0")
        echo "PARTIAL:$line_count"
        return
    fi
    
    # Check for OOM error in logs
    if [ -d "$run_dir/logs" ]; then
        # Use find to avoid shell globbing issues and limit scope
        if find "$run_dir/logs" -name "*.log" -type f -exec grep -l "OutOfMemoryError" {} + 2>/dev/null | head -1 | grep -q .; then
            echo "OOM"
            return
        fi
    fi
    
    # Check if run started (has README.md)
    if [ -f "$run_dir/README.md" ]; then
        # Check for MAYBE_STUCK condition using job liveness and log update times
        liveness_file="$run_dir/job_liveness.txt"
        
        if [ -f "$liveness_file" ]; then
            # Get the last liveness timestamp
            last_liveness_time=$(cat "$liveness_file" 2>/dev/null || echo "0")
            current_time=$(date +%s)
            liveness_age_seconds=$((current_time - last_liveness_time))
            
            # If liveness timestamp is recent (within 30 seconds), job is likely still alive
            if [ $liveness_age_seconds -lt 30 ]; then
                # Job is alive, check log update times
                if [ -d "$run_dir/logs" ]; then
                    # Find the most recent log file update
                    latest_log_time=$(find "$run_dir/logs" -name "*.log" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
                    
                    if [ ! -z "$latest_log_time" ]; then
                        log_age_seconds=$(echo "$current_time - $latest_log_time" | bc -l)
                        log_age_minutes=$(echo "$log_age_seconds / 60" | bc -l)
                        
                        # If logs haven't been updated for >5 minutes while job is alive, it's stuck
                        if (( $(echo "$log_age_minutes > 5" | bc -l) )); then
                            echo "MAYBE_STUCK:${log_age_minutes%.*}min"
                            return
                        fi
                    fi
                fi
                
                # Job is alive and logs are recent, or no logs yet
                echo "STARTED"
                return
            else
                # Job liveness is old, so job is probably dead - just started then died
                echo "STARTED"
                return
            fi
        else
            # No liveness file, fall back to basic STARTED status
            echo "STARTED"
            return
        fi
    fi
    
    echo "UNKNOWN"
}

# Function to find existing run folders (copied from main script)
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
    local search_pattern="*.${mode}-n${nnodes}-t${num_tokens}-b${batch_size}-mb${microbatch_size}-cp${cp_size}tp${tp_size}pp${pp_size}-${model_path_normalized}"
    
    # Search for matching directories
    if [ -d "$OUTPUT_DIR_PREFIX" ]; then
        find "$OUTPUT_DIR_PREFIX" -maxdepth 1 -type d -name "$search_pattern" 2>/dev/null | sort
    fi
}

# Function to analyze all runs for a config and determine overall status
analyze_config_status() {
    local mode=$1
    local nnodes=$2
    local num_tokens=$3
    local batch_size=$4
    local microbatch_size=$5
    local cp_size=$6
    local tp_size=$7
    local pp_size=$8
    local model_path_normalized=$9
    
    local existing_runs=$(find_existing_runs "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized")
    
    if [ -z "$existing_runs" ]; then
        echo "NO_RUNS|"
        return
    fi
    
    local has_success=0
    local has_partial=0
    local has_oom=0
    local has_maybe_stuck=0
    local has_other_failure=0
    local total_runs=0
    
    # Track representative directories for each status type
    local success_dir=""
    local partial_dir=""
    local oom_dir=""
    local maybe_stuck_dir=""
    local other_failure_dir=""
    
    while IFS= read -r run_dir; do
        if [ -z "$run_dir" ]; then continue; fi
        total_runs=$((total_runs + 1))
        
        # Extract just the directory name (not full path)
        local dir_name=$(basename "$run_dir")
        
        local status=$(analyze_run_status "$run_dir")
        case $status in
            SUCCESS:*)
                has_success=1
                if [ -z "$success_dir" ]; then
                    success_dir="$dir_name"
                fi
                ;;
            PARTIAL:*)
                has_partial=1
                if [ -z "$partial_dir" ]; then
                    partial_dir="$dir_name"
                fi
                ;;
            OOM)
                has_oom=1
                if [ -z "$oom_dir" ]; then
                    oom_dir="$dir_name"
                fi
                ;;
            MAYBE_STUCK:*)
                has_maybe_stuck=1
                if [ -z "$maybe_stuck_dir" ]; then
                    maybe_stuck_dir="$dir_name"
                fi
                ;;
            STARTED|UNKNOWN)
                has_other_failure=1
                if [ -z "$other_failure_dir" ]; then
                    other_failure_dir="$dir_name"
                fi
                ;;
        esac
    done <<< "$existing_runs"
    
    # Decision logic and return status with representative directory:
    # 1. If any run succeeded -> PASS
    # 2. If any run has partial results (but no success) -> PARTIAL
    # 3. If all runs are OOM -> FAIL_OOM
    # 4. If all runs are maybe stuck -> FAIL_MAYBE_STUCK
    # 5. If all runs failed for other reasons -> FAIL_OTHER
    # 6. Mixed failures -> FAIL_MIXED
    
    if [ $has_success -eq 1 ]; then
        echo "PASS|$success_dir"
    elif [ $has_partial -eq 1 ]; then
        echo "PARTIAL|$partial_dir"
    elif [ $has_oom -eq 1 ] && [ $has_maybe_stuck -eq 0 ] && [ $has_other_failure -eq 0 ]; then
        echo "FAIL_OOM|$oom_dir"
    elif [ $has_maybe_stuck -eq 1 ] && [ $has_oom -eq 0 ] && [ $has_other_failure -eq 0 ]; then
        echo "FAIL_MAYBE_STUCK|$maybe_stuck_dir"
    elif [ $has_oom -eq 0 ] && [ $has_maybe_stuck -eq 0 ] && [ $has_other_failure -eq 1 ]; then
        echo "FAIL_OTHER|$other_failure_dir"
    else
        # For mixed failures, prefer directories in this order: oom > maybe_stuck > other_failure
        local mixed_dir=""
        if [ -n "$oom_dir" ]; then
            mixed_dir="$oom_dir"
        elif [ -n "$maybe_stuck_dir" ]; then
            mixed_dir="$maybe_stuck_dir"
        elif [ -n "$other_failure_dir" ]; then
            mixed_dir="$other_failure_dir"
        fi
        echo "FAIL_MIXED|$mixed_dir"
    fi
}

# Function to update env_var with status
update_env_var_with_status() {
    local env_var="$1"
    local status="$2"
    local result_dir="$3"
    
    # Remove existing EXPERIMENT__STATUS, EXPERIMENT__FAIL_REASON, and RESULT_DIR if present
    env_var=$(echo "$env_var" | sed 's/,*EXPERIMENT__STATUS=[^,]*//g')
    env_var=$(echo "$env_var" | sed 's/,*EXPERIMENT__FAIL_REASON=[^,]*//g')
    env_var=$(echo "$env_var" | sed 's/,*RESULT_DIR=[^,]*//g')
    
    # Add new status (handle empty env_var case to avoid leading comma)
    local separator=""
    if [[ -n "$env_var" && "$env_var" != "'" ]]; then
        separator=","
    fi
    
    # Build the status addition string, handling commas properly
    local status_addition=""
    local result_dir_addition=""
    if [[ -n "$result_dir" ]]; then
        result_dir_addition=",RESULT_DIR=$result_dir"
    fi
    
    case $status in
        PASS)
            status_addition="EXPERIMENT__STATUS=PASS${result_dir_addition}"
            ;;
        PARTIAL)
            status_addition="EXPERIMENT__STATUS=PARTIAL${result_dir_addition}"
            ;;
        FAIL_OOM)
            status_addition="EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM${result_dir_addition}"
            ;;
        FAIL_MAYBE_STUCK)
            status_addition="EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=MAYBE_STUCK${result_dir_addition}"
            ;;
        FAIL_OTHER|FAIL_MIXED)
            status_addition="EXPERIMENT__STATUS=FAIL${result_dir_addition}"
            ;;
    esac
    
    # Now build the final env_var
    if [[ "$env_var" == *"'" ]]; then
        env_var="${env_var%\'*}${separator}${status_addition}'"
    else
        env_var="$env_var${separator}${status_addition}"
    fi
    
    echo "$env_var"
}

# Process the config file
echo "🔄 Processing config file..."
total_lines=0
updated_lines=0
skipped_lines=0
config_line_number=0

while IFS= read -r line; do
    total_lines=$((total_lines + 1))
    config_line_number=$((config_line_number + 1))
    echo L#{$config_line_number} $line
    
    # Skip empty lines, comments, and lines already processed
    if [[ -z "$line" || "$line" =~ ^[[:space:]]*$ || "$line" =~ ^[[:space:]]*#.*$ ]]; then
        echo "$line" >> "$UPDATED_CONFIG"
        continue
    fi
    
    # Check if line already has status
    if [[ "$line" == *"EXPERIMENT__STATUS="* ]]; then
        echo "$line" >> "$UPDATED_CONFIG"
        skipped_lines=$((skipped_lines + 1))
        continue
    fi
    
    # Try to parse the config line
    # Remove comments first
    config_line="$line"
    if [[ "$config_line" == *"#"* ]]; then
        config_line=$(echo "$config_line" | sed 's/#.*$//')
    fi
    
    # Parse config parameters
    if read -r nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size comment env_var <<< "$config_line"; then
        # Validate that we have the minimum required fields
        if [[ -n "$nnodes" && -n "$batch_size" && -n "$microbatch_size" && -n "$num_tokens" && -n "$mode" && -n "$cp_size" && -n "$pp_size" && -n "$tp_size" ]]; then
            
            # Create model_path_normalized (assuming codellama/CodeLlama-34b-hf)
            model_path_normalized="codellama_CodeLlama-34b-hf"
            
            echo "  📊 Analyzing ($ORIGINAL_CONFIG:$config_line_number): $mode n=$nnodes bs=$batch_size mb=$microbatch_size t=$num_tokens cp=$cp_size pp=$pp_size tp=$tp_size"
            
            # Analyze the config status - returns "STATUS|RESULT_DIR"
            status_result=$(analyze_config_status "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized")
            
            # Parse status and result directory
            status=$(echo "$status_result" | cut -d'|' -f1)
            result_dir=$(echo "$status_result" | cut -d'|' -f2)
            
            case $status in
                NO_RUNS)
                    echo "    ❓ No runs found - keeping original line"
                    echo "$line" >> "$UPDATED_CONFIG"
                    ;;
                PASS)
                    echo "    ✅ Found successful runs - marking as PASS (example: $result_dir)"
                    updated_env_var=$(update_env_var_with_status "$env_var" "PASS" "$result_dir")
                    updated_line=$(echo "$line" | sed "s|$env_var|$updated_env_var|")
                    echo "$updated_line" >> "$UPDATED_CONFIG"
                    updated_lines=$((updated_lines + 1))
                    ;;
                PARTIAL)
                    echo "    🟡 Found partial results (no full success) - marking as PARTIAL (example: $result_dir)"
                    updated_env_var=$(update_env_var_with_status "$env_var" "PARTIAL" "$result_dir")
                    updated_line=$(echo "$line" | sed "s|$env_var|$updated_env_var|")
                    echo "$updated_line" >> "$UPDATED_CONFIG"
                    updated_lines=$((updated_lines + 1))
                    ;;
                FAIL_OOM)
                    echo "    🔴 All runs failed with OOM - marking as FAIL with OOM reason (example: $result_dir)"
                    updated_env_var=$(update_env_var_with_status "$env_var" "FAIL_OOM" "$result_dir")
                    updated_line=$(echo "$line" | sed "s|$env_var|$updated_env_var|")
                    echo "$updated_line" >> "$UPDATED_CONFIG"
                    updated_lines=$((updated_lines + 1))
                    ;;
                FAIL_MAYBE_STUCK)
                    echo "    🟡 All runs appear stuck - marking as FAIL with MAYBE_STUCK reason (example: $result_dir)"
                    updated_env_var=$(update_env_var_with_status "$env_var" "FAIL_MAYBE_STUCK" "$result_dir")
                    updated_line=$(echo "$line" | sed "s|$env_var|$updated_env_var|")
                    echo "$updated_line" >> "$UPDATED_CONFIG"
                    updated_lines=$((updated_lines + 1))
                    ;;
                FAIL_OTHER|FAIL_MIXED)
                    echo "    🔴 All runs failed - marking as FAIL (example: $result_dir)"
                    updated_env_var=$(update_env_var_with_status "$env_var" "FAIL_OTHER" "$result_dir")
                    updated_line=$(echo "$line" | sed "s|$env_var|$updated_env_var|")
                    echo "$updated_line" >> "$UPDATED_CONFIG"
                    updated_lines=$((updated_lines + 1))
                    ;;
            esac
        else
            echo "    ⚠️  Could not parse config line $config_line_number - keeping original"
            echo "$line" >> "$UPDATED_CONFIG"
        fi
    else
        echo "    ⚠️  Could not parse config line $config_line_number - keeping original"
        echo "$line" >> "$UPDATED_CONFIG"
    fi
    
done < "$ORIGINAL_CONFIG"

echo ""
echo "✅ Processing complete!"
echo "📊 Summary:"
echo "   Total lines processed: $total_lines"
echo "   Lines updated: $updated_lines"
echo "   Lines skipped (already had status): $skipped_lines"
echo ""
echo "📁 Files created:"
echo "   Backup: $BACKUP_CONFIG"
echo "   Updated: $UPDATED_CONFIG"
echo ""
echo "🔍 To review changes:"
echo "   diff $ORIGINAL_CONFIG $UPDATED_CONFIG"
echo ""
echo "📝 To apply changes:"
echo "   cp $UPDATED_CONFIG $ORIGINAL_CONFIG"
