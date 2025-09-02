set -x
# ⚪ Running
for num_tokens in 262144 131072 65536 524288 1048576; do
    for nodes in 32 16 8 4 2 1; do
        for batch_size in 1 2 4 8; do
            for cp_size in 32 16 8 4 2 1; do
                if [ $cp_size -gt $nodes ]; then
                    continue
                fi
                MODE=wlbllm BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=100 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=4 sbatch --nodes $nodes test_e2e_combined.slurm.sh
                sleep 2
            done
            MODE=d2 BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=100 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=4 sbatch --nodes $nodes test_e2e_combined.slurm.sh
            sleep 2
        done
    done
done
set +x