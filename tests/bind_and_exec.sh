#!/usr/bin/env bash
set -euo pipefail

# Resolve rank ids
LR=${LOCAL_RANK:-${SLURM_LOCALID:-0}}
RID=${RANK:-$LR}
HOST=${HOSTNAME:-$(hostname)}

# ---- Per-rank logfile ----
LOGDIR=${LOG_DIR:-./logs}
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/${HOST}.r${RID}.lr${LR}.log"
echo "Logging $RID to $LOGFILE"

# ---- Logging + exec ----
exec > >(tee -a "$LOGFILE") 2>&1

# ---- CPU bind (edit slices as you like) ----
CORE_MAP=( "0-15" "16-31" "32-47" "48-63" "64-79" "80-95" "96-111" "112-127" )
CORES=${CORE_MAP[$LR]}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# If you want, set the CUDA device explicitly:
# python will usually set this, but being explicit doesn't hurt:
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  # optional

exec taskset -c "$CORES" "$@"