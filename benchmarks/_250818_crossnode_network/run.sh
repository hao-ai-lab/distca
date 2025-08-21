mkdir -p nsys-profile
HOSTNAME=$(hostname)
set -x
# nsys profile \
#     --sample=none \
#     --trace=cuda,nvtx \
#     --force-overwrite=true \
#     --output=nsys-profile/all_gather.H${HOSTNAME}.N${1}.nsys \
    torchrun --nnodes=$1 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=$2 --rdzv_id=unique_id main.py
set +x


# bash run.sh 2 fs-mbz-gpu-240:29500
# bash run.sh 4 fs-mbz-gpu-240:29500