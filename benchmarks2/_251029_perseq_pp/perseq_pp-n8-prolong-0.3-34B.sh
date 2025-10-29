
# WLBLLM PerSeq
# - Dataset: Pretrain 
# - Model: 34B 
# - Node: 16

# Global variable overrides (use # $$ syntax)
# $$OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251029_perseq_pp/logs.v1-n8-prolong-0.3-34B
# $$FOLDER_SEPARATOR=1
# $$EXPERIMENT_DISTS=("prolong 0.3")
# $$MAX_SAMPLE_ID=3
# $$NUM_LAYERS=24
# $$CHANGE_LONG_DOC_RATIO=0.0
# $$EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5
# $$MIN_TOLERANCE_FACTOR=0.15
# $$EXPERIMENT_REPEAT_TIMES=1
# $$EXPERIMENT_WARMUP_TIMES=0

# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251029_perseq_pp/salloc_srun.pp_34b.v2.sh --config /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251029_perseq_pp/perseq_pp-n8-prolong-0.3-34B.sh


# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0


# 128k tbs=64 node=16
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
    16   4  8 131072    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    16   4  8 131072    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'

# 256k tbs=32 node=16
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
    16   4  4 262144    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    16   4  4 262144    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'

# 384k tbs=8 node=16
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
    16   1  4 393216    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    16   1  4 393216    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'




# 128k tbs=64 node=32
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
    32   4  8 131072    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    32   4  8 131072    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'

# 256k tbs=32 node=32
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
    32   4  4 262144    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    32   4  4 262144    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'

# 384k tbs=8 node=32
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
    32   1  4 393216    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    32   1  4 393216    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'




# 128k tbs=128 node=64
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
    64   8  8 131072    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    64   8  8 131072    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'

# 256k tbs=32 node=64
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
    64   4  4 262144    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    64   4  4 262144    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'

# 384k tbs=16 node=64
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
    64   2  4 393216    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    64   2  4 393216    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'

# >>>


# # ------------ Stop here ------------