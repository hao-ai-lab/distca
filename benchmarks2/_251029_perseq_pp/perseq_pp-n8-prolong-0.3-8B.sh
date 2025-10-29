
# WLBLLM PerSeq
# - Dataset: Pretrain 
# - Model: 8B 
# - Node: 8

# Global variable overrides (use # $$ syntax)
# $$OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251029_perseq_pp/logs.v1-n8-prolong-0.3-8B
# $$FOLDER_SEPARATOR=1
# $$EXPERIMENT_DISTS=("prolong 0.3")
# $$MAX_SAMPLE_ID=3
# $$SHOULD_ADD_DEBUG_CASES=0
# $$EXPERIMENT_REPEAT_TIMES=1
# $$EXPERIMENT_WARMUP_TIMES=0
# $$NUM_LAYERS=32
# $$ENABLE_NSYS=0

# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251029_perseq_pp/salloc_srun.pp_8b.v2.sh --config /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251029_perseq_pp/perseq_pp-n8-prolong-0.3-8B.sh

# <<< 


# 128k tbs=32 node=8
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
    
     8   4  4 131072    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
     8   4  4 131072    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'


# 256k tbs=16 node=8
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
     8   2  4 262144    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
     8   2  4 262144    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'

# 512k tbs=8 node=8
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
     8   1  4 524288    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
     8   1  4 524288    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'




# 128k tbs=64 node=16
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
    16   4  8 131072    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    16   4  8 131072    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'

# 256k tbs=32 node=16
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
    16   4  4 262144    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    16   4  4 262144    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'

# 512k tbs=8 node=16
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
    16   1  4 524288    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    16   1  4 524288    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'




# 128k tbs=128 node=32
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
    32   8  8 131072    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    32   8  8 131072    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'

# 256k tbs=32 node=32
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
    32   4  4 262144    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    32   4  4 262144    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'

# 512k tbs=16 node=32
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
    32   2  4 524288    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    32   2  4 524288    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'
# >>>


# # ------------ Stop here ------------