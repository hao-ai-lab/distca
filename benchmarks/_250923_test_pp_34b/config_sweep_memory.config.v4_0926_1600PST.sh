

# Global variable overrides (use # $$ syntax)
# $$OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v6-sweep-pp-34b--small-scale-optimal-search
# $$FOLDER_SEPARATOR=1


# ----------------------------------------------------------------------------
#      Ensure batch logic is correct and aligned (2025/09/26 16:00:00 PM PST)
# ----------------------------------------------------------------------------
# <<< EXPERIMENT__SKIP=1,EXPERIMENT__COMPLETED=1,MAX_SAMPLE_ID=5,EXPERIMENT_REPEAT_TIMES=1
# Result: Confirmed that batch logic is correct and aligned.
    64   4   4 393216    d2       64  1  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'
    64   2   8 393216  wlbllm     16  1  8 'some_comment'  ''
# >>>

# <<< EXPERIMENT__SKIP=1,EXPERIMENT__COMPLETED=1,MAX_SAMPLE_ID=5,EXPERIMENT_REPEAT_TIMES=1,SHOULD_ADD_DEBUG_CASES=1
# Result: Confirmed that batch logic is correct and aligned with debug cases.
    64   4   4 393216    d2       64  1  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'
    64   2   8 393216  wlbllm     16  1  8 'some_comment'  ''
# >>>

# -------------------------------------------------------------------------------
#      Align basic cases - debug case (2025/09/26 16:20:00 PM PST)
# -------------------------------------------------------------------------------
# <<< EXPERIMENT__SKIP=1,EXPERIMENT__COMPLETED=1,MAX_SAMPLE_ID=1,EXPERIMENT_REPEAT_TIMES=1,SHOULD_ADD_DEBUG_CASES=1
# Result: Confirm that in this case, D2 is indeed better than WLBLLM, with PP and with some wlb DPCP logic.

    # 102000ms
    64   2   8 393216  d2         32  2  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,RESULT_DIR=26_163355.d2-n64-t393216-b2-mb8-cp32tp8pp2-codellama_CodeLlama-34b-hf-tol0.15'

    # 109880ms
    64   2   8 393216  wlbllm     16  2  8 'some_comment'  'RESULT_DIR=26_164155.wlbllm-n64-t393216-b2-mb8-cp16tp8pp2-codellama_CodeLlama-34b-hf'
# >>>

# -------------------------------------------------------------------------------
#      Align basic cases - Try smaller num layers (2025/09/26 16:20:00 PM PST)
# -------------------------------------------------------------------------------
# <<< EXPERIMENT__SKIP=1,EXPERIMENT__COMPLETED=1,MAX_SAMPLE_ID=1,EXPERIMENT_REPEAT_TIMES=1,SHOULD_ADD_DEBUG_CASES=1,NUM_LAYERS=8
# Question: Why, if wlbllm is using gradient accumulation, is faster than D2?
#   Should we just disable gradient accumulation in config space searching?

    # 18002ms
    64   2   8 393216    d2       32  2  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'
    
    # 18444ms
    64   2   8 393216  wlbllm     16  2  8 'some_comment'  'RESULT_DIR=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v5-sweep-pp-34b/26_173326.wlbllm-n64-t393216-b2-mb8-cp16tp8pp2-codellama_CodeLlama-34b-hf'

    # 16394ms
    64   2   8 393216  wlbllm     16  1  8 'some_comment'  'RESULT_DIR=26_165926.wlbllm-n64-t393216-b2-mb8-cp16tp8pp1-codellama_CodeLlama-34b-hf'
    
    # 17499ms
    64   2   8 393216    d2       64  1  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,RESULT_DIR=26_165407.d2-n64-t393216-b2-mb8-cp64tp8pp1-codellama_CodeLlama-34b-hf-tol0.15'
    
# >>>

# -------------------------------------------------------------------------------
#      Nsys Analysis (2025/09/26 17:40:00 PM PST)
# -------------------------------------------------------------------------------
# <<< EXPERIMENT__SKIP=1,EXPERIMENT__COMPLETED=1,MAX_SAMPLE_ID=1,EXPERIMENT_REPEAT_TIMES=1,SHOULD_ADD_DEBUG_CASES=1,NUM_LAYERS=8,ENABLE_NSYS=1
    # 16402ms
    64   2   8 393216  wlbllm     16  1  8 'some_comment'  'RESULT_DIR=26_165926.wlbllm-n64-t393216-b2-mb8-cp16tp8pp1-codellama_CodeLlama-34b-hf'

    # 17581ms
    64   2   8 393216    d2       64  1  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,RESULT_DIR=26_165407.d2-n64-t393216-b2-mb8-cp64tp8pp1-codellama_CodeLlama-34b-hf-tol0.15,RESULT_DIR=26_174354.d2-n64-t393216-b2-mb8-cp64tp8pp1-codellama_CodeLlama-34b-hf-tol0.15'

    # # 18002ms
    # 64   2   8 393216    d2       32  2  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'

    # # 18444ms
    # 64   2   8 393216  wlbllm     16  2  8 'some_comment'  'RESULT_DIR=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v5-sweep-pp-34b/26_173326.wlbllm-n64-t393216-b2-mb8-cp16tp8pp2-codellama_CodeLlama-34b-hf'   
    
    64   2   8 393216    d2       64  1  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,RESULT_DIR=26_165407.d2-n64-t393216-b2-mb8-cp64tp8pp1-codellama_CodeLlama-34b-hf-tol0.15,RESULT_DIR=26_174354.d2-n64-t393216-b2-mb8-cp64tp8pp1-codellama_CodeLlama-34b-hf-tol0.15'
    
# >>>

# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=1,EXPERIMENT_REPEAT_TIMES=1,SHOULD_ADD_DEBUG_CASES=1,NUM_LAYERS=8,ENABLE_NSYS=1
# >>>


# -------------------------------------------------------------------------------
#      384k small OOM sweep on wlbllm and d2 (2025/09/26 21:25:00 PM PST - 23:00:00 PM PST)
# -------------------------------------------------------------------------------
# Debug case section
# <<< EXPERIMENT__SKIP=1,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=1,EXPERIMENT_REPEAT_TIMES=0,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=1,CHECK_TOTAL_BS=16

    # Passed Cases
    64   1  16 393216    d2       32  2  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,EXPERIMENT__STATUS=PASS'
    64   1  16 393216    d2       16  4  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,EXPERIMENT__STATUS=PASS'
    64   2   8 393216    d2       32  2  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,EXPERIMENT__STATUS=PASS'
    64   2   8 393216    d2       16  4  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,EXPERIMENT__STATUS=PASS'
    64   4   4 393216    d2       32  2  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,EXPERIMENT__STATUS=PASS'
    64   2   8 393216  wlbllm     32 2  8 'some_comment'  'EXPERIMENT__STATUS=PASS'
    64   2   8 393216  wlbllm     16 2  8 'some_comment'  'EXPERIMENT__STATUS=PASS'
    64   2   8 393216  wlbllm     8  2  8 'some_comment'  ''
    64   4   4 393216  wlbllm     8  2  8 'some_comment'  'EXPERIMENT__STATUS=PASS'
    64   2   8 393216  wlbllm     8  4  8 'some_comment'  'EXPERIMENT__STATUS=PASS'
    64   2   8 393216  wlbllm     4  4  8 'some_comment'  'EXPERIMENT__STATUS=PASS'
    
    # Non Passing Cases
    64   16  1 393216  wlbllm     64 1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_192334.wlbllm-n64-t393216-b16-mb1-cp64tp8pp1-codellama_CodeLlama-34b-hf' # will oom
    64   16  1 393216  wlbllm     32 1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM'
    64   16  1 393216  wlbllm     16 1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM'
    64  16   1 393216    d2       64  1  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM'
    

    64   4   4 393216  wlbllm     32 2  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM'
    64   4   4 393216  wlbllm     16 2  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM'
    
    64   2   8 393216  wlbllm     16 4  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM'
# >>>


# ----------------------------------------------------------------------------
#      Ensure batch logic is correct and aligned - again (2025/09/26 23:33:00 PM PST)
# ----------------------------------------------------------------------------
# <<< EXPERIMENT__SKIP=1,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=5,EXPERIMENT_REPEAT_TIMES=1,OUTPUT_DIR_SUFFIX_ADDON=-test-batch-logic
# Result: Confirmed that batch logic is correct and aligned.
#    n  bs  mb   t     mode      cp  pp tp    comment    env_var
    64   4   4 393216    d2       32  2  8 'some_comment'  ''
    64   4   4 393216  wlbllm     32  2  8 'some_comment'  ''
# >>>





# -------------------------------------------------------------------------------
#      384k Small Scale Optimal Config sweep (2025/09/26 23:00:00 PM PST)
# -------------------------------------------------------------------------------

# Small scale with 16 layers
# <<< IMPORTANT=1,EXPERIMENT__SKIP=1,EXPERIMENT__COMPLETED=1,MAX_SAMPLE_ID=3,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,CHECK_TOTAL_BS=16,NUM_LAYERS=16
    # Optimal
    64   4   4 393216    d2       32  2  8 '23971ms' 'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,EXPERIMENT_SCHEDULE_PRIORITY=100'
    64   2   8 393216    d2       32  2  8 '23799ms' 'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,EXPERIMENT_SCHEDULE_PRIORITY=100'
    64   2   8 393216  wlbllm     8  2  8 '23751ms'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    64   2   8 393216  wlbllm    16  2  8 '23423ms'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
# >>>

# <<< EXPERIMENT__SKIP=1,EXPERIMENT__COMPLETED=1,MAX_SAMPLE_ID=3,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,CHECK_TOTAL_BS=16,NUM_LAYERS=16
    64   4   4 393216    d2       32  2  8 'some_comment' 'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,EXPERIMENT_SCHEDULE_PRIORITY=60'
    64   1  16 393216    d2       16  4  8 'some_comment' 'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,EXPERIMENT_SCHEDULE_PRIORITY=50'
    64   1  16 393216    d2       32  2  8 'some_comment' 'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,EXPERIMENT_SCHEDULE_PRIORITY=40'
    64   2   8 393216    d2       32  2  8 'some_comment' 'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,EXPERIMENT_SCHEDULE_PRIORITY=30'
    64   2   8 393216    d2       16  4  8 'some_comment' 'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,EXPERIMENT_SCHEDULE_PRIORITY=30'
    
    64   2   8 393216  wlbllm     16 2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
    64   2   8 393216  wlbllm     8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40'
    64   2   8 393216  wlbllm     32 2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
    64   4   4 393216  wlbllm     8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40'
    64   2   8 393216  wlbllm     8  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=30'
    64   2   8 393216  wlbllm     4  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=30'
# >>>

# ----------- Start here -----------------

# -------------------------------------------------------------------------------
#      384k Large Scale Optimal Config sweep (2025/09/27 02:00:00 AM PST)
# -------------------------------------------------------------------------------

# Small scale with full 48 layers

# <<< IMPORTANT=1,EXPERIMENT__SKIP=1,EXPERIMENT__COMPLETED=1,MAX_SAMPLE_ID=3,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,CHECK_TOTAL_BS=16,NUM_LAYERS=16


# >>>

# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=5,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,CHECK_TOTAL_BS=16,NUM_LAYERS=48,OUTPUT_DIR_SUFFIX_ADDON=-full-layers
    # Optimal
    64   4   4 393216    d2       32  2  8 '23971ms' 'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,EXPERIMENT_SCHEDULE_PRIORITY=100'
    64   2   8 393216    d2       32  2  8 '23799ms' 'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,EXPERIMENT_SCHEDULE_PRIORITY=80'
    64   4   4 393216    d2       32  2  8 '23971ms' 'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.125,EXPERIMENT_SCHEDULE_PRIORITY=60'
    64   4   4 393216    d2       32  2  8 '23971ms' 'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.175,EXPERIMENT_SCHEDULE_PRIORITY=60'
    

    64   2   8 393216  wlbllm    16  2  8 '23423ms'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    64   2   8 393216  wlbllm     8  2  8 '23751ms'  'EXPERIMENT_SCHEDULE_PRIORITY=80'
# >>>



# ----------- Stop here -----------------

# -------------------------------------------------------------------------------
#      384k Ablation on wlbllm and d2 (2025/09/26 18:40:00 PM PST)
# -------------------------------------------------------------------------------
# <<< EXPERIMENT__SKIP=1,EXPERIMENT__COMPLETED=1,MAX_SAMPLE_ID=3,EXPERIMENT_REPEAT_TIMES=1,SHOULD_ADD_DEBUG_CASES=1,NUM_LAYERS=8
    64   2   8 393216  wlbllm     16  1  8 'some_comment'  ''
    64   2   8 393216    d2       64  1  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'
# >>>    

# Debug case section
# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=1,EXPERIMENT_REPEAT_TIMES=1,SHOULD_ADD_DEBUG_CASES=1,NUM_LAYERS=8,CHECK_TOTAL_BS=16
    # no gradient accumulation
    # n  bs  mb   t     mode      cp  pp tp    comment    env_var
    64   1  16 393216    d2       32  2  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'
    64   1  16 393216    d2       16  4  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'
    64   2   8 393216    d2       32  2  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'
    64   2   8 393216    d2       16  4  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'
    64   4   4 393216    d2       32  2  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'
    64  16   1 393216    d2       64  1  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,EXPERIMENT_SCHEDULE_PRIORITY=10' # should oom?
    
    64   16  1 393216  wlbllm     64 1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=10,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_192334.wlbllm-n64-t393216-b16-mb1-cp64tp8pp1-codellama_CodeLlama-34b-hf' # will oom
    64   16  1 393216  wlbllm     32 1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=10,EXPERIMENT__STATUS=PASS,RESULT_DIR=26_192604.wlbllm-n64-t393216-b16-mb1-cp32tp8pp1-codellama_CodeLlama-34b-hf' # will oom
    64   16  1 393216  wlbllm     16 1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=10,EXPERIMENT__STATUS=PASS,RESULT_DIR=26_192901.wlbllm-n64-t393216-b16-mb1-cp16tp8pp1-codellama_CodeLlama-34b-hf' # will oom

    64   2   8 393216  wlbllm     32 2  8 'some_comment'  ''
    64   2   8 393216  wlbllm     16 2  8 'some_comment'  ''
    64   2   8 393216  wlbllm     8  2  8 'some_comment'  ''

    64   4   4 393216  wlbllm     32 2  8 'some_comment'  ''
    64   4   4 393216  wlbllm     16 2  8 'some_comment'  ''
    64   4   4 393216  wlbllm     8  2  8 'some_comment'  ''
    
    64   2   8 393216  wlbllm     16 4  8 'some_comment'  ''
    64   2   8 393216  wlbllm     8  4  8 'some_comment'  ''
    64   2   8 393216  wlbllm     4  4  8 'some_comment'  ''
    
    # enable gradient accumulation
    64   1  16 393216    d2       64  1  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'

# >>>

# Non debug case section
# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=5,EXPERIMENT_REPEAT_TIMES=1,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=8,CHECK_TOTAL_BS=16
    # no gradient accumulation
    # n  bs  mb   t     mode      cp  pp tp    comment    env_var
    64   1  16 393216    d2       32  2  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'
    64   1  16 393216    d2       16  4  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'
    64   2   8 393216    d2       32  2  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'
    64   2   8 393216    d2       16  4  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'
    64   4   4 393216    d2       32  2  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'
    64  16   1 393216    d2       64  1  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,EXPERIMENT_SCHEDULE_PRIORITY=10' # should oom?
    
    64   16  1 393216  wlbllm     64 1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=10' # will oom
    64   16  1 393216  wlbllm     32 1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=10' # will oom
    64   16  1 393216  wlbllm     16 1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=10' # will oom

    64   2   8 393216  wlbllm     32 2  8 'some_comment'  ''
    64   2   8 393216  wlbllm     16 2  8 'some_comment'  ''
    64   2   8 393216  wlbllm     8  2  8 'some_comment'  ''

    64   4   4 393216  wlbllm     32 2  8 'some_comment'  ''
    64   4   4 393216  wlbllm     16 2  8 'some_comment'  ''
    64   4   4 393216  wlbllm     8  2  8 'some_comment'  ''
    
    64   2   8 393216  wlbllm     16 4  8 'some_comment'  ''
    64   2   8 393216  wlbllm     8  4  8 'some_comment'  ''
    64   2   8 393216  wlbllm     4  4  8 'some_comment'  ''
    
    # enable gradient accumulation
    64   1  16 393216    d2       64  1  8 'some_comment'  'MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'

# >>>



# ----------- Stop here -----------------





# <<< EXPERIMENT__SKIP=1,EXPERIMENT__COMPLETED=1,MAX_SAMPLE_ID=1
    # 64   2   8 393216    d2       64  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,COMMENT="Slower than having PP"'
    # 64   2   8 393216    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'
    # 64   2   8 393216    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,MIN_TOLERANCE_FACTOR=0.125,OUTPUT_DIR_SUFFIX_ADDON=-tol0.125'
    # 64   2   8 393216  wlbllm     16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
    # 64   2   8 393216    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,MIN_TOLERANCE_FACTOR=0.175,OUTPUT_DIR_SUFFIX_ADDON=-tol0.175'
    # 64   2   8 393216    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,MIN_TOLERANCE_FACTOR=0.10,OUTPUT_DIR_SUFFIX_ADDON=-tol0.10' 
    # 64   2   8 393216    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,MIN_TOLERANCE_FACTOR=0.20,OUTPUT_DIR_SUFFIX_ADDON=-tol0.20'

# >>>

# ------------- Stop here -----------------

