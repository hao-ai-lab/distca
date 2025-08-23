nsys profile -t nvtx,cuda -w true -o data-hetero/cp1.65536.1024.bs1.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 1 --data_dist 1lns --batch_size 1
nsys profile -t nvtx,cuda -w true -o data-hetero/cp2.65536.1024.bs1.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 2 --data_dist 1lns --batch_size 1
nsys profile -t nvtx,cuda -w true -o data-hetero/cp4.65536.1024.bs1.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 4 --data_dist 1lns --batch_size 1
nsys profile -t nvtx,cuda -w true -o data-hetero/cp8.65536.1024.bs1.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 8 --data_dist 1lns --batch_size 1
nsys profile -t nvtx,cuda -w true -o data-hetero/cp16.65536.1024.bs1.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 16 --data_dist 1lns --batch_size 1
nsys profile -t nvtx,cuda -w true -o data-hetero/cp32.65536.1024.bs1.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 32 --data_dist 1lns --batch_size 1
nsys profile -t nvtx,cuda -w true -o data-hetero/cp1.65536.1024.bs32.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 1 --data_dist 1lns --batch_size 32
nsys profile -t nvtx,cuda -w true -o data-hetero/cp2.65536.1024.bs32.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 2 --data_dist 1lns --batch_size 32
nsys profile -t nvtx,cuda -w true -o data-hetero/cp4.65536.1024.bs32.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 4 --data_dist 1lns --batch_size 32
nsys profile -t nvtx,cuda -w true -o data-hetero/cp8.65536.1024.bs32.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 8 --data_dist 1lns --batch_size 32
nsys profile -t nvtx,cuda -w true -o data-hetero/cp16.65536.1024.bs32.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 16 --data_dist 1lns --batch_size 32
nsys profile -t nvtx,cuda -w true -o data-hetero/cp32.65536.1024.bs32.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 32 --data_dist 1lns --batch_size 32



# mkdir -p data-hetero
# BATCH_SIZE=1

# for BATCH_SIZE in 1 32; do
#     for LONG_SEQ_LEN in 65536; do
#         for SINGLE_SEQ_LEN in 1024; do
#             for cp_degree in 1 2 4 8 16 32; do
#                 NSYS_OUTPUT=data-hetero/cp${cp_degree}.${LONG_SEQ_LEN}.${SINGLE_SEQ_LEN}.bs${BATCH_SIZE}.nsys-rep
#                 if [ -f $NSYS_OUTPUT ]; then
#                     echo "Skipping $NSYS_OUTPUT because it already exists"
#                     continue
#                 fi
#                 echo nsys profile \
#                     -t nvtx,cuda -w true \
#                     -o $NSYS_OUTPUT \
#                     --force-overwrite true \
#                 python main.py \
#                     --nhead 1 \
#                     --head_dim 128 \
#                     --total_tokens $((128 * 1024)) \
#                     --longest_seq_len ${LONG_SEQ_LEN} \
#                     --single_seq_len ${SINGLE_SEQ_LEN} \
#                     --cp_degree $cp_degree \
#                     --data_dist '1lns' \
#                     --batch_size $BATCH_SIZE \
#                     ;
                    
#                 # echo "<< cp_degree = $cp_degree, longest_seq_len = $LONG_SEQ_LEN, single_seq_len = $SINGLE_SEQ_LEN >>" | tee -a output.hetero.txt
#                 # nsys stats $NSYS_OUTPUT --force-export=true | grep -C 3 flash_fwd_kernel | tee -a output.hetero.txt
#                 # echo ""
#             done
#         done
#     done
# done


# # mkdir -p data-hetero

# # for LONG_SEQ_LEN in 16384 32768 49152 65536 114688 126976; do
# #     for SINGLE_SEQ_LEN in 1024; do
# #         for cp_degree in 1 2 4 8 16 32; do
# #             NSYS_OUTPUT=data-hetero/cp${cp_degree}.${LONG_SEQ_LEN}.${SINGLE_SEQ_LEN}.nsys-rep
# #             if [ -f $NSYS_OUTPUT ]; then
# #                 echo "Skipping $NSYS_OUTPUT because it already exists"
# #                 continue
# #             fi
            
# #             nsys profile \
# #                 -t nvtx,cuda -w true \
# #                 -o $NSYS_OUTPUT \
# #                 --force-overwrite true \
# #             python main.py \
# #                 --nhead 1 \
# #                 --head_dim 128 \
# #                 --total_tokens $((128 * 1024)) \
# #                 --longest_seq_len ${LONG_SEQ_LEN} \
# #                 --single_seq_len ${SINGLE_SEQ_LEN} \
# #                 --cp_degree $cp_degree \
# #                 --data_dist '1lns' ;
                
# #             echo "<< cp_degree = $cp_degree, longest_seq_len = $LONG_SEQ_LEN, single_seq_len = $SINGLE_SEQ_LEN >>" | tee -a output.hetero.txt
# #             nsys stats $NSYS_OUTPUT --force-export=true | grep -C 3 flash_fwd_kernel | tee -a output.hetero.txt
# #             echo ""
# #         done
# #     done
# # done







# # mkdir -p data-homo

# # for SINGLE_SEQ_LEN in 1024 2048 4096 8192 16384 32768; do
# #     for cp_degree in 1 2 4 8 16 32; do
# #         NSYS_OUTPUT=data-homo/cp${cp_degree}.${SINGLE_SEQ_LEN}.nsys-rep
# #         if [ -f $NSYS_OUTPUT ]; then
# #             echo "Skipping $NSYS_OUTPUT because it already exists"
# #             continue
# #         fi
        
# #         nsys profile \
# #             -t nvtx,cuda -w true \
# #             -o $NSYS_OUTPUT \
# #             --force-overwrite true \
# #         python main.py \
# #             --nhead 1 \
# #             --head_dim 128 \
# #             --total_tokens $((128 * 1024)) \
# #             --single_seq_len $((SINGLE_SEQ_LEN)) \
# #             --cp_degree $cp_degree \
# #             --data_dist 'homo'
            
# #         echo "<< cp_degree = $cp_degree, single_seq_len = $SINGLE_SEQ_LEN >>" | tee -a output.txt
# #         nsys stats $NSYS_OUTPUT --force-export=true --report cuda_gpu_kern_sum | grep -C 3 flash_fwd_kernel | tee -a output.txt
# #         echo ""
# #         # exit
# #     done
# # done



# # nsys profile -t nvtx,cuda -w true -o data-hetero/cp1.65536.1024.bs1.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 1 --data_dist 1lns --batch_size 1
# # nsys profile -t nvtx,cuda -w true -o data-hetero/cp2.65536.1024.bs1.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 2 --data_dist 1lns --batch_size 1
# # nsys profile -t nvtx,cuda -w true -o data-hetero/cp4.65536.1024.bs1.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 4 --data_dist 1lns --batch_size 1
# # nsys profile -t nvtx,cuda -w true -o data-hetero/cp8.65536.1024.bs1.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 8 --data_dist 1lns --batch_size 1
# # nsys profile -t nvtx,cuda -w true -o data-hetero/cp16.65536.1024.bs1.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 16 --data_dist 1lns --batch_size 1
# # nsys profile -t nvtx,cuda -w true -o data-hetero/cp32.65536.1024.bs1.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 32 --data_dist 1lns --batch_size 1
# # nsys profile -t nvtx,cuda -w true -o data-hetero/cp1.65536.1024.bs32.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 1 --data_dist 1lns --batch_size 32
# # nsys profile -t nvtx,cuda -w true -o data-hetero/cp2.65536.1024.bs32.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 2 --data_dist 1lns --batch_size 32
# # nsys profile -t nvtx,cuda -w true -o data-hetero/cp4.65536.1024.bs32.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 4 --data_dist 1lns --batch_size 32
# # nsys profile -t nvtx,cuda -w true -o data-hetero/cp8.65536.1024.bs32.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 8 --data_dist 1lns --batch_size 32
# # nsys profile -t nvtx,cuda -w true -o data-hetero/cp16.65536.1024.bs32.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 16 --data_dist 1lns --batch_size 32
# # nsys profile -t nvtx,cuda -w true -o data-hetero/cp32.65536.1024.bs32.nsys-rep --force-overwrite true python main.py --nhead 1 --head_dim 128 --total_tokens 131072 --longest_seq_len 65536 --single_seq_len 1024 --cp_degree 32 --data_dist 1lns --batch_size 32

