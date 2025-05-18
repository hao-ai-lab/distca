import pandas as pd
import numpy as np
import torch

def _load_and_filter_data(df, num_gpus, nelem, dtype):    
    # Filter for the specific number of GPUs
    df_filtered = df[df['num_gpus_per_node'] == num_gpus]
    
    if len(df_filtered) == 0:
        df_filtered = df[df['num_gpus_per_node'] == (num_gpus // 2)].copy()
        df_filtered['time(us)'] /= 2
        
    # Convert size to KB for matching with data
    size_kb = nelem * dtype.itemsize / 1024
    
    return df_filtered, size_kb

def _interpolate_time(df_filtered, size_kb) -> 'int[us]':
    # Get the sizes and times
    sizes = df_filtered['size(kb)'].values
    times = df_filtered['time(us)'].values
    
    # Find nearest size points and interpolate
    if size_kb <= sizes[0]:
        return float(times[0])
    elif size_kb >= sizes[-1]:
        x = np.array(sizes[-2:])
        y = np.array(times[-2:])
        slope, intercept = np.polyfit(x, y, 1)
        return float(intercept + slope * size_kb)
    else:
        idx = np.searchsorted(sizes, size_kb)
        x0, x1 = sizes[idx-1], sizes[idx]
        y0, y1 = times[idx-1], times[idx]
        return float(y0 + (y1 - y0) * (size_kb - x0) / (x1 - x0))

def all_gather_time_all(cp, nelem, dtype):
    if not hasattr(all_gather_time_all, '_data'):
        all_gather_time_all._data = pd.read_csv('data/comm-A100-SXM-80GB/all_gather.csv')
    if cp == 1: 
        return 0
    df_filtered, size_kb = _load_and_filter_data(all_gather_time_all._data, cp, nelem, dtype)
    return _interpolate_time(df_filtered, size_kb)

def reduce_scatter_time_all(cp, nelem, dtype):
    if not hasattr(reduce_scatter_time_all, '_data'):
        reduce_scatter_time_all._data = pd.read_csv('data/comm-A100-SXM-80GB/reduce_scatter.csv')
    if cp == 1: 
        return 0
    df_filtered, size_kb = _load_and_filter_data(reduce_scatter_time_all._data, cp, nelem, dtype)
    return _interpolate_time(df_filtered, size_kb)

def all_reduce_time_all(tp, nelem, dtype):
    if not hasattr(all_reduce_time_all, '_data'):
        all_reduce_time_all._data = pd.read_csv('data/comm-A100-SXM-80GB/all_reduce.csv')
    if tp == 1: 
        return 0
    df_filtered, size_kb = _load_and_filter_data(all_reduce_time_all._data, tp, nelem, dtype)
    return _interpolate_time(df_filtered, size_kb)


K = 1024

def print_doc_cp_attn_network_table(
        head_dim = 128,
    num_qo_heads = 32,
    num_kv_heads = 32,
):
    # all_gather
    print("tp,cp,sf,all_gather,reduce_scatter,all_reduce,fwd,bwd")
    for tp in [1, 2, 4, 8]:
        for cp in [1, 2, 4, 8]:
            for sf in [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 48, 64, 96, 128]:
                doc_lens = int(sf * K)
                qo_heads = num_qo_heads // tp
                kv_heads = num_kv_heads // tp

                qo_hidden_dim = head_dim * num_qo_heads
                kv_hidden_dim = head_dim * num_kv_heads
                byte_size = 2 # fp16

                all_gather_time = all_gather_time_all(cp, sf * K * kv_hidden_dim, dtype=torch.bfloat16) / 1000 # ms
                reduce_scatter_time = reduce_scatter_time_all(cp, sf * K * kv_hidden_dim, dtype=torch.bfloat16) / 1000 # ms
                all_reduce_time = all_reduce_time_all(tp, sf * K * qo_hidden_dim, dtype=torch.bfloat16) / 1000 # ms
                
                fwd_time = all_gather_time + reduce_scatter_time + all_reduce_time
                bwd_time = all_gather_time + reduce_scatter_time + all_reduce_time

                print(f"{tp},{cp},{sf},{all_gather_time:.2f},{reduce_scatter_time:.2f},{all_reduce_time:.2f},{fwd_time:.2f},{bwd_time:.2f}")


    # reduce_scatter_tensor
    pass
