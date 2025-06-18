# TODO: Add more models.

INF = int(1e15)

def get_mlp_time(x: int, tp: int, cp: int) -> float:
    # length -> time ms
    unit_time = {
        64:10.96,
        128:11.68,
        256:11.01,
        512:13.24,
        1024:11.46,
        2048:12.56,
        4096:13.87,
        8192:17.79,
        16384:25.86,
        32768:44.25,
        65536:76.6,
        131072:153.2,
        262144:306.4,
        524288:612.8
    }

    
    y = x / (tp * cp)
    # Find the two closest keys to y
    sorted_keys = list(unit_time.keys())
    # This is guaranteed to be sorted anyways.

    if y < 64:
        return unit_time[64]
    elif y > 524288:
        return unit_time[524288] / 524288 * y
    
    # If y is an exact match, return immediately
    if y in unit_time:
        z = unit_time[y]
    else:
        # Find the two closest keys
        lower_key = max(k for k in sorted_keys if k <= y)
        upper_key = min(k for k in sorted_keys if k >= y)
        
        # Linear interpolation
        lower_time = unit_time[lower_key]
        upper_time = unit_time[upper_key]
        
        # Interpolate
        z = lower_time + (upper_time - lower_time) * (y - lower_key) / (upper_key - lower_key)
    return z


# (tp, cp) -> {seq_len: latency(ms)}
ATTN_plan2dict = None


def setup_attn_data():
    global ATTN_plan2dict
    if ATTN_plan2dict is not None:
        return
    
    from d2.profiling.get_attn_data import get_attn_data
    attn_mapping = get_attn_data()
    ATTN_plan2dict = attn_mapping
    return 


def get_attn_time(
    x: int, tp: int, cp: int,
    hqo: int = 64, hkv: int = 4, d: int = 128,
) -> float:
    setup_attn_data()
    scoped_time = ATTN_plan2dict[(tp, cp)]

    y = x
    min_key = min(scoped_time.keys())
    max_key = max(scoped_time.keys())

    if y < min_key:
        return scoped_time[min_key]
    
    if y > max_key:
        return scoped_time[max_key] * (y / max_key) ** 2
    
    if y in scoped_time:
        return scoped_time[y]
    
    # Interpolate
    import bisect

    sorted_keys = list(scoped_time.keys())

    # lower_index = bisect.bisect_right(sorted_keys, y) - 1
    # upper_index = bisect.bisect_left(sorted_keys, y)
    # lower_key = sorted_keys[lower_index]
    # upper_key = sorted_keys[upper_index]

    lower_key = max(k for k in sorted_keys if k <= y)
    upper_key = min(k for k in sorted_keys if k >= y)

    lower_time = scoped_time[lower_key]
    upper_time = scoped_time[upper_key]
    result = lower_time + (upper_time - lower_time) * (y - lower_key) / (upper_key - lower_key)
    return result
