# %%
import os
import pandas as pd
# %%
a = ! ls logs/
len(a)
# %%
import re
def parse_key_value_file(path: str) -> dict:
    """Parse a file with lines formatted as 'key: value' into a dictionary."""
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("-"):
                continue
            if not line or ":" not in line:
                continue  # skip empty or malformed lines
            key, value = line.split(":", 1)  # split only on the first colon
            key = key[2:]
            result[key.strip()] = value.strip()
    return result
# %%
b = f"logs/{a[0]}/README.md"
b
parse_key_value_file(b)
os.path.exists(b)
# %%
import json

DF_INCLUDE_BENCHMARK = True
# DF_ONLY_SHOW_SUCCESS = True
DF_ONLY_SHOW_SUCCESS = False
DF_ONLY_SHOW_FAILED = False

results = []
for exp_name in a:
    file_path = f"logs/{exp_name}/README.md"
    if not os.path.exists(file_path):
        continue
    config = parse_key_value_file(file_path)

    slurm_env_file = f"logs/{exp_name}/slurm.env"
    with open(slurm_env_file, "r") as f:
        slurm_env = f.read()
        slurm_env = slurm_env.split("\n")
        slurm_env_dict = {}
        for line in slurm_env:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)  # split only on the first colon
            slurm_env_dict[key.strip()] = value.strip()

        nnodes = int(slurm_env_dict["SLURM_NNODES"])
        batch_size = int(slurm_env_dict["BATCH_SIZE"])
        num_layers = int(slurm_env_dict["NUM_LAYERS"])
        # rdzv_id = int(slurm_env_dict["RANDOM"])

    benchmark_file = f"logs/{exp_name}/benchmark.json"
    run_successfully = os.path.exists(benchmark_file)
    benchmark_datas = {}
    if run_successfully and DF_INCLUDE_BENCHMARK:
        with open(benchmark_file, "r") as f:
            benchmark = json.load(f)
        samples = benchmark["samples"]
        for sample in samples:
            sample_id = sample["sample_id"]
            samples_duration = sample["duration_ms"]
            benchmark_datas[f"sample[{sample_id}]"] = samples_duration
    
    DP_SIZE = nnodes // (
        int(config.get("CP_SIZE")) * int(config.get("PP_SIZE"))
    )
    
    results.append(dict(
        success=run_successfully,
        nnodes=nnodes,
        batch_size=batch_size,
        DP_SIZE=DP_SIZE,
        **config,
        num_layers=num_layers,
        exp_name=exp_name,
        # rdzv_id=rdzv_id,
        **benchmark_datas,
    ))


df = pd.DataFrame(results)
sample_columns = [col for col in df.columns if col.startswith("sample[")]
df['total_time'] = df[sample_columns].sum(axis=1)

df_display = df.copy()

if DF_ONLY_SHOW_SUCCESS:
    df_display = df_display[df_display['success'] == True]
df_display['success'] = df_display['success'].map({True: '✅', False: '❌'})
df_display['MODE_EMOJI'] = df_display['MODE'].map({
    'wlbllm': '🟥',
    'd2': '🟦',
})
# Sort by gid
df_display['gid'] = df_display['nnodes'].astype(str) + "_" + df_display['NUM_TOKENS'].astype(str) + "_" + df_display['BATCH_SIZE'].astype(str)


# Then, aggregate by gid, and then compare the (only one) d2 row's total_time with all the wlbllm row's total_time in that aggregated group. Add a speedup = wlbllm min total time in that window / d2 total time
# Group by gid and calculate speedup
import numpy as np
speedups = []
for gid, group in df_display.groupby('gid'):
    d2_rows = group[group['MODE_EMOJI'] == '🟦']
    wlbllm_rows = group[group['MODE_EMOJI'] == '🟥']
    
    if len(d2_rows) == 1 and len(wlbllm_rows) >= 1:
        d2_time = d2_rows['total_time'].iloc[0]
        wlbllm_min_time = wlbllm_rows['total_time'].min()
        speedup = wlbllm_min_time / d2_time
        
        # Only add speedup to d2 row
        for idx in d2_rows.index:
            speedups.append((idx, speedup))

# Add speedup column
df_display['speedup'] = np.nan
for idx, speedup in speedups:
    df_display.loc[idx, 'speedup'] = speedup

# Format speedup to 2 decimal places where it exists
df_display['speedup'] = df_display['speedup'].apply(lambda x: f'{x:.2f}' if pd.notnull(x) else '')

# Reorder columns to put key columns first
key_columns = ['gid', 'nnodes', 'NUM_TOKENS', 'BATCH_SIZE', 'MODE_EMOJI',]
front_columns = ['gid', 'nnodes', 'NUM_TOKENS', 'BATCH_SIZE', 'MODE_EMOJI', 'total_time', 'speedup']
other_columns = [col for col in df_display.columns if col not in front_columns]
df_display = df_display[front_columns + other_columns]
df_display.sort_values(by=key_columns, ascending=True, inplace=True)


# When display the total_time column, make it %2f
df_display['total_time'] = df_display['total_time'].astype(float).round(2)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
df_display
# %%
# Save this to a file
df_display.to_csv("analyze_status.csv", index=False)

# %%


# %%
df_display.sort_values(by='total_time', ascending=True, inplace=True)

# %%
# Aggregate by gid, and then for each `sample[%d]` column, take the 

# %%
success_rate = df['success'].mean()
success_rate
# %%
# Log inspection

import glob
import time
exp_name = "20250829_064849.jobe2e-656191"
exp_name = "20250829_083600.jobe2e-656141"

log_files = glob.glob(f"logs/{exp_name}/logs/*")
for i, log_file in enumerate(log_files):
    print(f"\n\n=== {log_file} ===")
    with open(log_file, "r") as f:
        log = f.read()
    print(log)
    # time.sleep(5)

# %%
# How to relaunch this experiment?
df[df['exp_name'] == exp_name]
# Recombined  to the command line expression
#                 MODE=wlbllm BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=100 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=4 sbatch --nodes $nodes 
# test_e2e_combined.slurm.sh
MODE = df[df['exp_name'] == exp_name]['MODE'].values[0]
BATCH_SIZE = df[df['exp_name'] == exp_name]['BATCH_SIZE'].values[0]
NUM_TOKENS = df[df['exp_name'] == exp_name]['NUM_TOKENS'].values[0]
MAX_SAMPLE_ID = df[df['exp_name'] == exp_name]['MAX_SAMPLE_ID'].values[0]
TP_SIZE = df[df['exp_name'] == exp_name]['TP_SIZE'].values[0]
CP_SIZE = df[df['exp_name'] == exp_name]['CP_SIZE'].values[0]
NUM_LAYERS = df[df['exp_name'] == exp_name]['num_layers'].values[0]
NODES = df[df['exp_name'] == exp_name]['nnodes'].values[0]

command_line = f"MODE={MODE} BATCH_SIZE={BATCH_SIZE} NUM_TOKENS={NUM_TOKENS} MAX_SAMPLE_ID={MAX_SAMPLE_ID} TP_SIZE={TP_SIZE} CP_SIZE={CP_SIZE} NUM_LAYERS={NUM_LAYERS} sbatch --nodes {NODES} test_e2e_combined.slurm.sh"
print(command_line)
# %%
