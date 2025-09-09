# %%
import wandb
import json
import os
import pandas as pd
import time
import datetime

wandb.login()

run = wandb.init(
    entity="junda-d2",
    project="d2-dpcp-0908-large-scale",
    name="v2",
    resume="allow",
)

while True:
    benchmark_data = []
    folders = os.listdir("logs.v2-large_scale")
    for folder in folders:
        print(folder)
        name = folder.split(".")[-1]
        
        benchmark_file = os.path.join("logs.v2-large_scale", folder, "benchmark.raw.jsonl")
        if not os.path.exists(benchmark_file):
            continue


        # d2-cp1-n32-b1-t131072-normal
        parts = name.split("-")
        mode = parts[0]
        cp_size = parts[1]
        nodes = parts[2]
        batch_size = parts[3]
        num_tokens = parts[4]
        
        this_benchmark_data = []
        with open(benchmark_file, "r") as f:
            for line in f:
                data = json.loads(line)
                benchmark_data.append((
                    name,
                    data['sample_id'],
                    data['duration_ms'],
                    mode,
                    cp_size,
                    nodes,
                    batch_size,
                    num_tokens,
                ))

        print(f"🟡 {name} has {len(this_benchmark_data)} samples")

    df = pd.DataFrame(benchmark_data, columns=['name', 'sample_id', 'duration_ms', 'mode', 'cp_size', 'nodes', 'batch_size', 'num_tokens'])
    
    # Pivot the dataframe to have sample_id as rows and name as columns
    pivot_df = df.drop_duplicates(subset=['sample_id', 'name'], keep='last').pivot(index='sample_id', columns='name', values='duration_ms').round(2)
    pivot_df.to_csv(f"pivot_df.csv")
    print(pivot_df)

    # Speedup dataframe
    speedup_df = df[[
        'nodes', 'batch_size', 'num_tokens', 'mode', 'cp_size', 'duration_ms',
    ]].sort_values(by=['nodes', 'batch_size', 'num_tokens', 'mode', 'cp_size'], ascending=True)
    speedup_df.to_csv(f"speedup_df.tsv", sep="\t", index=False)



    run.summary["benchmark_data"] = wandb.Table(dataframe=pivot_df)
    run.log({
        "num_folders": len(folders),
        "num_points": len(benchmark_data),
    })
    
    time.sleep(15)
    

"""
is_suspicious_job = False
# if since its start time, 5 min it still haven't progress, it's suspicious.
is_successfully_exited = os.path.exists(os.path.join(folder, "benchmark.json"))
if not is_successfully_exited:
    start_time = folder.split(".")[0]
    start_time = datetime.datetime.strptime(start_time, "%Y%m%d_%H%M%S")
    if time.time() - start_time.timestamp() < 4 * 60: # timeout 4 min
        continue
    is_suspicious_job = True
    pass

"""