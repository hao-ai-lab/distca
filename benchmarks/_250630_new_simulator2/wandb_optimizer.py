from samples import (
    sample_wlbllm_docs, 
    sample_wlbllm_docs_altered,
    sample_multimodal_gaussian,
    batch_documents,
)
import timemodule as tm

import wandb
from datetime import datetime

# Define sample configuration first
sample_name="wlbllm_docs"

batches = list(batch_documents(
    sample_wlbllm_docs(size=1000), 
    max_ctx_length=64 * tm.K,
))

# Create a unique run name with sample name and timestamp
run_name = f"{sample_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

wandb.init(project="d2", name=run_name)

# Define metrics for W&B
wandb.define_metric("step")
wandb.define_metric("ratio/*", step_metric="step")

from wandb_optimizer_attnserver_lessvar import main as attnserver_solver
from wandb_optimizer_wlbllm import main as wlbllm_solver

wandb.config.update(dict(
    sample_name=sample_name,
    sample_configs=dict(
        means=[8*tm.K, 16*tm.K, 32*tm.K, 64*tm.K],
        sigmas=[1*tm.K, 2*tm.K, 4*tm.K, 8*tm.K],
        weights=[0.25, 0.25, 0.25, 0.25],
        size=1000,
    ),
))

for idx, batch in enumerate(batches):
    wlbllm_results = {}
    for tp in (1, 2, 4, 8):
        for cp in (1, 2, 4, 8):
            wlbllm_results[(tp, cp)] = wlbllm_solver(
                batch=batch,
                num_total_devices=64,
                tp=tp,
                cp=cp,
                max_time_in_seconds=360,
                sample_name=sample_name,
                sample_id=idx,
            )
    
    attnserver_results = attnserver_solver(
        batch=batch,
        num_total_devices=64,
        mlp_tp=8,
        mlp_cp=4,
        num_workers=8,
        max_time_in_seconds=360,
        sample_name=sample_name,
        sample_id=idx,
        sweep_all_mlp_plans=True,
    )

    log_data = {"step": idx}
    for tp in (1, 2, 4, 8):
        for cp in (1, 2, 4, 8):
            wlb_result = wlbllm_results[(tp, cp)]
            attnserver_result = attnserver_results['sweep_results'][(tp, cp)]
            ratio = attnserver_result["batch_total_time"] / wlb_result["max_worker_latency_us"]
            key = f"ratio/tp{tp}_cp{cp}"
            log_data[key] = ratio

            log_data[f"wlbllm/tp{tp}_cp{cp}"] = wlb_result
            log_data[f"attnserver/tp{tp}_cp{cp}"] = attnserver_result

    wandb.log(log_data)
    # breakpoint()