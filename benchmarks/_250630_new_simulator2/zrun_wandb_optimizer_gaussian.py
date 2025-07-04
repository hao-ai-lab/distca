import wandb
import os
os.environ["WANDB_API_KEY"] = "02575b6c73e438f9885daa7cf691a45939d26a71"

wandb.login(
    key="02575b6c73e438f9885daa7cf691a45939d26a71"
)


from samples import (
    sample_wlbllm_docs, 
    sample_wlbllm_docs_altered,
    sample_multimodal_gaussian,
    batch_documents,
)
import timemodule as tm

import wandb
from datetime import datetime

from wandb_optimizer_attnserver_lessvar import main as attnserver_solver
from wandb_optimizer_wlbllm import main as wlbllm_solver


def run_gaussian_experiment(
    max_ctx_length,
    num_total_devices=64,
    max_num_workers_attnserver=8,
):
    # Define sample configuration
    sample_name = "gaussian"
    means = [8*tm.K, 16*tm.K, 32*tm.K, 64*tm.K]

    batches = list(batch_documents(
        sample_multimodal_gaussian(
            means=means,
            sigmas=[1*tm.K, 2*tm.K, 4*tm.K, 8*tm.K],
            weights=[0.25, 0.25, 0.25, 0.25],
            size=1000,
        ), 
        max_ctx_length=max_ctx_length,
    ))

    # Create a run name with sample name, means, and max context length
    means_str = "_".join([f"{int(m/tm.K)}K" for m in means])
    ctx_str = f"{int(max_ctx_length/tm.K)}K"
    run_name = f"{sample_name}_means_{means_str}_ctx_{ctx_str}_gpu{num_total_devices}_attnwkrs{max_num_workers_attnserver}"

    wandb.init(
        entity="junda-d2",
        project="d2", 
        name=run_name
    )

    # Define metrics for W&B
    wandb.define_metric("step")
    wandb.define_metric("ratio/*", step_metric="step")
    wandb.define_metric("wlb/*", step_metric="step")
    wandb.define_metric("attnserver/*", step_metric="step")
    wandb.define_metric("input/batch", step_metric="step")

    wandb.config.update(dict(
        sample_name=sample_name,
        sample_configs=dict(
            means=means,
            sigmas=[1*tm.K, 2*tm.K, 4*tm.K, 8*tm.K],
            weights=[0.25, 0.25, 0.25, 0.25],
            size=1000,
        ),
        max_ctx_length=max_ctx_length,
        num_total_devices=num_total_devices,
        max_num_workers_attnserver=max_num_workers_attnserver,
    ))

    for idx, batch in enumerate(batches):
        wlbllm_results = {}
        for tp in (1, 2, 4, 8):
            for cp in (1, 2, 4, 8):
                wlbllm_results[(tp, cp)] = wlbllm_solver(
                    batch=batch,
                    num_total_devices=num_total_devices,
                    tp=tp,
                    cp=cp,
                    max_time_in_seconds=360,
                    sample_name=sample_name,
                    sample_id=idx,
                )
        
        attnserver_results = attnserver_solver(
            batch=batch,
            num_total_devices=num_total_devices,
            mlp_tp=8,
            mlp_cp=4,
            num_workers=max_num_workers_attnserver,
            max_time_in_seconds=360,
            sample_name=sample_name,
            sample_id=idx,
            sweep_all_mlp_plans=True,
        )

        log_data = {"step": idx}
        log_items = []
        for tp in (1, 2, 4, 8):
            for cp in (1, 2, 4, 8):
                dp = num_total_devices // (tp * cp)
                wlb_result = wlbllm_results[(tp, cp)]
                attnserver_result = attnserver_results['sweep_results'][(tp, cp)]
                ratio = attnserver_result["batch_total_time"] / wlb_result["max_worker_latency_us"]
                
                # Log the ratio of attnserver / wlbllm
                key = f"ratio/tp{tp}_cp{cp}_dp{dp}"
                log_data[key] = ratio
                # Log the abs value
                key = f"wlb/tp{tp}_cp{cp}_dp{dp}"
                log_data[key] = wlb_result["max_worker_latency_us"]
                key = f"attnserver/tp{tp}_cp{cp}_dp{dp}"
                log_data[key] = attnserver_result["batch_total_time"]
                

                log_item = (dict(
                    idx=idx,
                    input_batch=batch,
                    run_name=run_name, tp=tp, cp=cp, dp=dp, 
                    ratio=ratio,
                    num_total_devices=num_total_devices, 
                    max_num_workers_attnserver=max_num_workers_attnserver,
                    wlb_result=wlb_result,
                    attnserver_result=attnserver_result
                ))
                log_items.append(log_item)

        key = f"input/batch"
        log_data[key] = batch
        wandb.log(log_data)
        with open(f"logs/{run_name}.jsonl", "a") as f:
            import json
            for log_item in log_items:
                log_item_str = json.dumps(log_item)
                f.write(f"{log_item_str}\n")
            f.flush()
        # breakpoint()

    
    # Finish the run
    wandb.finish()


if __name__ == "__main__":
    for max_length in [64, 128, 256, 512, 1024]:
        run_gaussian_experiment(max_ctx_length=max_length * tm.K)