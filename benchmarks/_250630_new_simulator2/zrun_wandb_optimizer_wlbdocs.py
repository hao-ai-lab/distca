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


def run_wlbdocs_altered_experiment(max_ctx_length):
    # Define sample configuration
    sample_name = "wlbdocs_altered"

    batches = list(batch_documents(
        sample_wlbllm_docs_altered(size=1000), 
        max_ctx_length=max_ctx_length,
    ))

    # Create a run name with sample name and max context length
    ctx_str = f"{int(max_ctx_length/tm.K)}K"
    run_name = f"{sample_name}_ctx_{ctx_str}"

    wandb.init(project="d2", name=run_name)

    # Define metrics for W&B
    wandb.define_metric("step")
    wandb.define_metric("ratio/*", step_metric="step")

    wandb.config.update(dict(
        sample_name=sample_name,
        sample_configs=dict(
            size=1000,
        ),
        max_ctx_length=max_ctx_length,
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
    
    # Finish the run
    wandb.finish()


if __name__ == "__main__":
    for max_length in [64, 128, 256, 512, 1024]:
        run_wlbdocs_altered_experiment(max_ctx_length=max_length * tm.K)