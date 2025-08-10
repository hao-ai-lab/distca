# %% [markdown]

# # Comparison

# %%
import os

all_tokens = ["8k", "16k", "32k", "64k"]

for tokens in all_tokens:

    data_dir = "../data/20250810_034629.e2e_combined.v3"
    d2_p0_path       = f"{data_dir}/d2.t{tokens}.p0.json"
    d2_p1_path       = f"{data_dir}/d2.t{tokens}.p1.json"
    baseline_path = f"{data_dir}/baseline.t{tokens}.json"
    output_dir    = "compare_d2_baseline.results.3.v3"
    os.makedirs(output_dir, exist_ok=True)
    # %%
    os.path.exists(d2_p0_path), os.path.exists(d2_p1_path), os.path.exists(baseline_path)

    # %%
    import json

    with open(d2_p0_path, "r") as f:
        d2_p0_data = json.load(f)

    with open(d2_p1_path, "r") as f:
        d2_p1_data = json.load(f)

    with open(baseline_path, "r") as f:
        baseline_data = json.load(f)

    # %%
    # First check the `config` field is the same except for `mode`.
    # Then, look at the `samples` field.`
    # for each sample, ensure `samples` is the same.
    # Then record the `duration_ms` field for d2 and baseline.
    # Finally, show the speedup of d2 over baseline.

    # %%
    d2_p0_config = d2_p0_data["config"]
    assert d2_p0_config["mode"] == "d2", "d2_p0_config is not d2"
    d2_p1_config = d2_p1_data["config"]
    assert d2_p1_config["mode"] == "d2", "d2_p1_config is not d2"
    baseline_config = baseline_data["config"]
    assert baseline_config["mode"] == "baseline", "baseline_config is not baseline"

    d2_p0_config.pop("mode", None)
    d2_p0_config.pop("replan_iter", None)
    d2_p1_config.pop("mode", None)
    d2_p1_config.pop("replan_iter", None)
    baseline_config.pop("mode", None)
    baseline_config.pop("replan_iter", None)
    assert d2_p0_config == baseline_config, "Config is not the same"
    assert d2_p1_config == baseline_config, "Config is not the same"


    # %%
    d2_p0_samples = d2_p0_data["samples"]
    d2_p1_samples = d2_p1_data["samples"]
    baseline_samples = baseline_data["samples"]
    assert len(d2_p0_samples) == len(d2_p1_samples) == len(baseline_samples), "Number of samples is not the same"

    # %%
    for d2_p0_sample, d2_p1_sample, baseline_sample in zip(d2_p0_samples, d2_p1_samples, baseline_samples):
        assert d2_p0_sample["samples"] == d2_p1_sample["samples"], "Samples are not the same"
        assert d2_p0_sample["samples"] == baseline_sample["samples"], "Samples are not the same"

    # %%
    d2_p0_duration_ms = [sample["duration_ms"] for sample in d2_p0_samples]
    d2_p1_duration_ms = [sample["duration_ms"] for sample in d2_p1_samples]
    baseline_duration_ms = [sample["duration_ms"] for sample in baseline_samples]
    speedup_p0 = [baseline_duration_ms[i] / d2_p0_duration_ms[i] for i in range(len(d2_p0_duration_ms))]
    speedup_p1 = [baseline_duration_ms[i] / d2_p1_duration_ms[i] for i in range(len(d2_p1_duration_ms))]
    speedup_perc_p0 = [100 * (baseline_duration_ms[i] - d2_p0_duration_ms[i]) / baseline_duration_ms[i] for i in range(len(d2_p0_duration_ms))]
    speedup_perc_p1 = [100 * (baseline_duration_ms[i] - d2_p1_duration_ms[i]) / baseline_duration_ms[i] for i in range(len(d2_p1_duration_ms))]
    # %%
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 14})

    plt.plot(speedup_perc_p0, label="D2 p0 (replan_iter=0)")
    plt.plot(speedup_perc_p1, label="D2 p1 (replan_iter=1)")
    plt.title(f"Llama 8B (nlayer=4) Seq {tokens} DP2TP8 - speedup", fontsize=16)
    plt.ylabel("Speedup % = (baseline - d2) / baseline", fontsize=14)
    plt.xlabel("sample id", fontsize=14)


    plt.legend(fontsize=12)
    plt.savefig(f"{output_dir}/speedup_perc.t{tokens}.png")
    plt.show()

    # %%
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_duration_ms, label="Baseline", marker='o')
    plt.plot(d2_p0_duration_ms, label="D2 p0 (replan_iter=0)", marker='s')
    plt.plot(d2_p1_duration_ms, label="D2 p1 (replan_iter=1)", marker='^')
    plt.title(f"Llama 8B (nlayer=4) Seq {tokens} DP2TP8 - Duration Comparison", fontsize=16)
    plt.ylabel("Duration (ms)", fontsize=14)
    plt.xlabel("Sample ID", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/duration_comparison.t{tokens}.png")
    plt.show()

    # %%
    samples = [sample["samples"] for sample in d2_p0_samples]

    # %%
    import pandas as pd

    df = pd.DataFrame({
        "d2_p0_duration_ms": d2_p0_duration_ms,
        "d2_p1_duration_ms": d2_p1_duration_ms,
        "baseline_duration_ms": baseline_duration_ms,
        "speedup_p0": speedup_p0,
        "speedup_p1": speedup_p1,
        "speedup_perc_p0": speedup_perc_p0,
        "speedup_perc_p1": speedup_perc_p1,
        "samples": samples,
    })

    # Function to format samples for markdown display
    def format_samples_for_markdown(samples_list):
        """Format samples list so each sublist appears on its own line in markdown"""
        if isinstance(samples_list, list):
            # Convert each sublist to string and join with <br> for line breaks
            formatted_sublists = [str(sublist) for sublist in samples_list]
            return "\n".join(formatted_sublists)
        else:
            return str(samples_list)

    df.rename(columns={
        "d2_p0_duration_ms": "D2(ms)", 
        "d2_p1_duration_ms": "D2(ms)", 
        "baseline_duration_ms": "Base(ms)", 
        "speedup_p0": "speedup=\nd2/base", 
        "speedup_p1": "speedup=\nd2/base", 
        "speedup_perc_p0": "speedup%=\n(base-d2)/base", 
        "speedup_perc_p1": "speedup%=\n(base-d2)/base"
    }, inplace=True)
    df.to_csv(f"{output_dir}/d2_baseline_comparison.t{tokens}.tp8dp2.csv", index=False)

    # Create a copy for markdown with formatted samples
    df_markdown = df.copy()
    df_markdown["samples"] = df_markdown["samples"].apply(format_samples_for_markdown)
    df_markdown.to_markdown(f"{output_dir}/d2_baseline_comparison.t{tokens}.tp8dp2.md")


    df.drop(columns=["samples"], inplace=True)
    df.rename(columns={
        "d2_p0_duration_ms": "D2(ms)", 
        "d2_p1_duration_ms": "D2(ms)", 
        "baseline_duration_ms": "Base(ms)", 
        "speedup_p0": "speedup=\nd2/base", 
        "speedup_p1": "speedup=\nd2/base", 
        "speedup_perc_p0": "speedup%=\n(base-d2)/base", 
        "speedup_perc_p1": "speedup%=\n(base-d2)/base"
    }, inplace=True)
    df.to_csv(f"{output_dir}/d2_baseline_comparison.t{tokens}.tp8dp2.simple.csv", index=False)
    df.to_markdown(f"{output_dir}/d2_baseline_comparison.t{tokens}.tp8dp2.simple.md")

    # %%
