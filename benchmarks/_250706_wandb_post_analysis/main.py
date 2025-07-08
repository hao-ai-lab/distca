import wandb
import pandas as pd
import os
import matplotlib.pyplot as plt

api = wandb.Api()
runs = api.runs("junda-d2/d2")

for run in runs:
    csv_path = f"plots/{run.id}.best_ratio.csv"
    png_path = f"plots/{run.id}.best_ratio.png"

    if os.path.exists(csv_path):
        best_ratio = pd.read_csv(csv_path)
        print(f"Loaded best ratio from {csv_path}")
    else:
        history = run.history(keys=None)
        attnserver_cols = [
            col for col in history.columns 
            if col.startswith("attnserver/")
        ]
        wlb_cols = [
            col for col in history.columns 
            if col.startswith("wlb/")
        ]

        attnserver_history = history[attnserver_cols]
        wlb_history = history[wlb_cols]

        # For attnserver_history, take the min across all columns
        attnserver_history["min_attnserver"] = attnserver_history.min(axis=1)

        # For wlb_history, take the min across all columns
        wlb_history["min_wlb"] = wlb_history.min(axis=1)

        # Combine the two dataframes
        best_ratio = attnserver_history["min_attnserver"] / wlb_history["min_wlb"]

        # Save the best ratio to a csv
        os.makedirs("plots", exist_ok=True)
        best_ratio.to_csv(csv_path, index=False)
        print(f"Saved best ratio to {csv_path}")

    # Plot the best ratio as a line chart using matplotlib
    plt.figure()
    x = best_ratio.index
    y = best_ratio.values

    plt.plot(x, y, label='Best Ratio')
    plt.axhline(y=1, color='r', linestyle='--', label='y=1')
    plt.ylim(0.5, 1.5)
    plt.xlabel('Index')
    plt.ylabel('Best Ratio')
    plt.title(f'Best Ratio for Run: {run.name} ({run.id})')
    plt.legend()
    plt.savefig(png_path)
    plt.close()
    print(f"Saved best ratio plot to {png_path}")
    