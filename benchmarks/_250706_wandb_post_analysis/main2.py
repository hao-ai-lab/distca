import wandb
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px

api = wandb.Api()
runs = api.runs("junda-d2/d2")

# Collect all data for box plot
all_data = []
run_names = []
run_ids = []

for i, run in enumerate(runs):
    csv_path = f"plots/{run.id}.best_ratio.csv"
    
    if os.path.exists(csv_path):
        best_ratio = pd.read_csv(csv_path)
        print(f"Loaded best ratio from {csv_path}")
        
        # Add data for this run
        run_data = {
            'run_id': i,
            'run_name': run.name,
            'run_hash': run.id,
            'best_ratio': best_ratio.iloc[:, 0].values  # Assuming first column contains the ratios
        }
        
        # Flatten the data for plotly
        for ratio in run_data['best_ratio']:
            all_data.append({
                'run_id': i,
                'run_name': run.name,
                'run_hash': run.id,
                'best_ratio': ratio
            })
        
        run_names.append(run.name)
        run_ids.append(i)
    else:
        print(f"CSV file not found for run {run.id}, skipping...")

# Convert to DataFrame
df = pd.DataFrame(all_data)

if len(df) > 0:
    # Create box plot using plotly
    fig = go.Figure()
    
    # Add box plot for each run
    for run_id in sorted(df['run_id'].unique()):
        run_data = df[df['run_id'] == run_id]
        run_name = run_data['run_name'].iloc[0]
        
        fig.add_trace(go.Box(
            y=run_data['best_ratio'],
            name=str(run_id),
            boxmean=True
        ))
    
    # Update layout
    fig.update_layout(
        title="Box Plot of Best Ratios Across Runs",
        xaxis_title="Run ID",
        yaxis_title="Best Ratio",
        showlegend=False,
        width=1200,
        height=600
    )
    
    # Add horizontal line at y=1
    fig.add_hline(y=1, line_dash="dash", line_color="red", 
                  annotation_text="y=1", annotation_position="bottom right")
    
    # Create plots2 directory
    os.makedirs("plots2", exist_ok=True)
    
    # Write run ID to name mapping to text file
    with open("plots2/run_mapping.txt", "w") as f:
        f.write("Run ID to Name Mapping:\n")
        f.write("=" * 30 + "\n")
        for i, name in zip(run_ids, run_names):
            f.write(f"{i}: {name}\n")
    
    # Save the plot
    fig.write_html("plots2/box_plot.html")
    fig.write_image("plots2/box_plot.png")
    print("Saved box plot to plots2/box_plot.html and plots2/box_plot.png")
    print("Saved run mapping to plots2/run_mapping.txt")
    
    # Show the plot
    fig.show()
else:
    print("No data found to plot!") 