# %%
# name = "attn-dist_wlbllm-64k"

import sys
import subprocess
import os

def is_jupyter():
    return 'get_ipython' in globals()

# Ensure the plot directory exists
os.makedirs("plot", exist_ok=True)

default_datasets = ["attn-dist_wlbllm-64k", "attn-dist_multimodal-64k"]

args = sys.argv[1:]
args = args or default_datasets

# If multiple names are provided, recursively call the script
if len(args) > 1 and not is_jupyter():
    for name in args:
        subprocess.run([sys.executable, __file__, name])
    sys.exit(0)

if is_jupyter():
    # Else use the single provided name, or default
    name = default_datasets[0]
else:
    name = args[0]

print(f"Processing: {name}")

# %%
import pandas as pd
filename = f"data-run/{name}.psv"
# filename = f"attn-dist_multimodal-64k.psv"
df = pd.read_csv(f"{filename}", sep="|")
df.head()

# %%
size = len(df['batch'].unique())
max_ctx_length = df['total_len'].max().item()
size, max_ctx_length

# %%

# %%
# Use plotly, for each (tp, cp), plot a scatter plot of (x = real_attn_time_ms, y = sim_attn_time_ms)
import plotly.express as px
import plotly.graph_objs as go

# Create a figure
fig = go.Figure()

# Iterate through unique (tp, cp) combinations
for (tp, cp) in df[['tp', 'cp']].drop_duplicates().itertuples(index=False):
    subset = df[(df['tp'] == tp) & (df['cp'] == cp)]
    
    # Add scatter trace for each (tp, cp) combination
    fig.add_trace(go.Scatter(
        x=subset['real_attn_time_ms'], 
        # y=subset['real_attn_time_ms'] - subset['sim_attn_time_ms'],
        y=subset['sim_attn_time_ms'],
        mode='markers',
        name=f'tp={tp}, cp={cp}',
        hovertemplate='Real Time: %{x:.2f} ms<br>Sim Time: %{y:.2f} ms<br>Total Len: %{customdata[0]}<br>Batch Size: %{customdata[1]}<extra></extra>',
        customdata=subset[['total_len', 'bs']].values
    ))

# Customize the layout
fig.update_layout(
    title=(
        f'Real vs Simulated Attention Time - {name} Dataset<br>'
        f'Dataset: {name}, Sample Size: {size}, Max Context Length: {max_ctx_length}'
    ),
    xaxis_title='Real Attention Time (ms)',
    yaxis_title='Simulated Attention Time (ms)',
    legend_title='TP and CP'
)

# Add a diagonal line for perfect prediction
min_val = df['real_attn_time_ms'].min()
max_val = df['real_attn_time_ms'].max()
fig.add_trace(go.Scatter(
    x=[min_val, max_val], 
    y=[min_val, max_val],
    mode='lines',
    name='Perfect Prediction',
    line=dict(color='red', dash='dash')
))

# Save the plot as an interactive HTML file
fig.write_html(f"plot/{name}.scatter.html")
# Save the plot as a PNG file
fig.write_image(f"plot/{name}.scatter.png")


# Show the plot
fig.show()



# %%
# Create a figure for the difference between real and simulated attention times
fig_diff = go.Figure()

# Get unique tp and cp values to determine color scales
tp_values = sorted(df['tp'].unique())
cp_values = sorted(df['cp'].unique())

# Iterate through unique (tp, cp) combinations
for (tp, cp) in df[['tp', 'cp']].drop_duplicates().itertuples(index=False):
    subset = df[(df['tp'] == tp) & (df['cp'] == cp)]
    
    # Calculate color based on tp (grey to red)
    tp_color_val = (tp_values.index(tp) / (len(tp_values) - 1)) if len(tp_values) > 1 else 0
    tp_color = f'rgb({int(255 * tp_color_val)}, {int(100 * (1 - tp_color_val))}, {int(100 * (1 - tp_color_val))})'
    
    # Calculate color based on cp (grey to green)
    cp_color_val = (cp_values.index(cp) / (len(cp_values) - 1)) if len(cp_values) > 1 else 0
    cp_color = f'rgb({int(100 * (1 - cp_color_val))}, {int(255 * cp_color_val)}, {int(100 * (1 - cp_color_val))})'
    
    # Blend the colors
    blended_color = f'rgb({(int(tp_color.split("(")[1].split(",")[0]) + int(cp_color.split("(")[1].split(",")[0]))//2}, ' \
                    f'{(int(tp_color.split(",")[1]) + int(cp_color.split(",")[1]))//2}, ' \
                    f'{(int(tp_color.split(",")[2].split(")")[0]) + int(cp_color.split(",")[2].split(")")[0]))//2})'
    
    # Add scatter trace for each (tp, cp) combination
    # Now plotting the difference between real and simulated attention times
    fig_diff.add_trace(go.Scatter(
        x=subset['real_attn_time_ms'], 
        y=subset['real_attn_time_ms'] - subset['sim_attn_time_ms'],
        mode='markers',
        name=f'tp={tp}, cp={cp}',
        marker=dict(color=blended_color),
        hovertemplate='Real Time: %{x:.2f} ms<br>Difference: %{y:.2f} ms<br>Total Len: %{customdata[0]}<br>Batch Size: %{customdata[1]}<extra></extra>',
        customdata=subset[['total_len', 'bs']].values
    ))

# Customize the layout
fig_diff.update_layout(
    title=(
        f'Difference Between Real and Simulated Attention Time - {name} Dataset<br>'
        f'Dataset: {name}, Size: {size}, Max Context Length: {max_ctx_length}'
    ),
    xaxis_title='Real Attention Time (ms)',
    yaxis_title='Real - Simulated Attention Time (ms)',
    legend_title='TP and CP'
)

# Save the plot as an interactive HTML file
fig_diff.write_html(f"plot/{name}.diff_scatter.html")
# Save the plot as a PNG file
fig_diff.write_image(f"plot/{name}.diff_scatter.png")

# Show the plot
fig_diff.show()



# %%
# Create a new figure for batch size vs attention time difference
fig_bs_diff = go.Figure()

# Iterate through unique (tp, cp) combinations
for (tp, cp) in df[['tp', 'cp']].drop_duplicates().itertuples(index=False):
    subset = df[(df['tp'] == tp) & (df['cp'] == cp)]
    
    # Calculate color based on tp (grey to red)
    tp_color_val = (tp_values.index(tp) / (len(tp_values) - 1)) if len(tp_values) > 1 else 0
    tp_color = f'rgb({int(255 * tp_color_val)}, {int(100 * (1 - tp_color_val))}, {int(100 * (1 - tp_color_val))})'
    
    # Calculate color based on cp (grey to green)
    cp_color_val = (cp_values.index(cp) / (len(cp_values) - 1)) if len(cp_values) > 1 else 0
    cp_color = f'rgb({int(100 * (1 - cp_color_val))}, {int(255 * cp_color_val)}, {int(100 * (1 - cp_color_val))})'
    
    # Blend the colors
    blended_color = f'rgb({(int(tp_color.split("(")[1].split(",")[0]) + int(cp_color.split("(")[1].split(",")[0]))//2}, ' \
                    f'{(int(tp_color.split(",")[1]) + int(cp_color.split(",")[1]))//2}, ' \
                    f'{(int(tp_color.split(",")[2].split(")")[0]) + int(cp_color.split(",")[2].split(")")[0]))//2})'
    
    # Add scatter trace for each (tp, cp) combination
    # Plotting the difference between real and simulated attention times vs batch size
    fig_bs_diff.add_trace(go.Scatter(
        x=subset['bs'], 
        y=subset['real_attn_time_ms'] - subset['sim_attn_time_ms'],
        mode='markers',
        name=f'tp={tp}, cp={cp}',
        marker=dict(color=blended_color),
        hovertemplate='Batch Size: %{x}<br>Difference: %{y:.2f} ms<br>Total Len: %{customdata[0]}<br>Real Time: %{customdata[1]:.2f} ms<extra></extra>',
        customdata=subset[['total_len', 'real_attn_time_ms']].values
    ))

# Customize the layout
fig_bs_diff.update_layout(
    title=(
        f'Difference Between Real and Simulated Attention Time vs Batch Size - {name} Dataset<br>'
        f'Dataset: {name}, Size: {size}, Max Context Length: {max_ctx_length}'
    ),
    xaxis_title='Batch Size',
    yaxis_title='Real - Simulated Attention Time (ms)',
    legend_title='TP and CP'
)

# Save the plot as an interactive HTML file
fig_bs_diff.write_html(f"plot/{name}.bs_diff_scatter.html")
# Save the plot as a PNG file
fig_bs_diff.write_image(f"plot/{name}.bs_diff_scatter.png")

# Show the plot
fig_bs_diff.show()

# %%
# Create a new figure for relative error
fig_bs_rel_error = go.Figure()

# Iterate through unique (tp, cp) combinations
for (tp, cp) in df[['tp', 'cp']].drop_duplicates().itertuples(index=False):
    subset = df[(df['tp'] == tp) & (df['cp'] == cp)].copy()  # Use .copy() to avoid SettingWithCopyWarning
    
    # Calculate color based on tp (grey to red)
    tp_color_val = (tp_values.index(tp) / (len(tp_values) - 1)) if len(tp_values) > 1 else 0
    tp_color = f'rgb({int(255 * tp_color_val)}, {int(100 * (1 - tp_color_val))}, {int(100 * (1 - tp_color_val))})'
    
    # Calculate color based on cp (grey to green)
    cp_color_val = (cp_values.index(cp) / (len(cp_values) - 1)) if len(cp_values) > 1 else 0
    cp_color = f'rgb({int(100 * (1 - cp_color_val))}, {int(255 * cp_color_val)}, {int(100 * (1 - cp_color_val))})'
    
    # Blend the colors
    blended_color = f'rgb({(int(tp_color.split("(")[1].split(",")[0]) + int(cp_color.split("(")[1].split(",")[0]))//2}, ' \
                    f'{(int(tp_color.split(",")[1]) + int(cp_color.split(",")[1]))//2}, ' \
                    f'{(int(tp_color.split(",")[2].split(")")[0]) + int(cp_color.split(",")[2].split(")")[0]))//2})'
    
    # Calculate relative error using .loc to avoid SettingWithCopyWarning
    subset.loc[:, 'relative_error'] = (subset['real_attn_time_ms'] - subset['sim_attn_time_ms']) / subset['real_attn_time_ms']
    
    # Add scatter trace for each (tp, cp) combination
    # Plotting the relative error vs batch size
    fig_bs_rel_error.add_trace(go.Scatter(
        x=subset['bs'], 
        y=subset['relative_error'],
        mode='markers',
        name=f'tp={tp}, cp={cp}',
        marker=dict(color=blended_color),
        hovertemplate='Batch Size: %{x}<br>Relative Error: %{y:.2%}<br>Total Len: %{customdata[0]}<br>Real Time: %{customdata[1]:.2f} ms<br>Sim Time: %{customdata[2]:.2f} ms<extra></extra>',
        customdata=subset[['total_len', 'real_attn_time_ms', 'sim_attn_time_ms']].values
    ))

# Customize the layout
fig_bs_rel_error.update_layout(
    title=(
        f'Relative Error of Simulated Attention Time vs Batch Size - {name} Dataset<br>'
        f'Dataset: {name}, Size: {size}, Max Context Length: {max_ctx_length}'
    ),
    xaxis_title='Batch Size',
    yaxis_title='Relative Error (Real - Sim) / Real',
    yaxis_tickformat='.1%',
    legend_title='TP and CP'
)

# Save the plot as an interactive HTML file
fig_bs_rel_error.write_html(f"plot/{name}.bs_rel_error_scatter.html")
# Save the plot as a PNG file
fig_bs_rel_error.write_image(f"plot/{name}.bs_rel_error_scatter.png")

# Show the plot
fig_bs_rel_error.show()

# %%
# Create a subplot for real and simulated attention time distribution
from plotly.subplots import make_subplots
import plotly.graph_objs as go

# Create a single figure with subplots for all (tp, cp) combinations
fig_time_dist = make_subplots(
    rows=len(tp_values), 
    cols=len(cp_values), 
    subplot_titles=[
        f'tp={tp}, cp={cp}' 
        for tp in tp_values 
        for cp in cp_values
    ],
    vertical_spacing=0.05,  # Increased vertical spacing
    horizontal_spacing=0.05
)

# Iterate through all (tp, cp) combinations
for row, tp in enumerate(tp_values, 1):
    for col, cp in enumerate(cp_values, 1):
        # Filter data for current (tp, cp) combination
        subset = df[(df['tp'] == tp) & (df['cp'] == cp)]
        real_times = subset['real_attn_time_ms']
        sim_times = subset['sim_attn_time_ms']
        total_len = subset['total_len'].max().item()

        # Add histogram for real attention times
        fig_time_dist.add_trace(
            go.Histogram(
                x=real_times, 
                name=f'Real Time (tp={tp}, cp={cp})',
                marker_color='blue',
                opacity=0.7
            ),
            row=row, col=col
        )

        # Add histogram for simulated attention times
        fig_time_dist.add_trace(
            go.Histogram(
                x=sim_times, 
                name=f'Sim Time (tp={tp}, cp={cp})',
                marker_color='red',
                opacity=0.7
            ),
            row=row, col=col
        )

        # Get MLP time for the current (tp, cp) combination
        from d2.timemodule import get_mlp_time
        mlp_time = get_mlp_time(
            x = total_len, tp = tp, cp = cp,
        )

        # Add vertical line for MLP time
        fig_time_dist.add_shape(
            type='line', 
            x0=mlp_time, 
            x1=mlp_time, 
            y0=0, 
            y1=1, 
            yref='paper',
            line=dict(color='green', width=2, dash='dot'),
            row=row, col=col
        )

        # Add annotation for MLP time
        fig_time_dist.add_annotation(
            x=mlp_time, 
            y=1, 
            xref='x', 
            yref='paper',
            text=f'MLP Time: {mlp_time:.2f} ms', 
            showarrow=True,
            arrowhead=1,
            row=row, col=col
        )

# Update layout for the entire figure
fig_time_dist.update_layout(
    title=(
        f'Distribution of Real vs Simulated Attention Times - {name} Dataset<br>'
        f'Dataset: {name}, Size: {size}, Max Context Length: {max_ctx_length}'
    ),
    height=300 * len(tp_values),
    width=300 * len(cp_values),
    showlegend=True
)

# Update x and y axis labels for all subplots
for row in range(1, len(tp_values) + 1):
    for col in range(1, len(cp_values) + 1):
        # Only show x-axis label for the last row
        if row == len(tp_values):
            fig_time_dist.update_xaxes(title_text='Time (ms)', row=row, col=col)
        else:
            fig_time_dist.update_xaxes(title_text='', row=row, col=col)
        
        fig_time_dist.update_yaxes(title_text='Frequency', row=row, col=col)

# Add buttons to toggle real and simulated time traces
fig_time_dist.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0.7,
            y=1.15,
            showactive=True,
            buttons=[
                dict(
                    label="Show Real Time",
                    method="update",
                    args=[{"visible": [True if 'Real' in trace.name else False for trace in fig_time_dist.data]}]
                ),
                dict(
                    label="Show Simulated Time", 
                    method="update",
                    args=[{"visible": [True if 'Sim' in trace.name else False for trace in fig_time_dist.data]}]
                ),
                dict(
                    label="Show All",
                    method="update", 
                    args=[{"visible": [True for trace in fig_time_dist.data]}]
                )
            ]
        )
    ]
)


# Save the plot as an interactive HTML file
fig_time_dist.write_html(f"plot/{name}.time_distribution.html")

# Save the plot as a PNG file
fig_time_dist.write_image(f"plot/{name}.time_distribution.png")

# Show the plot
fig_time_dist.show()
# %%
