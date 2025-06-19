import d2.timemodule as tm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

K = 1024
M = 1024 * 1024

def test_get_attn_time_monotonically_increase():
    for tp in [1, 2, 4, 8]:
        for cp in [1, 2, 4, 8]:
            prev = 0
            for seq_len in [
                32, 64, 128, 256, 512,
                1 * K, 2 * K, 4 * K, 8 * K, 
                16 * K, 32 * K, 64 * K, 128 * K, 
            ]:
                a = tm.get_attn_time(seq_len, tp, cp)
                assert a >= prev, f"tp={tp}, cp={cp}, seq_len={seq_len}, a={a}, prev={prev} is not increasing: {a - prev = }"
                print(f"tp={tp}, cp={cp}, seq_len={seq_len}, a={a}")
    


def test_plot_get_attn_time():
    """Use matplotlib to plot the result of attn time"""
    plt_dir = Path(__file__).parent / "test_timemodule_plots"
    plt_dir.mkdir(parents=True, exist_ok=True)

    # Plotly plot
    import plotly.graph_objects as go

    fig = go.Figure()

    for tp in [1, 2, 4, 8]:
        for cp in [1, 2, 4, 8]:
            x = np.concatenate([
                np.arange(4, 1 * K, 4),
                np.arange(1 * K, 1 * M, 1 * K)
            ])
            x = np.sort(x)
            y = [tm.get_attn_time(x, tp, cp) for x in x]
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"tp={tp}, cp={cp}"))

    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    # add the toggle buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=1.0, y=1.15,
                showactive=True,
                buttons=[
                    dict(
                        label="Log-Log",
                        method="relayout",
                        args=[{"xaxis.type": "log", "yaxis.type": "log"}]
                    ),
                    dict(
                        label="Linear",
                        method="relayout",
                        args=[{"xaxis.type": "linear", "yaxis.type": "linear"}]
                    ),
                ],
            )
        ]
    )

    # add title
    fig.update_layout(title="Attention Time (ms)")
    # add x-axis and y-axis labels
    fig.update_xaxes(title="Sequence Length (tokens)")
    fig.update_yaxes(title="Time (ms)")

    fig.write_html(plt_dir / "attn_time.html")
    fig.write_image(plt_dir / "attn_time.png")

    plt.close()

def test_plot_get_mlp_time():
    """Use matplotlib to plot the result of mlp time"""
    plt_dir = Path(__file__).parent / "test_timemodule_plots"
    plt_dir.mkdir(parents=True, exist_ok=True)

    # Plotly plot
    import plotly.graph_objects as go

    fig = go.Figure()
    
    import d2.timemodule.compute as tm

    for tp in [1, 2, 4, 8]:
        for cp in [1, 2, 4, 8]:
            x = np.concatenate([
                np.arange(4, 1 * K, 4),
                np.arange(1 * K, 1 * M, 1 * K)
            ])
            x = np.sort(x)
            y = [tm.get_mlp_time(x, tp, cp) for x in x]
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"tp={tp}, cp={cp}"))

    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    # add the toggle buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=1.0, y=1.15,
                showactive=True,
                buttons=[
                    dict(
                        label="Log-Log",
                        method="relayout",
                        args=[{"xaxis.type": "log", "yaxis.type": "log"}]
                    ),
                    dict(
                        label="Linear",
                        method="relayout",
                        args=[{"xaxis.type": "linear", "yaxis.type": "linear"}]
                    ),
                ],
            )
        ]
    )

    # add title
    fig.update_layout(title="MLP Time (ms)")
    # add x-axis and y-axis labels
    fig.update_xaxes(title="Sequence Length (tokens)")
    fig.update_yaxes(title="Time (ms)")

    fig.write_html(plt_dir / "mlp_time.html")
    fig.write_image(plt_dir / "mlp_time.png")

    plt.close()
    

if __name__ == "__main__":
    test_get_attn_time_monotonically_increase()
    test_plot_get_attn_time()
    test_plot_get_mlp_time()