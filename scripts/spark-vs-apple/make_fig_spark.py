"""Apple M5 Pro vs DGX Spark figure for the vllm-metal v0.4.0 blog post.

One serving architecture on two unified-memory machines, same SiliconBench agent
split, same 100 prompts, same client. The Apple arm is vllm-metal (MLX + Metal);
the Spark arm is upstream vLLM (CUDA) on GB10.

Data source: SiliconBench (github.com/WindChimeRan/SiliconBench), agent split.
  M5 Pro:     results/<MODEL>/m5pro/agent/vllm_metal_*.json
                Qwen3.5-0.8B   vllm_metal_20260821_121501.json
                Gemma-4-E4B-it vllm_metal_20260822_050741.json
                Qwen3.8-27B    vllm_metal_20260824_171825.json
  DGX Spark:  results/<MODEL>/dgxspark/agent/vllm_*.json
                Qwen3.5-0.8B   vllm_20260827_004302.json
                Gemma-4-E4B-it vllm_20260827_005037.json
                Qwen3.8-27B    vllm_20260827_013402.json

Rows carry their own concurrency axis. The two small models sweep 1/8/16. The
27B sweeps 1/2/4 because that is where the 64 GB Mac ran out of headroom, and a
Spark point above 4 would have no Apple counterpart to sit beside.
"""
import matplotlib.pyplot as plt

from _style import BLUE, ORANGE, MUTED, save_figure, add_legend, log2_x

# (model label, xlabels, apple, spark)
# Qwen3.5-0.8B is measured but not plotted: at 0.1-0.25s TTFT on both machines
# the comparison carries no deployment decision, and it was the Mac's closest
# row, so dropping it costs vllm-metal rather than flattering it.
ROWS = [
    (
        "Gemma 4 E4B",
        ["1", "8", "16"],
        {"ttft": [0.71, 1.14, 1.89], "out": [21.06, 52.32, 63.61], "latency": [4.44, 13.74, 21.66]},
        {"ttft": [0.62, 0.19, 0.21], "out": [15.68, 133.94, 225.13], "latency": [5.73, 4.90, 5.48]},
    ),
    (
        "Qwen3.8-27B\n8-bit",
        ["1", "2", "4"],
        {"ttft": [10.15, 11.88, 14.66], "out": [3.94, 4.70, 5.25], "latency": [17.92, 29.56, 52.70]},
        {"ttft": [1.77, 0.58, 0.62], "out": [6.11, 13.30, 24.02], "latency": [11.63, 10.61, 11.18]},
    ),
]

# Two seconds-valued panels first, then the rate: one change of direction
# instead of two, and TTFT sits beside the latency it is a component of.
METRICS = [
    ("ttft", "TTFT avg (s) · lower is better"),
    ("latency", "End-to-end latency avg (s) · lower is better"),
    ("out", "Output token throughput (tok/s) · higher is better"),
]

series_legend = [
    ("Apple M5 Pro 64 GB · vllm-metal", BLUE, "-", None),
    ("DGX Spark 128 GB · vLLM", ORANGE, "-", None),
]

fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.8))

for r, (model, xlabels, apple, spark) in enumerate(ROWS):
    x = log2_x(xlabels)
    for cidx, (key, title) in enumerate(METRICS):
        ax = axes[r][cidx]
        for data, color in ((apple, BLUE), (spark, ORANGE)):
            ys = data[key]
            ax.plot(x[:len(ys)], ys, color=color, linestyle="-", linewidth=2.2, zorder=3)
            ax.plot(x[:len(ys)], ys, "o", ms=7, color=color, zorder=4)
        if r == 0:
            ax.set_title(title, fontsize=12, pad=10)
        ax.set_xticks(x, xlabels)
        ax.set_xlabel("concurrency")
        ax.grid(axis="y", color="#e4e4ec", linewidth=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if cidx == 0:
            ax.set_ylabel(model, fontsize=12.5, fontweight="bold", labelpad=12)

add_legend(fig, series_legend, y=1.0)
fig.suptitle("SiliconBench agent split · one serving architecture, two unified-memory machines",
             fontsize=13, fontweight="bold", y=1.035)
fig.text(0.5, -0.062,
         "Same 100 prompts, same measurement client, closed loop. The 27B row sweeps 1/2/4 because that is where the 64 GB Mac ran "
         "out of headroom; it compares an 8-bit MLX conversion on the Mac with Qwen's FP8 checkpoint on the Spark.",
         ha="center", fontsize=10.5, color=MUTED, wrap=True)
fig.tight_layout(rect=(0, 0.012, 1, 1))
save_figure(fig, "spark-vs-apple", png=False)
plt.close(fig)
print("done")
