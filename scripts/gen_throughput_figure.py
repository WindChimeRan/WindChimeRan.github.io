#!/usr/bin/env python3
"""vllm-metal vs mlx_lm output throughput on SiliconBench chat/agent splits.

Reads comparison.json for the chat and agent splits from the SiliconBench
results tree and writes a 2-panel bar chart into the blog's assets directory.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

RESULTS = Path("/Users/ran/workspace/applebench/results/Qwen3-0.6B")
OUT = Path(__file__).resolve().parent.parent / "assets" / "img" / "throughput_vllm_metal_vs_mlx_lm.png"

COLOR_MLX = "#ff7f0e"   # mlx_lm
COLOR_VLLM = "#d62728"  # vllm-metal
PARTIAL_THRESHOLD = 90  # match paper script


def load_throughput(split):
    with open(RESULTS / split / "comparison.json") as f:
        data = json.load(f)
    out = {}
    for fw in ("mlx_lm", "vllm_metal"):
        out[fw] = {}
        for c_result in data["results"][fw]["concurrency_results"]:
            c = c_result["concurrency"]
            tps = c_result.get("output_throughput_tps")
            out[fw][c] = {
                "tps": tps if tps is not None else 0.0,
                "successful": c_result.get("successful", 0),
                "ttft_p50_ms": c_result.get("ttft_p50_ms") or 0.0,
                "wall_s": c_result.get("wall_time_s") or 0.0,
            }
    return out


def emit_markdown_table(split_name, data):
    """Print a markdown table of stats for one split."""
    print(f"\n**{split_name} split**\n")
    print("| Engine | c | Success | TTFT p50 (ms) | Throughput (tok/s) | Wall (s) |")
    print("|---|---:|---:|---:|---:|---:|")
    label = {"mlx_lm": "mlx_lm", "vllm_metal": "vllm-metal"}
    for fw in ("mlx_lm", "vllm_metal"):
        for c in (1, 8, 16):
            d = data[fw][c]
            # Wall time is only comparable across engines when every prompt
            # succeeded; otherwise the run terminated on whatever fraction
            # the engine could serve, so the elapsed time is misleading.
            wall_cell = f"{d['wall_s']:.1f}" if d["successful"] == 100 else "-"
            print(
                f"| {label[fw]} | {c} | {d['successful']}/100 | "
                f"{d['ttft_p50_ms']:.0f} | {d['tps']:.1f} | {wall_cell} |"
            )


def render_panel(ax, data, title, ymax):
    concurrencies = [1, 8, 16]
    bar_width = 0.36

    for i, c in enumerate(concurrencies):
        mlx = data["mlx_lm"][c]
        vllm = data["vllm_metal"][c]

        x_mlx = i - bar_width / 2
        x_vllm = i + bar_width / 2

        bar_mlx = ax.bar(
            x_mlx, mlx["tps"], width=bar_width,
            color=COLOR_MLX, edgecolor="black", linewidth=0.4,
            label="mlx_lm" if i == 0 else None,
        )[0]
        if mlx["successful"] < PARTIAL_THRESHOLD:
            bar_mlx.set_hatch("///")
            ax.text(
                x_mlx, mlx["tps"] + ymax * 0.02,
                f"{mlx['successful']}/100",
                ha="center", va="bottom", fontsize=8.5, color="#333",
            )

        ax.bar(
            x_vllm, vllm["tps"], width=bar_width,
            color=COLOR_VLLM, edgecolor="black", linewidth=0.4,
            label="vllm-metal" if i == 0 else None,
        )

    ax.set_xticks(range(len(concurrencies)))
    ax.set_xticklabels([f"c={c}" for c in concurrencies])
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Output throughput (tok/s)")


def main():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })

    chat = load_throughput("chat")
    agent = load_throughput("agent")

    chat_max = max(d[c]["tps"] for d in chat.values() for c in (1, 8, 16))
    agent_max = max(d[c]["tps"] for d in agent.values() for c in (1, 8, 16))

    fig, (ax_chat, ax_agent) = plt.subplots(1, 2, figsize=(9, 3.5))
    render_panel(ax_chat, chat, "chat split", chat_max * 1.22)
    render_panel(ax_agent, agent, "agent split (~4K-token prompts)", agent_max * 1.22)

    handles, labels = ax_chat.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
        fontsize=10, columnspacing=2.4,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT)
    print(f"wrote {OUT}")

    # also print markdown tables for the blog post
    emit_markdown_table("chat", chat)
    emit_markdown_table("agent", agent)


if __name__ == "__main__":
    main()
