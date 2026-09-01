"""Shared plotting style for the vllm-metal v0.4.0 blog figures."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

INK, MUTED, GRID, PAD = "#1a1a2e", "#5a5a72", "#e4e4ec", "#d3d5dc"
# Categorical slots 1-3 of a validated palette, assigned in fixed order.
# Checked with the six-check validator on the all-pairs list (light surface):
# worst CVD dE 9.2, worst normal-vision dE 24.0, all contrast >= 3:1.
# The previous blue/violet pairing in the machine-comparison figure was a hard
# fail: normal-vision dE 13.0, below the 15 floor, so full-colour readers could
# not separate them either. ORANGE now carries the second series in every
# figure; llama.cpp and DGX Spark never share an axis, and both legends name
# their arms, so the reuse costs less than the failing pair did.
BLUE, ORANGE, TEAL = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.size": 12, "text.color": INK, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "svg.fonttype": "none", "figure.facecolor": "white", "axes.facecolor": "white",
    "svg.hashsalt": "vllm-metal-v0.4.0",
})

SVG_METADATA = {"Date": None}


def log2_x(xlabels):
    """X positions for a geometric concurrency sweep.

    The sweep points are powers of two, so equal spacing would draw 1->8 (three
    doublings) the same width as 8->16 (one), and slope would mean nothing.
    Placing them at log2 makes a slope read as change per doubling of load.
    """
    from math import log2
    return [log2(float(l)) for l in xlabels]


def save_figure(fig, stem, *, png=True):
    svg_path = f"{OUT}/{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight", metadata=SVG_METADATA)
    with open(svg_path, encoding="utf-8") as f:
        svg = f.read()
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
    if png:
        fig.savefig(f"{OUT}/{stem}.png", bbox_inches="tight", dpi=170)


def draw_panels(fig, axes, series, x, xlabels, panels, direct_labels=False):
    """series: (label, color, linestyle, {metric: values}). Truncated series
    (fewer points than x) simply stop early.

    Markers are uniformly filled. Partial completion is a property of the
    prompts, not of the arm serving them, so it is stated in the caption rather
    than encoded on every point.
    """
    for ax, key, title in panels:
        for label, color, ls, data in series:
            ys = data[key]
            xs = x[:len(ys)]
            ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2.2, zorder=3)
            ax.plot(xs, ys, "o", ms=7, color=color, zorder=4)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xticks(x, xlabels)
        ax.set_xlabel("concurrency")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if direct_labels:
            _label_line_ends(ax, series, key, x)


def _label_line_ends(ax, series, key, x, short=None):
    """Name each line at its own end, so identity does not depend on the legend.

    Reserves a right gutter and nudges labels apart when two lines finish close
    together, which the vllm-metal baseline and its MTP arm routinely do.
    """
    span = x[-1] - x[0]
    ax.set_xlim(x[0] - span * 0.06, x[-1] + span * 0.62)
    lo, hi = ax.get_ylim()
    ends = sorted(
        ((s[3][key][-1], (short or {}).get(s[0], s[0]), s[1], x[len(s[3][key]) - 1])
         for s in series),
        key=lambda t: t[0],
    )
    gap = (hi - lo) * 0.085
    placed = []
    for value, label, color, xi in ends:
        y = value if not placed else max(value, placed[-1] + gap)
        placed.append(y)
        # anchor to the series' own last point, so a truncated arm is labelled
        # where it actually stops rather than out in the shared gutter
        ax.annotate(label, xy=(xi, value), xytext=(xi + span * 0.07, y),
                    color=color, fontsize=10, fontweight="bold",
                    va="center", ha="left", annotation_clip=False)
    ax.set_ylim(lo, max(hi, placed[-1] + gap * 0.5))


def add_legend(fig, series, y=1.0):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, linestyle=ls, linewidth=2.2,
                      marker="o", markersize=8, markerfacecolor=c,
                      markeredgecolor=c, markeredgewidth=2.2, label=label)
               for label, c, ls, *_ in series]
    fig.legend(handles=handles, loc="upper center", ncol=len(series),
               frameon=False, bbox_to_anchor=(0.5, y), fontsize=11.5,
               handlelength=2.6, columnspacing=2.0)
