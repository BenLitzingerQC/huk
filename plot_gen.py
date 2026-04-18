"""Standalone plotting functions for the business-value PR analysis.

Two functions, one per plot:
  - plot_pr_value_landscape: PR-curve overlaid on a net-value heatmap with iso-lines
  - plot_efficient_frontier:  Recall vs. net-value curve with optimum marker

Both take the same input arrays and business parameters and draw onto a
user-provided Axes, so they can be dropped into any figure.
"""

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

ACCENT = "#FFCC00"  # HUK yellow (highlight)
GREY = "#cccccc"
GREEN = "#2a9d8f"


def _net_value(precision, recall, total_pos, review_cost, savings_per_tp):
    """NV = TP * savings - (TP+FP) * review_cost, with TP = recall * total_pos."""
    tp = recall * total_pos
    total_flagged = tp / precision if precision > 0 else 0
    return tp * savings_per_tp - total_flagged * review_cost


def plot_pr_value_landscape(
    ax,
    precision,        # list/array of precision per greedy step
    recall,           # list/array of recall per greedy step
    steps,            # list of step numbers (for the Optimum label)
    total_pos,        # total positives in the dataset
    review_cost,      # € cost per flagged document (Sachbearbeiterprüfung)
    savings_per_tp,   # € expected savings per true positive
    best_i=None,      # index of optimum step; computed from NV if None
):
    """Precision-Recall curve on top of a € net-value heatmap.

    The background shades the profit zone (NV > 0), iso-lines mark levels of
    equal net value, and the rule-curve shows how each greedy step trades off
    precision and recall. Yellow marker = business optimum (max NV).
    """
    be_prec = review_cost / savings_per_tp
    nv = [_net_value(p, r, total_pos, review_cost, savings_per_tp)
          for p, r in zip(precision, recall)]
    if best_i is None:
        best_i = int(np.argmax(nv))

    # Net-value surface over the full (recall, precision) plane.
    rr = np.linspace(0.01, 1.0, 200)
    pp = np.linspace(0.01, 1.0, 200)
    RR, PP = np.meshgrid(rr, pp)
    NV = RR * total_pos * (savings_per_tp - review_cost / PP)
    max_gain = total_pos * savings_per_tp

    # Profit-only heatmap (loss zone stays white).
    ax.contourf(RR, PP, np.where(NV > 0, NV, np.nan),
                levels=40, cmap="Greens", vmin=0, vmax=max_gain,
                alpha=0.6, zorder=0)

    # Iso-lines in round € steps (5K / 10K / 25K / ... whichever gives <=8 lines).
    for s in (5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000):
        if max_gain / s <= 8:
            step_size = s
            break
    else:
        step_size = 1_000_000
    iso_levels = np.arange(step_size, max_gain + step_size, step_size)
    ax.contour(RR, PP, NV, levels=iso_levels, colors="#444444",
               linewidths=0.6, alpha=0.6, zorder=1)

    # Place iso-labels in a clean row at a fixed low precision where the
    # rule-curve does not pass (avoids label-on-curve collisions).
    target_prec = 0.45
    denom = savings_per_tp - review_cost / target_prec
    if denom > 0:
        for v in iso_levels:
            rec = v / (total_pos * denom)
            if 0.12 < rec < 0.85:
                ax.text(rec, target_prec, f"{v/1000:.0f}K €",
                        fontsize=7, color="#333333",
                        ha="center", va="center", rotation=-30,
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="white", edgecolor="none",
                                  alpha=0.75),
                        zorder=2)

    # Black break-even iso-line (NV = 0).
    cs0 = ax.contour(RR, PP, NV, levels=[0], colors="black",
                     linewidths=1.0, zorder=2)
    ax.clabel(cs0, inline=True, fontsize=8, fmt={0: "Break-even (0 €)"})

    # Rule curve: line + dots (grey below break-even, dark above, yellow at optimum).
    ax.plot(recall, precision, color="#333333", linewidth=1.2, zorder=3)
    dot_colors = [
        ACCENT if i == best_i else ("#333333" if precision[i] >= be_prec else GREY)
        for i in range(len(steps))
    ]
    ax.scatter(recall, precision, c=dot_colors, s=55, zorder=4,
               edgecolors="white", linewidths=1)

    # Highlight optimum point.
    ax.scatter([recall[best_i]], [precision[best_i]], color=ACCENT, s=120,
               zorder=5, edgecolors="white", linewidths=1.5)
    ax.annotate(
        f"Optimum\nStep {steps[best_i]}\n"
        f"Prec={precision[best_i]:.0%}, Rec={recall[best_i]:.0%}",
        xy=(recall[best_i], precision[best_i]), xycoords="data",
        xytext=(0.05, 0.72), textcoords="axes fraction",
        fontsize=8, color="black", fontweight="bold",
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#999999", linewidth=0.8),
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1),
        zorder=6,
    )

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall mit Netto-Wert-Landschaft\n"
                 "(Iso-Linien = gleicher € Netto-Wert)",
                 loc="left", fontweight="bold", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            mpatches.Patch(color="#4daf4a", alpha=0.6, label="Gewinn-Zone"),
            mpatches.Patch(color=ACCENT, label="Business-Optimum"),
        ],
        fontsize=8, frameon=True, facecolor="white",
        edgecolor="#cccccc", loc="lower left",
    )


def plot_efficient_frontier(
    ax,
    precision,        # list/array of precision per greedy step
    recall,           # list/array of recall per greedy step
    total_pos,        # total positives in the dataset
    review_cost,      # € cost per flagged document
    savings_per_tp,   # € expected savings per true positive
    best_i=None,      # index of optimum step; computed from NV if None
):
    """Recall vs. achievable net-value, with the optimum marked.

    Shows — for each greedy step — the € net value the composition would
    generate. The peak is the business optimum: adding more rules past it
    costs more review effort than it saves.
    """
    nv = [_net_value(p, r, total_pos, review_cost, savings_per_tp)
          for p, r in zip(precision, recall)]
    if best_i is None:
        best_i = int(np.argmax(nv))

    # Frontier curve + dots (green in profit, grey in loss, yellow at optimum).
    ax.plot(recall, nv, color=GREY, linewidth=1.5, zorder=2)
    ax.scatter(recall, nv, zorder=3, s=60,
               c=[ACCENT if i == best_i else (GREEN if nv[i] > 0 else GREY)
                  for i in range(len(nv))])
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")

    # Highlight optimum with annotation box.
    ax.scatter([recall[best_i]], [nv[best_i]], color=ACCENT, s=140,
               zorder=5, edgecolors="white", linewidths=1.5)
    ax.annotate(
        f"Optimum\nRecall={recall[best_i]:.0%}\n{nv[best_i]:,.0f} €",
        xy=(recall[best_i], nv[best_i]), xycoords="data",
        xytext=(recall[best_i] - 0.35, nv[best_i]), textcoords="data",
        fontsize=8, color="black", fontweight="bold",
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#999999", linewidth=0.8),
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1),
        zorder=6,
    )

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Netto-Wert (€)", fontsize=11)
    ax.set_title("Efficient Frontier\n(Recall vs. erzielbarer Netto-Wert)",
                 loc="left", fontweight="bold", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f"{x / 1000:.0f}K €" if abs(x) >= 1000 else f"{x:.0f} €"
    ))


# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Replace with your real greedy-trace values.
    rng = np.random.default_rng(42)
    recall = np.linspace(0.05, 0.97, 30)
    precision = np.clip(
        0.92 / (1 + np.exp(18 * (recall - 0.99))) + 0.06
        + rng.normal(0, 0.008, 30), 0.01, 0.99)
    steps = list(range(1, 31))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_pr_value_landscape(axes[0], precision, recall, steps,
                            total_pos=1000, review_cost=25, savings_per_tp=150)
    plot_efficient_frontier(axes[1], precision, recall,
                            total_pos=1000, review_cost=25, savings_per_tp=150)
    plt.tight_layout()
    plt.savefig("analysis/business_plots_demo.png", bbox_inches="tight", dpi=150)
