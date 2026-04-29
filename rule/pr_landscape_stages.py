"""Step-by-step construction of the PR-value landscape plot.

Returns a list of five figures that incrementally add:
  1. PR curve + rule points
  2. + positive prediction counts
  3. + break-even precision
  4. + net-value iso contours and profit zone
  5. + business optimum highlight
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from evaluate_rule import ACCENT, GREY, _net_value, _window_name_to_subtitle


def _base_ax(window_name=None, review_cost=None, avg_savings_per_dz=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        "Precision-recall trade-off",
        loc="left",
        fontweight="bold",
        pad=25,
    )
    if window_name and review_cost is not None and avg_savings_per_dz is not None:
        ax.text(
            x=0,
            y=1.02,
            s=f"Review cost={review_cost:.2f} EUR, Avg. saving per DZ={avg_savings_per_dz:.2f} EUR, {_window_name_to_subtitle(window_name)}",
            transform=ax.transAxes,
            fontsize=10,
            ha="left",
            va="bottom",
        )
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def _draw_curve_and_dots(ax, precision, recall, dot_colors):
    ax.plot(recall, precision, color="#333333", linewidth=1.2, zorder=3)
    ax.scatter(
        recall,
        precision,
        c=dot_colors,
        s=55,
        zorder=4,
        edgecolors="white",
        linewidths=1,
    )


def _annotate_counts(ax, recall, precision, num_pos_preds):
    bbox_props = dict(boxstyle="round,pad=0.2", fc="lightgrey", ec="none", alpha=0.8)
    for r, p, n in zip(recall, precision, num_pos_preds):
        ax.annotate(
            str(n),
            (r, p),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            bbox=bbox_props,
            zorder=7,
        )


def _draw_break_even(ax, break_even_prec):
    ax.axhline(
        break_even_prec,
        color="black",
        linewidth=1.0,
        linestyle="--",
        zorder=2,
    )
    ax.text(
        0.02,
        break_even_prec + 0.01,
        f"Break-Even precision ({break_even_prec:.1%})",
        fontsize=8,
        color="black",
        ha="left",
        va="bottom",
        zorder=3,
    )


def _draw_iso_contours(ax, total_pos, review_cost, avg_savings_per_dz):
    recall_axis = np.linspace(0.01, 1.0, 200)
    precision_axis = np.linspace(0.01, 1.0, 200)
    recall_grid, precision_grid = np.meshgrid(recall_axis, precision_axis)
    NV = recall_grid * total_pos * (avg_savings_per_dz - review_cost / precision_grid)
    max_gain = total_pos * avg_savings_per_dz

    contourf_step_size = 2000
    ax.contourf(
        recall_grid,
        precision_grid,
        np.where(NV > 0, NV, np.nan),
        levels=np.arange(0, max_gain + contourf_step_size, contourf_step_size),
        cmap="Greens",
        vmin=0,
        vmax=max_gain,
        alpha=0.6,
        zorder=0,
    )

    iso_step_size = 10_000
    iso_levels = np.arange(iso_step_size, max_gain + iso_step_size, iso_step_size)
    ax.contour(
        recall_grid,
        precision_grid,
        NV,
        levels=iso_levels,
        colors="#444444",
        linewidths=0.6,
        alpha=0.6,
        zorder=1,
    )

    target_prec = 0.5
    denom = avg_savings_per_dz - review_cost / target_prec
    for v in iso_levels:
        if denom <= 0:
            break
        rec = v / (total_pos * denom)
        if 0.05 < rec < 0.95:
            ax.text(
                rec,
                target_prec,
                f"{v / 1000:.0f}K €",
                fontsize=7,
                color="#333333",
                ha="center",
                va="center",
                rotation=-30,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                ),
                zorder=2,
            )


def _highlight_optimum(ax, recall, precision, best_i):
    ax.scatter(
        [recall[best_i]],
        [precision[best_i]],
        color=ACCENT,
        s=120,
        zorder=5,
        edgecolors="white",
        linewidths=1.5,
    )
    ax.annotate(
        f"Optimum\nPrec={precision[best_i]:.0%}, Rec={recall[best_i]:.0%}",
        xy=(recall[best_i], precision[best_i]),
        xycoords="data",
        xytext=(0.6, 0.3),
        textcoords="axes fraction",
        fontsize=8,
        color="black",
        fontweight="bold",
        ha="left",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="#999999",
            linewidth=0.8,
        ),
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1),
        zorder=6,
    )


def build_pr_value_landscape_stages(
    precision,
    recall,
    num_pos_preds,
    total_pos,
    review_cost,
    avg_savings_per_dz,
    best_i=None,
    window_name=None,
):
    """Returns a list of 5 figures that incrementally build the full PR-value landscape."""
    break_even_prec = review_cost / avg_savings_per_dz
    if best_i is None:
        nv = [
            _net_value(p, r, total_pos, review_cost, avg_savings_per_dz)
            for p, r in zip(precision, recall)
        ]
        best_i = int(np.argmax(nv))

    def dot_colors(include_optimum=False, include_break_even=False):
        colors = []
        for i in range(len(precision)):
            if include_optimum and i == best_i:
                colors.append(ACCENT)
            elif include_break_even and precision[i] >= break_even_prec:
                colors.append("#333333")
            else:
                colors.append(GREY)
        return colors

    figures = []

    # Stage 1 — PR curve + rule points
    fig, ax = _base_ax(window_name, review_cost, avg_savings_per_dz)
    _draw_curve_and_dots(ax, precision, recall, dot_colors())
    figures.append(fig)

    # Stage 2 — + counts
    fig, ax = _base_ax(window_name, review_cost, avg_savings_per_dz)
    _draw_curve_and_dots(ax, precision, recall, dot_colors())
    _annotate_counts(ax, recall, precision, num_pos_preds)
    figures.append(fig)

    # Stage 3 — + break-even
    fig, ax = _base_ax(window_name, review_cost, avg_savings_per_dz)
    _draw_break_even(ax, break_even_prec)
    _draw_curve_and_dots(ax, precision, recall, dot_colors(include_break_even=True))
    _annotate_counts(ax, recall, precision, num_pos_preds)
    figures.append(fig)

    # Stage 4 — + iso contours + profit zone
    fig, ax = _base_ax(window_name, review_cost, avg_savings_per_dz)
    _draw_iso_contours(ax, total_pos, review_cost, avg_savings_per_dz)
    _draw_break_even(ax, break_even_prec)
    _draw_curve_and_dots(ax, precision, recall, dot_colors(include_break_even=True))
    _annotate_counts(ax, recall, precision, num_pos_preds)
    ax.legend(
        handles=[mpatches.Patch(color="#4daf4a", alpha=0.6, label="Profit zone")],
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        loc="lower left",
    )
    figures.append(fig)

    # Stage 5 — + optimum highlight
    fig, ax = _base_ax(window_name, review_cost, avg_savings_per_dz)
    _draw_iso_contours(ax, total_pos, review_cost, avg_savings_per_dz)
    _draw_break_even(ax, break_even_prec)
    _draw_curve_and_dots(
        ax,
        precision,
        recall,
        dot_colors(include_optimum=True, include_break_even=True),
    )
    _annotate_counts(ax, recall, precision, num_pos_preds)
    _highlight_optimum(ax, recall, precision, best_i)
    ax.legend(
        handles=[
            mpatches.Patch(color="#4daf4a", alpha=0.6, label="Profit zone"),
            mpatches.Patch(color=ACCENT, label="Optimum"),
        ],
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        loc="lower left",
    )
    figures.append(fig)

    return figures
