import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from da_hf5_utils.db2 import get_engine
from sqlalchemy import text
from sqlalchemy.types import VARCHAR

from historic_precision import (
    combined_precision,
    cumulative_precision,
    estimate_historic_precision,
    expected_savings_curve,
    labeled_cumulative_precision,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    stream=sys.stdout,
)

ACCENT = "#FFCC00"  # HUK yellow (highlight)
GREY = "#cccccc"
GREEN = "#2a9d8f"

WINDOW_NOV_OCT = (
    pl.col("HUKIMPORTTIME").dt.year().eq(2025)
    & pl.col("HUKIMPORTTIME").dt.month().eq(11)
    & (
        (
            pl.col("HUKIMPORTTIME__2").dt.year().eq(2025)
            & pl.col("HUKIMPORTTIME__2").dt.month().eq(11)
        )
        | (
            pl.col("HUKIMPORTTIME__2").dt.year().eq(2025)
            & pl.col("HUKIMPORTTIME__2").dt.month().eq(10)
        )
    )
)

WINDOW_NOV_2YR = (
    pl.col("HUKIMPORTTIME").dt.year().eq(2025)
    & pl.col("HUKIMPORTTIME").dt.month().eq(11)
    & pl.col("HUKIMPORTTIME__2").ge(pl.datetime(2023, 11, 1))
    & pl.col("HUKIMPORTTIME__2").lt(pl.datetime(2025, 12, 1))
)

EVAL_WINDOWS = [
    ("November_2025__October_2025", WINDOW_NOV_OCT),
    ("November_2025__2_years_historic", WINDOW_NOV_2YR),
]


def generate_lar_heatmap(data):
    heat_agg = (
        data.group_by(["lar", "lar__2"])
        .agg(
            pl.len().alias("N_TOTAL"),
            pl.col("labels").sum().alias("TOTAL_POS"),
            pl.col("or_mask").sum().alias("POS_PRED"),
            pl.all_horizontal("labels", "or_mask").sum().alias("TP"),
        )
        .with_columns(
            (pl.col("TP") / pl.col("POS_PRED") * 100).fill_nan(0).alias("PREC_PCT"),
            (pl.col("TP") / pl.col("TOTAL_POS") * 100).fill_nan(0).alias("REC_PCT"),
        )
    )

    total_mat = (
        heat_agg.pivot(
            index="lar", on="lar__2", values="N_TOTAL", aggregate_function="first"
        )
        .fill_null(0)
        .sort("lar")
    )
    tp_mat = (
        heat_agg.pivot(
            index="lar", on="lar__2", values="TP", aggregate_function="first"
        )
        .fill_null(0.0)
        .sort("lar")
    )
    pos_mat = (
        heat_agg.pivot(
            index="lar", on="lar__2", values="TOTAL_POS", aggregate_function="first"
        )
        .fill_null(0.0)
        .sort("lar")
    )
    pos_pred_mat = (
        heat_agg.pivot(
            index="lar", on="lar__2", values="POS_PRED", aggregate_function="first"
        )
        .fill_null(0.0)
        .sort("lar")
    )
    prec_mat = (
        heat_agg.pivot(
            index="lar", on="lar__2", values="PREC_PCT", aggregate_function="first"
        )
        .fill_null(float("nan"))
        .sort("lar")
    )
    rec_mat = (
        heat_agg.pivot(
            index="lar", on="lar__2", values="REC_PCT", aggregate_function="first"
        )
        .fill_null(float("nan"))
        .sort("lar")
    )

    lars_ordered = total_mat["lar"].to_list()
    sorted_cols = [c for c in lars_ordered if c in total_mat.columns]

    totals = total_mat.select(sorted_cols).to_numpy()
    tps = tp_mat.select(sorted_cols).to_numpy()
    pos = pos_mat.select(sorted_cols).to_numpy()
    precs = prec_mat.select(sorted_cols).to_numpy()
    recs = rec_mat.select(sorted_cols).to_numpy()
    pos_preds = pos_pred_mat.select(sorted_cols).to_numpy()

    row_labels = lars_ordered
    col_labels = sorted_cols

    annot = np.empty_like(totals, dtype=object)
    for i in range(totals.shape[0]):
        for j in range(totals.shape[1]):
            if totals[i, j] > 0:
                prec_val = precs[i, j]
                rec_val = recs[i, j]
                prec_str = f"{prec_val:.2f}%" if not np.isnan(prec_val) else "N/A"
                rec_str = f"{rec_val:.2f}%" if not np.isnan(rec_val) else "N/A"
                annot[i, j] = (
                    f"{int(tps[i, j])}/{int(pos[i, j])}\n"
                    f"{int(pos_preds[i, j])}/{int(totals[i, j])}\n"
                    f"P: {prec_str}\nR: {rec_str}"
                )
            else:
                annot[i, j] = ""

    fig = plt.figure(
        figsize=(max(10, len(col_labels) * 0.7), max(8, len(row_labels) * 0.5))
    )
    ax = sns.heatmap(
        pos,
        annot=annot,
        annot_kws={"fontsize": 8},
        fmt="",
        cmap="huk_heatmap",
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"label": "Number of DZs per LAR combination"},
        linewidths=3.5,
    )
    ax.set_xlabel("LAR Doc. 2")
    ax.set_ylabel("LAR Doc. 1")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    return fig


def generate_time_difference_plot(data):
    """Generates a stacked bar chart showing recall on time difference of pairs."""
    data = data.with_columns(
        (pl.col("HUKIMPORTTIME") - pl.col("HUKIMPORTTIME__2"))
        .dt.total_days()
        .alias("time_difference")
    ).filter(
        pl.col("time_difference").is_not_null(),
        pl.col("time_difference").gt(0),
        pl.col("time_difference").lt(730),
    )

    all_preds = (
        data.filter(pl.col("or_mask"))
        .get_column("time_difference")
        .cast(pl.Int32)
        .to_numpy()
    )
    true_pos = (
        data.filter(pl.col("or_mask") & pl.col("labels"))
        .get_column("time_difference")
        .cast(pl.Int32)
        .to_numpy()
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    if len(all_preds) == 0:
        ax.text(
            0.5, 0.5, "No positive predictions", transform=ax.transAxes, ha="center"
        )
        ax.set_title("Time Difference between Positive Predictions")
        return fig

    max_day = int(max(all_preds.max(), true_pos.max() if len(true_pos) > 0 else 0))
    bins = range(0, max_day + 2)

    counts_all, _ = np.histogram(all_preds, bins=bins)
    counts_tp, _ = np.histogram(true_pos, bins=bins)

    ax.bar(
        bins[:-1],
        counts_all,
        width=1.0,
        color="lightgray",
        label="Positive Predictions",
    )
    ax.bar(bins[:-1], counts_tp, width=1.0, color="green", label="True Positives")

    ax.set_xlabel("Time Difference in Days (Doc. 1 - Doc. 2)")
    ax.set_ylabel("Count")
    ax.set_title(
        "Time Difference of Positive Predictions for Business Optimal Rule",
        loc="left",
        pad=25,
    )
    ax.text(
        x=0,
        y=1.02,
        s="",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="normal",
        ha="left",
        va="bottom",
    )
    ax.legend()

    return fig


def plot_savings_by_time_difference(enriched: pl.DataFrame, window_name: str):
    """Average MIN_MAX_REIMBURSEMENT binned by |Doc.1 − Doc.2| days.

    Two lines on the same axis:
      - All labeled positives (true DZ distribution)
      - True positives — i.e. labeled positives caught by or_mask
    FPs are absent by construction (only positives are enriched).
    """
    df = enriched.with_columns(
        (pl.col("HUKIMPORTTIME") - pl.col("HUKIMPORTTIME__2"))
        .dt.total_days()
        .alias("time_difference")
    ).filter(
        pl.col("time_difference").is_not_null(),
        pl.col("time_difference").gt(0),
        pl.col("time_difference").lt(730),
        pl.col("MIN_MAX_REIMBURSEMENT").is_not_null(),
    )

    bin_width = 14
    df = df.with_columns(
        (pl.col("time_difference") // bin_width * bin_width).alias("td_bin")
    )

    def _binned_mean(d: pl.DataFrame) -> pl.DataFrame:
        return (
            d.group_by("td_bin")
            .agg(
                pl.col("MIN_MAX_REIMBURSEMENT").mean().alias("avg_savings"),
                pl.len().alias("n"),
            )
            .sort("td_bin")
        )

    all_pos = _binned_mean(df)
    tp_only = _binned_mean(df.filter(pl.col("OR_MASK").cast(pl.Boolean)))

    fig, ax = plt.subplots(figsize=(12, 6))

    if all_pos.is_empty():
        ax.text(0.5, 0.5, "No enriched positives", transform=ax.transAxes, ha="center")
        ax.set_title("Average savings per time difference")
        return fig

    ax.plot(
        all_pos["td_bin"].to_numpy(),
        all_pos["avg_savings"].to_numpy(),
        color=GREEN, linewidth=1.4, marker="o", markersize=4,
        label="All labeled positives",
    )
    if not tp_only.is_empty():
        ax.plot(
            tp_only["td_bin"].to_numpy(),
            tp_only["avg_savings"].to_numpy(),
            color=ACCENT, linewidth=1.4, marker="o", markersize=4,
            label="True positives (rule hits)",
        )

    ax.set_xlabel("Time difference in days (Doc. 1 − Doc. 2), binned")
    ax.set_ylabel("Avg. MIN_MAX_REIMBURSEMENT (€)")
    ax.set_title(
        "Average DZ savings per time difference",
        loc="left", fontweight="bold", pad=25,
    )
    ax.text(
        x=0, y=1.02,
        s=f"Bin width={bin_width}d, {_window_name_to_subtitle(window_name)}",
        transform=ax.transAxes, fontsize=10, ha="left", va="bottom",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        loc="best",
    )

    return fig


def savings_analysis(data: pl.DataFrame, label_col: str):
    """
    Fetches MAX_REIMBURSEMENT for all positives (label=True) in one DB round-trip,

    then computes aggregates for two subsets:
      - all_pos:  every labelled positive regardless of or_mask
      - tp_only:  true positives (label=True AND or_mask=True)
    Returns
    -------
    {
        "all_pos":        {"total_savings": float, "average_savings": float,
                           "both_paid_total": float, "both_paid_avg": float},
        "tp_only":        {"total_savings": float, "average_savings": float,
                           "both_paid_total": float, "both_paid_avg": float},
    }
    The "average_savings" of "all_pos" is intended as the data-driven
    savings_per_tp estimate for the PR-value landscape.
    """
    data = data.drop(["LAR", "LAR__2"])
    positives = data.filter(pl.col(label_col).cast(pl.Boolean).fill_null(False))

    conn_sw = get_engine(database="spielwiese")
    raw_ids = ["StackID", "DocID", "SubDocID", "StackID__2", "DocID__2", "SubDocID__2"]

    logging.info(
        f"savings_analysis: total rows={data.height}, positives={positives.height}"
    )

    with conn_sw.begin() as conn:
        try:
            conn.execute(text("DROP TABLE DA00249.TEMP_DZ_SAVINGS"))
            logging.info("savings_analysis: dropped existing TEMP_DZ_SAVINGS")
        except Exception as e:
            logging.info(f"savings_analysis: no table to drop (expected): {e}")

    logging.info(f"savings_analysis: writing {positives.height} positive rows")
    positives.write_database(
        table_name="DA00249.TEMP_DZ_SAVINGS",
        connection=conn_sw,
        engine_options={"dtype": {c: VARCHAR(length=50) for c in raw_ids}},
    )
    logging.info("savings_analysis: write complete")

    # Create indices for performance
    with conn_sw.connect() as conn:
        conn.execute(
            text("""
            CREATE INDEX idx_temp_ids
            ON DA00249.TEMP_DZ_SAVINGS ("StackID", "DocID", "SubDocID")
        """)
        )
        conn.execute(
            text("""
            CREATE INDEX idx_temp_ids_2
            ON DA00249.TEMP_DZ_SAVINGS ("StackID__2", "DocID__2", "SubDocID__2")
        """)
        )
        conn.commit()
    logging.info("savings_analysis: indices created")

    # Read back only positives with enrichment
    data_enriched = (
        pl.read_database(
            query="""
                SELECT t.*, h1.MAX_REIMBURSEMENT, h2.MAX_REIMBURSEMENT AS MAX_REIMBURSEMENT__2
                FROM DA00249.TEMP_DZ_SAVINGS t
                LEFT JOIN (
                    SELECT STACKID, DOCID, SUBDOCID, MAX_REIMBURSEMENT FROM CUR.VW_MF_PKL_DA_HANDLER_DATA
                    WHERE (STACKID, DOCID, SUBDOCID) IN (
                        SELECT "StackID", "DocID", "SubDocID" FROM DA00249.TEMP_DZ_SAVINGS
                    )
                ) h1
                ON t."StackID" = h1.STACKID AND t."DocID" = h1.DOCID AND t."SubDocID" = h1.SUBDOCID
                LEFT JOIN (
                    SELECT STACKID, DOCID, SUBDOCID, MAX_REIMBURSEMENT FROM CUR.VW_MF_PKL_DA_HANDLER_DATA
                    WHERE (STACKID, DOCID, SUBDOCID) IN (
                        SELECT "StackID__2", "DocID__2", "SubDocID__2" FROM DA00249.TEMP_DZ_SAVINGS
                    )
                ) h2
                ON t."StackID__2" = h2.STACKID AND t."DocID__2" = h2.DOCID AND t."SubDocID__2" = h2.SUBDOCID
            """,
            connection=conn_sw,
        )
    ).with_columns(
        pl.min_horizontal("^MAX_REIMBURSEMENT.*$").alias("MIN_MAX_REIMBURSEMENT")
    )
    logging.info(f"savings_analysis: enriched rows={data_enriched.height}")

    # Filter to positives and split by or_mask — all in Polars
    tp_enriched = data_enriched.filter(pl.col("OR_MASK").cast(pl.Boolean))
    logging.info(
        f"savings_analysis: all_pos rows={data_enriched.height}, tp_only rows={tp_enriched.height}"
    )

    logging.info(
        f"# tp_enriched {tp_enriched.height}, # data_enriched: {data_enriched.height}"
    )

    def _agg(df: pl.DataFrame) -> dict:
        total = float(df["MIN_MAX_REIMBURSEMENT"].sum())
        avg = total / df.height

        paid = df.filter(pl.all_horizontal("^PAID_OUT.*$"))

        p_total = float(paid["MIN_MAX_REIMBURSEMENT"].sum())
        p_avg = p_total / paid.height
        return {
            "total_savings": total,
            "average_savings": avg,
            "both_paid_total": p_total,
            "both_paid_avg": p_avg,
        }

    return {
        "all_pos": _agg(data_enriched),
        "tp_only": _agg(tp_enriched),
        "enriched": data_enriched,
    }


def in_stack_out_stack_analysis(data: pl.DataFrame) -> dict[str, dict[str, Any]]:
    # --- In-Stack vs. Out-of-Stack --- #
    in_stack_total_pos, in_stack_pos_pred, in_stack_tp = (
        data.filter(pl.col("StackID").eq(pl.col("StackID__2")))
        .select(
            pl.col("labels").sum(),
            pl.col("or_mask").sum(),
            pl.all_horizontal("labels", "or_mask").sum().alias("tp"),
        )
        .row(0)
    )

    out_of_stack_total_pos, out_of_stack_pos_pred, out_of_stack_tp = (
        data.filter(pl.col("StackID").ne(pl.col("StackID__2")))
        .select(
            pl.col("labels").sum(),
            pl.col("or_mask").sum(),
            pl.all_horizontal("labels", "or_mask").sum().alias("tp"),
        )
        .row(0)
    )

    return {
        "in_stack_stats": {
            "total_pos": in_stack_total_pos,
            "pos_pred": in_stack_pos_pred,
            "tp": in_stack_tp,
        },
        "out_of_stack_stats": {
            "total_pos": out_of_stack_total_pos,
            "pos_pred": out_of_stack_pos_pred,
            "tp": out_of_stack_tp,
        },
    }


def full_data_prec_rec_per_rule(data, label_col, rules):
    """Cumulative OR precision/recall on the full dataset, one point per rule."""
    labels_full = data[label_col].cast(pl.Boolean).fill_null(False)
    total_pos_full = int(labels_full.sum())
    n_rows_full = len(labels_full)

    prec_values, rec_values, num_pos_preds = [], [], []
    and_masks = []
    cum_or = pl.Series("", [False] * n_rows_full, dtype=pl.Boolean)

    for rule in rules:
        # rule is already a Polars expression (pl.all_horizontal(...))
        and_mask = data.select(rule).to_series().cast(pl.Boolean).fill_null(False)
        and_masks.append(and_mask)
        cum_or = cum_or | and_mask

        tp = int((cum_or & labels_full).sum())
        pos = int(cum_or.sum())
        prec_values.append(tp / pos if pos > 0 else 0.0)
        rec_values.append(tp / total_pos_full if total_pos_full > 0 else 0.0)
        num_pos_preds.append(pos)

    logging.info(
        f"Full data curve: {len(prec_values)} points"
        + (
            f", last prec={prec_values[-1]:.4%} rec={rec_values[-1]:.4%}"
            if prec_values
            else " EMPTY"
        )
    )
    return prec_values, rec_values, num_pos_preds, and_masks


def _net_value(precision, recall, total_pos, review_cost, savings_per_tp):
    """NV = TP * savings - (TP+FP) * review_cost, with TP = recall * total_pos."""
    tp = recall * total_pos
    total_flagged = tp / precision if precision > 0 else 0
    return tp * savings_per_tp - total_flagged * review_cost


def _window_name_to_subtitle(window_name: str) -> str:
    """
    Converts window name to a human-readable Doc.

    1/Doc. 2 subtitle.
    """
    mapping = {
        "November_2025__October_2025": "Doc. 1=November 2025, Doc. 2=Oktober/November 2025",
        "November_2025__2_years_historic": "Doc. 1=November 2025, Doc. 2=November 2023–November 2025",
    }
    return mapping.get(window_name, window_name)


BLOOD_RED = "#b22222"


def plot_pr_value_landscape(
    precision,
    recall,
    steps,
    num_pos_preds,
    total_pos,
    review_cost,
    avg_savings_per_dz,
    best_i=None,
    window_name=None,
    precision_se=None,
    ci_z=1.96,
    has_evidence=None,
):
    """
    Precision-Recall curve on top of a € net-value heatmap.

    Returns fig.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    be_prec = review_cost / avg_savings_per_dz
    nv = [
        _net_value(p, r, total_pos, review_cost, avg_savings_per_dz)
        for p, r in zip(precision, recall)
    ]
    if best_i is None:
        best_i = int(np.argmax(nv))

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
    for v in iso_levels:
        rec = v / (total_pos * (avg_savings_per_dz - review_cost / target_prec))
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

    cs0 = ax.contour(
        recall_grid,
        precision_grid,
        NV,
        levels=[0],
        colors="black",
        linewidths=1.0,
        zorder=2,
    )
    ax.clabel(
        cs0,
        inline=True,
        fontsize=8,
        fmt={0: "Break-Even precision (0 EUR)"},
        manual=[(0.5, be_prec)],
    )

    ax.plot(recall, precision, color="#333333", linewidth=1.2, zorder=3)

    errorbar_container = None
    if precision_se is not None:
        recall_arr = np.asarray(recall, dtype=float)
        prec_arr = np.asarray(precision, dtype=float)
        se_arr = np.asarray(precision_se, dtype=float)
        lower_err = prec_arr - np.clip(prec_arr - ci_z * se_arr, 0, 1)
        upper_err = np.clip(prec_arr + ci_z * se_arr, 0, 1) - prec_arr
        errorbar_container = ax.errorbar(
            recall_arr,
            prec_arr,
            yerr=np.vstack([lower_err, upper_err]),
            fmt="none",
            ecolor=BLOOD_RED,
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            alpha=0.8,
            zorder=3,
            label=f"±{ci_z:g}·SE (95% CI)",
        )
    dot_colors = []
    for i in range(len(steps)):
        if has_evidence is not None and not has_evidence[i]:
            dot_colors.append(BLOOD_RED)
        elif i == best_i:
            dot_colors.append(ACCENT)
        elif precision[i] >= be_prec:
            dot_colors.append("#333333")
        else:
            dot_colors.append(GREY)
    ax.scatter(
        recall,
        precision,
        c=dot_colors,
        s=55,
        zorder=4,
        edgecolors="white",
        linewidths=1,
    )

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

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        "Precision-recall trade-off and expected value optimum derivation",
        loc="left",
        fontweight="bold",
        pad=25,
    )
    ax.text(
        x=0,
        y=1.02,
        s=f"Review cost={review_cost:.2f} EUR, Avg. saving per DZ={avg_savings_per_dz:.2f} EUR, {_window_name_to_subtitle(window_name)}",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="normal",
        ha="left",
        va="bottom",
    )
    ax.spines[["top", "right"]].set_visible(False)
    legend_handles = [
        mpatches.Patch(color="#4daf4a", alpha=0.6, label="Profit zone"),
        mpatches.Patch(color=ACCENT, label="Optimum"),
    ]
    if errorbar_container is not None:
        legend_handles.append(errorbar_container)
    if has_evidence is not None and not all(has_evidence):
        legend_handles.append(
            mpatches.Patch(color=BLOOD_RED, label="No historic sample evidence"),
        )
    ax.legend(
        handles=legend_handles,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        loc="lower left",
    )
    return fig, best_i


def filter_data_to_best_rules(
    base_data: pl.DataFrame,
    comp,
    label_col: str,
    best_i: int,
) -> pl.DataFrame:
    """
    Re-applies only the first best_i+1 rules as or_mask.

    Used to restrict plots/analysis to the business-optimal rule subset.
    """
    best_comp = comp[: best_i + 1]
    return base_data.with_columns(
        pl.col(label_col).cast(pl.Boolean).fill_null(False).alias("labels"),
        pl.any_horizontal(best_comp).alias("or_mask"),
    )


def plot_expected_savings(
    combined_with_nv,
    avg_savings_per_dz,
    review_cost,
    window_name,
    ci_z=1.96,
):
    """Expected net value per rule count m with ±CI, using the combined estimator."""
    m_vals = combined_with_nv["m"].to_numpy()
    nv = combined_with_nv["NV_est_m"].to_numpy()
    se = combined_with_nv["SE_NV_m"].to_numpy()
    has_ev = combined_with_nv["has_historic_evidence"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    lower = nv - ci_z * se
    upper = nv + ci_z * se

    ax.fill_between(m_vals, lower, upper, color=BLOOD_RED, alpha=0.2, linewidth=0, zorder=2)
    ax.plot(m_vals, nv, color="#333333", linewidth=1.2, zorder=3)

    dot_colors = [BLOOD_RED if not e else "#333333" for e in has_ev]
    ax.scatter(m_vals, nv, c=dot_colors, s=35, zorder=4, edgecolors="white", linewidths=1)

    best_i = int(np.argmax(nv))
    ax.scatter(
        [m_vals[best_i]], [nv[best_i]],
        color=ACCENT, s=120, zorder=5, edgecolors="white", linewidths=1.5,
    )
    ax.annotate(
        f"Optimum m={m_vals[best_i]}\nNV={nv[best_i]/1000:.1f}K €",
        xy=(m_vals[best_i], nv[best_i]),
        xytext=(10, 10), textcoords="offset points",
        fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999999", linewidth=0.8),
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1),
    )

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("Number of OR-rules (m)")
    ax.set_ylabel("Expected net value (€)")
    ax.set_title(
        "Expected net value per rule count (labeled + historic-estimated)",
        loc="left", fontweight="bold", pad=25,
    )
    ax.text(
        x=0, y=1.02,
        s=f"Review cost={review_cost:.2f} €, Avg. saving per DZ={avg_savings_per_dz:.2f} €, {_window_name_to_subtitle(window_name)}",
        transform=ax.transAxes, fontsize=10, ha="left", va="bottom",
    )
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=BLOOD_RED, alpha=0.2, label=f"±{ci_z:g}·SE (95% CI)"),
        mpatches.Patch(color=ACCENT, label="Optimum"),
    ]
    if not all(has_ev):
        legend_handles.append(
            mpatches.Patch(color=BLOOD_RED, label="No historic sample evidence"),
        )
    ax.legend(
        handles=legend_handles,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        loc="best",
    )

    return fig


def historic_estimated_plot(
    window_data,
    and_masks,
    label_col,
    rec_values,
    num_pos_preds,
    steps,
    total_pos,
    review_cost,
    avg_savings_per_dz,
    window_name,
):
    """Builds the historic-window PR plot with hits-weighted precision (labeled exact + historic estimated) plus error bars.

    Returns (pr_fig, nv_fig).
    """
    per_k = estimate_historic_precision(and_masks, window_data, label_col)
    historic_cum = cumulative_precision(per_k)
    labeled_cum = labeled_cumulative_precision(and_masks, window_data, label_col)
    combined = combined_precision(labeled_cum, historic_cum)

    p_combined = combined["P_combined_m"].to_list()[: len(rec_values)]
    se_combined = combined["SE_combined_m"].to_list()[: len(rec_values)]
    has_evidence = combined["has_historic_evidence"].to_list()[: len(rec_values)]

    pr_fig, _ = plot_pr_value_landscape(
        precision=p_combined,
        recall=rec_values,
        steps=steps,
        num_pos_preds=num_pos_preds,
        total_pos=total_pos,
        review_cost=review_cost,
        avg_savings_per_dz=avg_savings_per_dz,
        window_name=window_name,
        precision_se=se_combined,
        has_evidence=has_evidence,
    )

    combined_nv = expected_savings_curve(combined, avg_savings_per_dz, review_cost)
    nv_fig = plot_expected_savings(
        combined_nv,
        avg_savings_per_dz=avg_savings_per_dz,
        review_cost=review_cost,
        window_name=window_name,
    )
    return pr_fig, nv_fig


def evaluate_rule(
    greedy_rules: list[list[str]],
    final_rules: list[list[str]],
    filter_expression,
    label_col: str,
    data_path: Path,
    review_cost: float,
) -> dict[str, dict[str, Any]]:
    """
    Evaluates greedy and final rule sets on two time windows.
    Args:


    Returns
    -------
    {
        "November_2025__October_2025": {
            "greedy": { "prec_rec", "pr_value", "lar_heatmap",
                        "time_difference", "savings", "in_stack_stats", ... },
            "final":  { ... },
        },
        "November_2025__2_years_historic": { ... },
    }
    """
    # Compute savings_per_tp once from all positives over the full base filter
    logging.info("Starting to read data.")
    base_data = pl.read_parquet(data_path).filter(filter_expression)
    logging.info("Finished reading the data.")
    return_dict: dict[str, dict] = {}

    for window_name, window_filter in EVAL_WINDOWS:
        return_dict[window_name] = {}

        for comp_name, rules in [("greedy", greedy_rules), ("final", final_rules)]:
            comp = [
                pl.all_horizontal(
                    [pl.col(c).cast(pl.Boolean).fill_null(False) for c in rule]
                )
                for rule in rules
            ]

            data = base_data.filter(window_filter).with_columns(
                pl.col(label_col).cast(pl.Boolean).fill_null(False).alias("labels"),
                pl.any_horizontal(comp).alias("or_mask"),
            )

            savings_result = savings_analysis(data, label_col)
            avg_savings_per_dz = savings_result["all_pos"]["average_savings"]

            prec_values, rec_values, num_pos_preds, and_masks = (
                full_data_prec_rec_per_rule(data, label_col, comp)
            )

            total_pos = int(data["labels"].sum())
            steps = list(range(1, len(prec_values) + 1))

            in_stack_out_stack = in_stack_out_stack_analysis(data)

            # Plots
            pr_value_plot, best_i = plot_pr_value_landscape(
                precision=prec_values,
                recall=rec_values,
                steps=steps,
                num_pos_preds=num_pos_preds,
                total_pos=total_pos,
                review_cost=review_cost,
                avg_savings_per_dz=avg_savings_per_dz,
                window_name=window_name,
            )

            data_optimum_rule = filter_data_to_best_rules(
                base_data.filter(window_filter), comp, label_col, best_i
            )

            lar_heatmap_plot = generate_lar_heatmap(data_optimum_rule)
            time_diff_plot = generate_time_difference_plot(data_optimum_rule)
            savings_td_plot = plot_savings_by_time_difference(
                savings_result["enriched"], window_name
            )

            result = {
                "pr_value": pr_value_plot,
                "lar_heatmap": lar_heatmap_plot,
                "time_difference": time_diff_plot,
                "savings_time_difference": savings_td_plot,
                "savings_tp": savings_result["tp_only"],
                "savings_all_pos": savings_result["all_pos"],
                **in_stack_out_stack,
            }

            return_dict[window_name][comp_name] = result

            if window_name == "November_2025__2_years_historic" and comp_name == "greedy":
                pr_fig, nv_fig = historic_estimated_plot(
                    window_data=data,
                    and_masks=and_masks,
                    label_col=label_col,
                    rec_values=rec_values,
                    num_pos_preds=num_pos_preds,
                    steps=steps,
                    total_pos=total_pos,
                    review_cost=review_cost,
                    avg_savings_per_dz=avg_savings_per_dz,
                    window_name=window_name,
                )
                return_dict[window_name]["greedy_estimated"] = {
                    **result,
                    "pr_value": pr_fig,
                    "expected_savings": nv_fig,
                }

    return return_dict
