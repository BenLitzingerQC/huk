import logging
import sys
from typing import Any
 
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sqlalchemy import text
from sqlalchemy.types import VARCHAR
 
from da_hf5_utils.db2 import get_engine
 
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
        (pl.col("HUKIMPORTTIME__2").dt.year().eq(2025) & pl.col("HUKIMPORTTIME__2").dt.month().eq(11))
        | (pl.col("HUKIMPORTTIME__2").dt.year().eq(2025) & pl.col("HUKIMPORTTIME__2").dt.month().eq(10))
    )
)
 
WINDOW_NOV_2YR = (
    pl.col("HUKIMPORTTIME").dt.year().eq(2025)
    & pl.col("HUKIMPORTTIME").dt.month().eq(11)
    & pl.col("HUKIMPORTTIME__2").ge(pl.datetime(2023,11,1))
    & pl.col("HUKIMPORTTIME__2").lt(pl.datetime(2025,12,1))
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
 
    totals   = total_mat.select(sorted_cols).to_numpy()
    tps      = tp_mat.select(sorted_cols).to_numpy()
    pos      = pos_mat.select(sorted_cols).to_numpy()
    precs    = prec_mat.select(sorted_cols).to_numpy()
    recs     = rec_mat.select(sorted_cols).to_numpy()
    pos_preds = pos_pred_mat.select(sorted_cols).to_numpy()
 
    row_labels = lars_ordered
    col_labels = sorted_cols
 
    annot = np.empty_like(totals, dtype=object)
    for i in range(totals.shape[0]):
        for j in range(totals.shape[1]):
            if totals[i, j] > 0:
                prec_val = precs[i, j]
                rec_val  = recs[i, j]
                prec_str = f"{prec_val:.2f}%" if not np.isnan(prec_val) else "N/A"
                rec_str  = f"{rec_val:.2f}%"  if not np.isnan(rec_val)  else "N/A"
                annot[i, j] = (
                    f"{int(tps[i, j])}/{int(pos[i, j])}\n"
                    f"{int(pos_preds[i, j])}/{int(totals[i, j])}\n"
                    f"P: {prec_str}\nR: {rec_str}"
                )
            else:
                annot[i, j] = ""
 
    fig = plt.figure(figsize=(max(10, len(col_labels) * 0.7), max(8, len(row_labels) * 0.5)))
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
    ax.set_xlabel("LAR Dok2 (lar__2)")
    ax.set_ylabel("LAR Dok1 (lar)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
 
    return fig
 
 
def generate_time_difference_plot(data):
    """
    Stacked bar chart:
    - Grau: alle positiven Predictions (or_mask)
    - Grün: True Positives (or_mask & labels)
    """
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
        ax.text(0.5, 0.5, "No positive predictions", transform=ax.transAxes, ha="center")
        ax.set_title("Time Difference between Positive Predictions")
        return fig
 
    max_day = int(max(all_preds.max(), true_pos.max() if len(true_pos) > 0 else 0))
    bins = range(0, max_day + 2)
 
    counts_all, _ = np.histogram(all_preds, bins=bins)
    counts_tp, _  = np.histogram(true_pos,  bins=bins)
 
    ax.bar(bins[:-1], counts_all, width=1.0, color="lightgray", label="Positive Predictions")
    ax.bar(bins[:-1], counts_tp,  width=1.0, color="green",     label="True Positives")
 
    ax.set_xlabel("Time Difference in Days (Dok1 - Dok2)")
    ax.set_ylabel("Count")
    ax.set_title("Time Difference of Positive Predictions")
    ax.legend()
 
    return fig
 
 
def generate_prec_rec_plot(
    prec_values,
    rec_values,
    num_pos_preds,
    prec_values_full=None,
    rec_values_full=None,
    num_pos_preds_full=None,
):
    fig = plt.figure(figsize=(10, 6))
 
    # Training curve
    plt.plot(rec_values, prec_values, marker=".", zorder=1, label="Train (downsampled)", color="steelblue")
    bbox_props = dict(boxstyle="round,pad=0.2", fc="lightgrey", ec="none", alpha=0.8)
    for r, p, n in zip(rec_values, prec_values, num_pos_preds):
        plt.annotate(str(int(n)), (r, p), textcoords="offset points", xytext=(1, 1), fontsize=9, bbox=bbox_props)
 
    # Full data curve (optional)
    if prec_values_full is not None:
        plt.plot(rec_values_full, prec_values_full, marker=".", zorder=1, label="Full data", color="darkorange", linestyle="--")
        bbox_props_full = dict(boxstyle="round,pad=0.2", fc="moccasin", ec="none", alpha=0.8)
        for r, p, n in zip(rec_values_full, prec_values_full, num_pos_preds_full):
            plt.annotate(str(int(n)), (r, p), textcoords="offset points", xytext=(1, 1), fontsize=9, bbox=bbox_props_full)
 
    plt.xlim(-0.02, 1.05)
    plt.ylim(-0.02, 1.05)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve for possible rules")
 
    legend_elements = [
        mpatches.Patch(facecolor="lightgrey", alpha=0.8, label="# positive predictions (train)"),
    ]
    if prec_values_full is not None:
        legend_elements.append(
            mpatches.Patch(facecolor="moccasin", alpha=0.8, label="# positive predictions (full)")
        )
 
    plt.legend(handles=legend_elements + [
        plt.Line2D([0], [0], color="steelblue",  marker=".", label="Train"),
        plt.Line2D([0], [0], color="darkorange", marker=".", linestyle="--", label="Full data"),
    ], loc="lower left")
    plt.grid(True, linestyle="--", color="lightgrey", alpha=0.5)
 
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
    conn_sw = get_engine(database="spielwiese")
    raw_ids = ["StackID", "DocID", "SubDocID", "StackID__2", "DocID__2", "SubDocID__2"]
 
    # Write ALL positives (label=True) — superset of TPs — to temp table.
    # We keep or_mask as a column so we can filter to TPs inside Python.
    with conn_sw.begin() as conn:
        try:
            conn.execute(text("DROP TABLE DA00249.TEMP_DZ_SAVINGS"))
        except Exception:
            pass
 
    data.filter(
        pl.col(label_col)
    ).drop(
        ["LAR", "LAR__2"]
    ).write_database(
        table_name="DA00249.TEMP_DZ_SAVINGS",
        connection=conn_sw,
        engine_options={"dtype": {c: VARCHAR(length=50) for c in raw_ids}},
    )
 
    with conn_sw.connect() as conn:
        conn.execute(text("""
            CREATE INDEX idx_temp_ids
            ON DA00249.TEMP_DZ_SAVINGS ("StackID", "DocID", "SubDocID")
        """))
        conn.execute(text("""
            CREATE INDEX idx_temp_ids_2
            ON DA00249.TEMP_DZ_SAVINGS ("StackID__2", "DocID__2", "SubDocID__2")
        """))
        conn.commit()
 
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
 
    def _agg(df: pl.DataFrame) -> dict:
        total   = float(df["MIN_MAX_REIMBURSEMENT"].sum())
        avg     = total / df.height if df.height > 0 else 0.0
        paid    = df.filter(pl.all_horizontal("^PAID_OUT.*$").cast(pl.Boolean))
        p_total = float(paid["MIN_MAX_REIMBURSEMENT"].sum())
        p_avg   = p_total / paid.height if paid.height > 0 else 0.0
        return {
            "total_savings":    total,
            "average_savings":  avg,
            "both_paid_total":  p_total,
            "both_paid_avg":    p_avg,
        }
 
    # or_mask column is present because data passed in still has it
    tp_mask = data_enriched["or_mask"].cast(pl.Boolean)
 
    return {
        "all_pos": _agg(data_enriched),
        "tp_only": _agg(data_enriched.filter(tp_mask)),
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
        }
    }
 
def full_data_prec_rec_per_rule(data, label_col, rules):
    """Cumulative OR precision/recall on the full dataset, one point per rule."""
    labels_full    = data[label_col].cast(pl.Boolean).fill_null(False)
    total_pos_full = int(labels_full.sum())
    n_rows_full    = len(labels_full)
 
    prec_values, rec_values, num_pos_preds = [], [], []
    cum_or = pl.Series("", [False] * n_rows_full, dtype=pl.Boolean)
 
    for rule in rules:
        # rule is already a Polars expression (pl.all_horizontal(...))
        and_mask = data.select(rule).to_series().cast(pl.Boolean).fill_null(False)
        cum_or   = cum_or | and_mask
 
        tp  = int((cum_or & labels_full).sum())
        pos = int(cum_or.sum())
        prec_values.append(tp / pos            if pos            > 0 else 0.0)
        rec_values.append(tp  / total_pos_full if total_pos_full > 0 else 0.0)
        num_pos_preds.append(pos)
 
    logging.info(
        f"Full data curve: {len(prec_values)} points"
        + (f", last prec={prec_values[-1]:.4%} rec={rec_values[-1]:.4%}"
           if prec_values else " EMPTY")
    )
    return prec_values, rec_values, num_pos_preds
 
def _net_value(precision, recall, total_pos, review_cost, savings_per_tp):
    """NV = TP * savings - (TP+FP) * review_cost, with TP = recall * total_pos."""
    tp             = recall * total_pos
    total_flagged  = tp / precision if precision > 0 else 0
    return tp * savings_per_tp - total_flagged * review_cost
 
def plot_pr_value_landscape(
    precision,
    recall,
    steps,
    total_pos,
    review_cost,
    savings_per_tp,
    best_i=None,
):
    """Precision-Recall curve on top of a € net-value heatmap. Returns fig."""
    fig, ax = plt.subplots(figsize=(10, 6))
 
    be_prec = review_cost / savings_per_tp
    nv = [_net_value(p, r, total_pos, review_cost, savings_per_tp)
          for p, r in zip(precision, recall)]
    if best_i is None:
        best_i = int(np.argmax(nv))
 
    rr = np.linspace(0.01, 1.0, 200)
    pp = np.linspace(0.01, 1.0, 200)
    RR, PP = np.meshgrid(rr, pp)
    NV       = RR * total_pos * (savings_per_tp - review_cost / PP)
    max_gain = total_pos * savings_per_tp
 
    ax.contourf(RR, PP, np.where(NV > 0, NV, np.nan),
                levels=40, cmap="Greens", vmin=0, vmax=max_gain,
                alpha=0.6, zorder=0)
 
    for s in (5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000):
        if max_gain / s <= 8:
            step_size = s
            break
    else:
        step_size = 1_000_000
 
    iso_levels = np.arange(step_size, max_gain + step_size, step_size)
    ax.contour(RR, PP, NV, levels=iso_levels, colors="#444444",
               linewidths=0.6, alpha=0.6, zorder=1)
 
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
                                  facecolor="white", edgecolor="none", alpha=0.75),
                        zorder=2)
 
    cs0 = ax.contour(RR, PP, NV, levels=[0], colors="black",
                     linewidths=1.0, zorder=2)
    ax.clabel(cs0, inline=True, fontsize=8, fmt={0: "Break-even (0 €)"})
 
    ax.plot(recall, precision, color="#333333", linewidth=1.2, zorder=3)
    dot_colors = [
        ACCENT if i == best_i else ("#333333" if precision[i] >= be_prec else GREY)
        for i in range(len(steps))
    ]
    ax.scatter(recall, precision, c=dot_colors, s=55, zorder=4,
               edgecolors="white", linewidths=1)
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
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        "Precision-Recall mit Netto-Wert-Landschaft\n"
        "(Iso-Linien = gleicher € Netto-Wert)",
        loc="left", fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            mpatches.Patch(color="#4daf4a", alpha=0.6, label="Gewinn-Zone"),
            mpatches.Patch(color=ACCENT,    label="Business-Optimum"),
        ],
        frameon=True, facecolor="white", edgecolor="#cccccc", loc="lower left",
    )
    return fig
 
def evaluate_rule(
    greedy_rules,
    final_rules,
    filter_expression,
    label_col,
    data_path,
    review_cost,
):
    """
    Evaluates greedy and final rule sets on two time windows.
 
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
    # --- Compute savings_per_tp once from all positives over the full base filter ---
    # We read only the label column + IDs for the global savings estimate.
    base_data = (
        pl.read_parquet(data_path)
        .filter(filter_expression)
    )
 
    return_dict: dict[str, dict] = {}
 
    for window_name, window_filter in EVAL_WINDOWS:
        return_dict[window_name] = {}
 
        for comp_name, rules in [("greedy", greedy_rules), ("final", final_rules)]:
            comp = [
                pl.all_horizontal([pl.col(c).cast(pl.Boolean).fill_null(False) for c in rule])
                for rule in rules
            ]
 
            data = (
                base_data
                .filter(window_filter)
                .with_columns(
                    pl.col(label_col).cast(pl.Boolean).fill_null(False).alias("labels"),
                    pl.any_horizontal(comp).alias("or_mask"),
                )
            )
 
            # --- Savings (Option A: one DB round-trip, returns all_pos + tp_only) ---
            savings_result = savings_analysis(data, label_col)
 
            # savings_per_tp: data-driven estimate from ALL positives (unrestricted avg)
            savings_per_tp = savings_result["all_pos"]["average_savings"]
 
            # --- Prec/rec curve on the windowed full data ---
            prec_values, rec_values, num_pos_preds = full_data_prec_rec_per_rule(
                data, label_col, comp
            )
 
            total_pos = int(data["labels"].sum())
            steps     = list(range(1, len(prec_values) + 1))
 
            in_stack_out_stack = in_stack_out_stack_analysis(data)
 
            # --- Plots ---
            prec_rec_plot = generate_prec_rec_plot(prec_values, rec_values, num_pos_preds)
            pr_value_plot = plot_pr_value_landscape(
                precision=prec_values,
                recall=rec_values,
                steps=steps,
                total_pos=total_pos,
                review_cost=review_cost,
                savings_per_tp=savings_per_tp,
            )
            lar_heatmap_plot = generate_lar_heatmap(data)
            time_diff_plot   = generate_time_difference_plot(data)
 
            return_dict[window_name][comp_name] = {
                "prec_rec":       prec_rec_plot,
                "pr_value":       pr_value_plot,
                "lar_heatmap":    lar_heatmap_plot,
                "time_difference": time_diff_plot,
                # savings sub-dicts kept flat for easy MLflow access
                "savings":             savings_result["tp_only"],
                "savings_all_pos":     savings_result["all_pos"],
                **in_stack_out_stack,
            }
 
    return return_dict
