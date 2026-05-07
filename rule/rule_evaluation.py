"""
Evaluation logic for a Composition on the Nov-2025 / 2-years-historic window.

Pipeline (all in-process, exactly one DB call):
  1. Read parquet, apply base + window filter.
  2. Union of all rule masks → unique docs touched by ANY pos-pred.
  3. ONE DB query enriches those docs with MAX_REIMBURSEMENT.
  4. Pair-wise min-reimbursement filter (in Polars).
  5. Per-step (m = 1..len(rules)) NV via union-find on TP-pairs.
  6. Best m = argmax(NV_total). Post-process composition[:m+1] with
     remove_subsumed + remove_redundant → optimum_rules.
  7. Build plots and metrics, return EvaluationResult.

Net-value rule:
  Per connected component of TP docs (sorted by reimbursement asc),
  reject the n-1 cheapest. NV_total uses every rejected reimbursement;
  NV_paid_out uses only those with PAID_OUT=True.
  Cost = pos_pred_pair_count * review_cost (review work is per pair).
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from da_hf5_dz.configs.plotting.plotting import setup_plotting
from da_hf5_utils.db2 import get_engine
from sqlalchemy import text
from sqlalchemy.types import VARCHAR

from rule_core import Rule, remove_redundant, remove_subsumed

ACCENT_COLOR = "#FFCC00"
GREEN_COLOR = "#2a9d8f"
GREY_COLOR = "#cccccc"

DOC_ID_COLUMNS = ("StackID", "DocID", "SubDocID")
DOC2_ID_COLUMNS = ("StackID__2", "DocID__2", "SubDocID__2")

WINDOW_NOV_2YR = (
    pl.col("HUKIMPORTTIME").dt.year().eq(2025)
    & pl.col("HUKIMPORTTIME").dt.month().eq(11)
    & pl.col("HUKIMPORTTIME__2").ge(pl.datetime(2023, 11, 1))
    & pl.col("HUKIMPORTTIME__2").lt(pl.datetime(2025, 12, 1))
)

REIMBURSEMENT_TEMP_TABLE = "DA00249.TEMP_DZ_REIMBURSEMENT"


@dataclass
class EvaluationResult:
    """Outcome of evaluating a Composition on the historic window."""

    plots: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    optimum_rules: list[Rule] = field(default_factory=list)
    optimum_step: int = 0


# --- Union-find ------------------------------------------------------------------


class _UnionFind:
    """Disjoint-set with path compression and union-by-size."""

    def __init__(self):
        self.parent: dict[Any, Any] = {}
        self.size: dict[Any, int] = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def compute_component_savings(
    pairs_df: pl.DataFrame,
    doc_reimbursement: dict[tuple, float],
    doc_paid_out: dict[tuple, bool],
) -> tuple[float, float]:
    """
    Build connected components from `pairs_df`, then per component reject the
    n-1 cheapest docs and sum their reimbursements.

    `pairs_df` must contain columns DOC_ID_COLUMNS and DOC2_ID_COLUMNS — each
    row is one TP pair (= one edge between two doc nodes).
    `doc_reimbursement` / `doc_paid_out` are keyed by the (StackID, DocID, SubDocID)
    tuple. Docs missing from the lookup are skipped (treated as 0 / False).

    Returns (total_savings, paid_out_savings). Both are summed across components,
    with paid_out_savings counting only docs whose PAID_OUT flag is True.
    """
    if pairs_df.is_empty():
        return 0.0, 0.0

    union_find = _UnionFind()
    for row in pairs_df.iter_rows(named=False):
        doc_a = row[: len(DOC_ID_COLUMNS)]
        doc_b = row[len(DOC_ID_COLUMNS) :]
        union_find.union(doc_a, doc_b)

    components: dict[Any, list[tuple]] = {}
    for doc in union_find.parent:
        components.setdefault(union_find.find(doc), []).append(doc)

    total_savings = 0.0
    paid_out_savings = 0.0
    for docs in components.values():
        if len(docs) < 2:
            continue
        annotated = [
            (doc, doc_reimbursement.get(doc, 0.0), doc_paid_out.get(doc, False))
            for doc in docs
        ]
        annotated.sort(key=lambda triple: triple[1])
        rejected = annotated[:-1]  # keep the most expensive, reject the rest
        for _, reimbursement, paid_out in rejected:
            total_savings += reimbursement
            if paid_out:
                paid_out_savings += reimbursement

    return total_savings, paid_out_savings


# --- DB enrichment ---------------------------------------------------------------


def fetch_reimbursements_for_pos_pred_docs(pos_pred_data: pl.DataFrame) -> pl.DataFrame:
    """
    Single-DB-call enrichment: for every doc that appears (on either side) in
    `pos_pred_data`, fetch MAX_REIMBURSEMENT from the pkl handler view.

    Returns a DataFrame with columns StackID, DocID, SubDocID, MAX_REIMBURSEMENT.
    Docs that don't match in the DB get a null reimbursement.
    """
    docs_left = pos_pred_data.select(list(DOC_ID_COLUMNS))
    docs_right = pos_pred_data.select(list(DOC2_ID_COLUMNS)).rename(
        dict(zip(DOC2_ID_COLUMNS, DOC_ID_COLUMNS))
    )
    unique_docs = pl.concat([docs_left, docs_right]).unique()

    logging.info(
        f"fetch_reimbursements_for_pos_pred_docs: {pos_pred_data.height} pos-pred pairs "
        f"→ {unique_docs.height} unique docs"
    )

    engine = get_engine(database="spielwiese")
    with engine.begin() as connection:
        try:
            connection.execute(text(f"DROP TABLE {REIMBURSEMENT_TEMP_TABLE}"))
        except Exception as exc:
            logging.info(f"no temp table to drop (expected): {exc}")

    unique_docs.write_database(
        table_name=REIMBURSEMENT_TEMP_TABLE,
        connection=engine,
        engine_options={
            "dtype": {column: VARCHAR(length=50) for column in DOC_ID_COLUMNS}
        },
    )

    with engine.connect() as connection:
        connection.execute(
            text(
                f"CREATE INDEX idx_dz_reimb_ids "
                f'ON {REIMBURSEMENT_TEMP_TABLE} ("StackID", "DocID", "SubDocID")'
            )
        )
        connection.commit()

    enriched = pl.read_database(
        query=f"""
            SELECT t."StackID", t."DocID", t."SubDocID",
                   h.MAX_REIMBURSEMENT
            FROM {REIMBURSEMENT_TEMP_TABLE} t
            LEFT JOIN (
                SELECT STACKID, DOCID, SUBDOCID, MAX_REIMBURSEMENT
                FROM CUR.VW_MF_PKL_DA_HANDLER_DATA
                WHERE (STACKID, DOCID, SUBDOCID) IN (
                    SELECT "StackID", "DocID", "SubDocID" FROM {REIMBURSEMENT_TEMP_TABLE}
                )
            ) h
            ON t."StackID"=h.STACKID AND t."DocID"=h.DOCID AND t."SubDocID"=h.SUBDOCID
        """,
        connection=engine,
    )

    db_to_original_case = {column.upper(): column for column in DOC_ID_COLUMNS}
    enriched = enriched.rename(
        {
            column: db_to_original_case[column]
            for column in enriched.columns
            if column in db_to_original_case
        }
    )

    n_no_match = int(enriched["MAX_REIMBURSEMENT"].is_null().sum())
    if n_no_match > 0:
        logging.warning(
            f"fetch_reimbursements_for_pos_pred_docs: {n_no_match} of "
            f"{enriched.height} docs had NO MAX_REIMBURSEMENT in DB"
        )
    return enriched


def join_reimbursements_to_pairs(
    pairs: pl.DataFrame, doc_reimbursements: pl.DataFrame
) -> pl.DataFrame:
    """
    Adds MAX_REIMBURSEMENT and MAX_REIMBURSEMENT__2 columns to `pairs` by joining
    `doc_reimbursements` (one row per doc) on each side.
    """
    left_joined = pairs.join(doc_reimbursements, on=list(DOC_ID_COLUMNS), how="left")
    right_view = doc_reimbursements.rename(
        dict(zip(DOC_ID_COLUMNS, DOC2_ID_COLUMNS))
        | {"MAX_REIMBURSEMENT": "MAX_REIMBURSEMENT__2"}
    )
    return left_joined.join(right_view, on=list(DOC2_ID_COLUMNS), how="left")


def filter_pairs_by_min_reimbursement(
    pairs: pl.DataFrame, threshold: float
) -> pl.DataFrame:
    """
    Pair-wise filter: drop pairs where min(MAX_REIMBURSEMENT, MAX_REIMBURSEMENT__2)
    is at most `threshold`. Pairs with a null on either side are kept (no DB match
    is not a reason to discard).
    """
    if threshold is None or threshold <= 0:
        return pairs
    pair_min = pl.min_horizontal("MAX_REIMBURSEMENT", "MAX_REIMBURSEMENT__2")
    kept = pairs.filter(pair_min.is_null() | pair_min.gt(threshold))
    logging.info(
        f"filter_pairs_by_min_reimbursement: {pairs.height} → {kept.height} pairs "
        f"(threshold={threshold})"
    )
    return kept


# --- Per-step net value ----------------------------------------------------------


@dataclass
class StepwiseNetValue:
    """Per-step (m = 1..n) statistics for a Composition under union-find savings."""

    pos_pred_pair_count: list[int] = field(default_factory=list)
    true_positive_pair_count: list[int] = field(default_factory=list)
    total_savings: list[float] = field(default_factory=list)
    paid_out_savings: list[float] = field(default_factory=list)
    net_value_total: list[float] = field(default_factory=list)
    net_value_paid_out: list[float] = field(default_factory=list)


def compute_stepwise_net_value(
    pairs: pl.DataFrame,
    rule_masks: list[pl.Series],
    labels: pl.Series,
    review_cost: float,
) -> StepwiseNetValue:
    """
    For each m = 1..len(rule_masks), evaluate the composition of the first m rules:

      1. or_mask_m = OR of rule_masks[:m+1]
      2. true_positive_pairs = pairs where (or_mask_m & labels)
      3. union-find on those pairs → total_savings, paid_out_savings
      4. NV_total = total_savings - pos_pred_count_m * review_cost
         NV_paid  = paid_out_savings - pos_pred_count_m * review_cost

    `pairs` must have DOC_ID_COLUMNS, DOC2_ID_COLUMNS, MAX_REIMBURSEMENT,
    MAX_REIMBURSEMENT__2, PAID_OUT, PAID_OUT__2 — and be aligned row-wise with
    `rule_masks` and `labels`.
    """
    doc_reimbursement, doc_paid_out = _build_doc_lookups(pairs)

    n_rows = pairs.height
    cumulative_or = pl.Series("", [False] * n_rows, dtype=pl.Boolean)
    result = StepwiseNetValue()

    for rule_mask in rule_masks:
        cumulative_or = cumulative_or | rule_mask
        true_positive_mask = cumulative_or & labels

        true_positive_pairs = pairs.filter(true_positive_mask).select(
            list(DOC_ID_COLUMNS) + list(DOC2_ID_COLUMNS)
        )

        total_savings, paid_out_savings = compute_component_savings(
            true_positive_pairs, doc_reimbursement, doc_paid_out
        )
        pos_pred_count = int(cumulative_or.sum())
        true_positive_count = int(true_positive_mask.sum())

        review_cost_total = pos_pred_count * review_cost
        result.pos_pred_pair_count.append(pos_pred_count)
        result.true_positive_pair_count.append(true_positive_count)
        result.total_savings.append(total_savings)
        result.paid_out_savings.append(paid_out_savings)
        result.net_value_total.append(total_savings - review_cost_total)
        result.net_value_paid_out.append(paid_out_savings - review_cost_total)

    return result


def _build_doc_lookups(
    pairs: pl.DataFrame,
) -> tuple[dict[tuple, float], dict[tuple, bool]]:
    """Extract per-doc reimbursement and paid_out from a pairs DataFrame."""
    # Explicit casts prevent SchemaError when a column is all-null on one side
    # (polars would otherwise infer Null dtype and fail to concat with Float64).
    left = pairs.select(
        list(DOC_ID_COLUMNS)
        + [
            pl.col("MAX_REIMBURSEMENT").cast(pl.Float64).alias("reimbursement"),
            pl.col("PAID_OUT").cast(pl.Boolean).alias("paid_out"),
        ]
    )
    right = pairs.select(
        [pl.col(c).alias(orig) for c, orig in zip(DOC2_ID_COLUMNS, DOC_ID_COLUMNS)]
        + [
            pl.col("MAX_REIMBURSEMENT__2").cast(pl.Float64).alias("reimbursement"),
            pl.col("PAID_OUT__2").cast(pl.Boolean).alias("paid_out"),
        ]
    )
    docs = pl.concat([left, right]).unique(subset=list(DOC_ID_COLUMNS))

    reimbursements: dict[tuple, float] = {}
    paid_out: dict[tuple, bool] = {}
    for row in docs.iter_rows(named=False):
        doc_id = row[: len(DOC_ID_COLUMNS)]
        reimbursement_value = row[len(DOC_ID_COLUMNS)]
        paid_out_value = row[len(DOC_ID_COLUMNS) + 1]
        reimbursements[doc_id] = (
            float(reimbursement_value) if reimbursement_value is not None else 0.0
        )
        paid_out[doc_id] = bool(paid_out_value) if paid_out_value is not None else False
    return reimbursements, paid_out


# --- Optimum + post-processing ---------------------------------------------------


def find_optimum_and_post_process(
    rules: list[Rule],
    net_value_total: list[float],
    groups,
    labels: pl.Series,
) -> tuple[list[Rule], int]:
    """
    Pick best m by argmax(net_value_total), trim composition to first m+1 rules,
    then run remove_subsumed and remove_redundant on that subset.

    Returns (optimum_rules, optimum_step). optimum_step is m (zero-indexed: the
    optimum_rules originate from rules[:optimum_step + 1] before reduction).
    """
    if not net_value_total:
        return [], 0
    optimum_step = max(range(len(net_value_total)), key=net_value_total.__getitem__)
    truncated = rules[: optimum_step + 1]
    after_subsumed = remove_subsumed(truncated, groups)
    after_redundant = remove_redundant(after_subsumed, labels)
    return after_redundant, optimum_step


# --- In-stack / out-of-stack analysis --------------------------------------------


def compute_in_stack_out_stack_metrics(
    pairs: pl.DataFrame, or_mask: pl.Series, labels: pl.Series
) -> dict[str, float]:
    """
    Pair-wise counts split by whether StackID == StackID__2.
    Returns flat dict suitable for MLflow (no nested structure).
    """
    annotated = pairs.with_columns(
        or_mask.alias("_or_mask"),
        labels.alias("_labels"),
    )

    def _counts(filtered: pl.DataFrame) -> tuple[int, int, int]:
        row = filtered.select(
            pl.col("_labels").sum().alias("total_pos"),
            pl.col("_or_mask").sum().alias("pos_pred"),
            pl.all_horizontal("_labels", "_or_mask").sum().alias("tp"),
        ).row(0)
        return int(row[0]), int(row[1]), int(row[2])

    in_total_pos, in_pos_pred, in_tp = _counts(
        annotated.filter(pl.col("StackID").eq(pl.col("StackID__2")))
    )
    out_total_pos, out_pos_pred, out_tp = _counts(
        annotated.filter(pl.col("StackID").ne(pl.col("StackID__2")))
    )
    return {
        "in_stack_total_pos": float(in_total_pos),
        "in_stack_pos_pred": float(in_pos_pred),
        "in_stack_tp": float(in_tp),
        "out_of_stack_total_pos": float(out_total_pos),
        "out_of_stack_pos_pred": float(out_pos_pred),
        "out_of_stack_tp": float(out_tp),
    }


# --- Plots -----------------------------------------------------------------------


def plot_precision_recall_curve(
    precision: list[float],
    recall: list[float],
    pos_pred_pair_count: list[int],
    optimum_step: int,
):
    """Plain PR-curve over per-step values, with the optimum highlighted."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recall, precision, color="#333333", linewidth=1.2, zorder=2)

    dot_colors = [
        ACCENT_COLOR if i == optimum_step else GREY_COLOR for i in range(len(precision))
    ]
    ax.scatter(
        recall,
        precision,
        c=dot_colors,
        s=55,
        zorder=3,
        edgecolors="white",
        linewidths=1,
    )

    annotation_box = dict(
        boxstyle="round,pad=0.2", fc="lightgrey", ec="none", alpha=0.8
    )
    for r, p, n in zip(recall, precision, pos_pred_pair_count):
        ax.annotate(
            str(n),
            (r, p),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            bbox=annotation_box,
            zorder=4,
        )

    if precision:
        ax.scatter(
            [recall[optimum_step]],
            [precision[optimum_step]],
            color=ACCENT_COLOR,
            s=120,
            zorder=5,
            edgecolors="white",
            linewidths=1.5,
        )

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        "Precision-Recall trade-off (Doc.1=Nov 2025, Doc.2=Nov 2023–Nov 2025)",
        loc="left",
        fontweight="bold",
        pad=15,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[mpatches.Patch(color=ACCENT_COLOR, label="Optimum")],
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        loc="lower left",
    )
    return fig


def plot_lar_heatmap(pairs: pl.DataFrame, or_mask: pl.Series, labels: pl.Series):
    """LAR x LAR__2 heatmap of pair counts and precision/recall annotations."""
    annotated = pairs.with_columns(
        or_mask.alias("_or_mask"),
        labels.alias("_labels"),
    )
    aggregated = (
        annotated.group_by(["lar", "lar__2"])
        .agg(
            pl.len().alias("n_total"),
            pl.col("_labels").sum().alias("total_pos"),
            pl.col("_or_mask").sum().alias("pos_pred"),
            pl.all_horizontal("_labels", "_or_mask").sum().alias("tp"),
        )
        .with_columns(
            (pl.col("tp") / pl.col("pos_pred") * 100)
            .fill_nan(0)
            .alias("precision_pct"),
            (pl.col("tp") / pl.col("total_pos") * 100).fill_nan(0).alias("recall_pct"),
        )
    )

    def _pivot(value_column: str, fill) -> pl.DataFrame:
        return (
            aggregated.pivot(
                index="lar",
                on="lar__2",
                values=value_column,
                aggregate_function="first",
            )
            .fill_null(fill)
            .sort("lar")
        )

    total_matrix = _pivot("n_total", 0)
    tp_matrix = _pivot("tp", 0.0)
    pos_matrix = _pivot("total_pos", 0.0)
    pos_pred_matrix = _pivot("pos_pred", 0.0)
    precision_matrix = _pivot("precision_pct", float("nan"))
    recall_matrix = _pivot("recall_pct", float("nan"))

    lar_order = total_matrix["lar"].to_list()
    column_order = [c for c in lar_order if c in total_matrix.columns]
    totals = total_matrix.select(column_order).to_numpy()
    tps = tp_matrix.select(column_order).to_numpy()
    positives = pos_matrix.select(column_order).to_numpy()
    pos_preds = pos_pred_matrix.select(column_order).to_numpy()
    precisions = precision_matrix.select(column_order).to_numpy()
    recalls = recall_matrix.select(column_order).to_numpy()

    annotations = np.empty_like(totals, dtype=object)
    for i in range(totals.shape[0]):
        for j in range(totals.shape[1]):
            if totals[i, j] > 0:
                precision_str = (
                    f"{precisions[i, j]:.2f}%"
                    if not np.isnan(precisions[i, j])
                    else "N/A"
                )
                recall_str = (
                    f"{recalls[i, j]:.2f}%" if not np.isnan(recalls[i, j]) else "N/A"
                )
                annotations[i, j] = (
                    f"{int(tps[i, j])}/{int(positives[i, j])}\n"
                    f"{int(pos_preds[i, j])}/{int(totals[i, j])}\n"
                    f"P: {precision_str}\nR: {recall_str}"
                )
            else:
                annotations[i, j] = ""

    fig = plt.figure(
        figsize=(max(10, len(column_order) * 0.7), max(8, len(lar_order) * 0.5))
    )
    ax = sns.heatmap(
        positives,
        annot=annotations,
        annot_kws={"fontsize": 8},
        fmt="",
        cmap="huk_heatmap",
        xticklabels=column_order,
        yticklabels=lar_order,
        cbar_kws={"label": "Number of DZs per LAR combination"},
        linewidths=3.5,
    )
    ax.set_xlabel("LAR Doc. 2")
    ax.set_ylabel("LAR Doc. 1")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    return fig


def plot_expected_savings(
    net_value_per_step: list[float],
    optimum_step: int,
    review_cost: float,
    title: str,
    min_reimbursement_threshold: float = 0.0,
):
    """Net value over m (number of rules in the composition)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if not net_value_per_step:
        ax.text(0.5, 0.5, "No rules", transform=ax.transAxes, ha="center")
        ax.set_title(title, loc="left", fontweight="bold", pad=15)
        return fig

    m_values = np.arange(1, len(net_value_per_step) + 1)
    net_values = np.asarray(net_value_per_step, dtype=float)

    ax.plot(m_values, net_values, color="#333333", linewidth=1.2, zorder=2)
    ax.scatter(
        m_values,
        net_values,
        c=[
            ACCENT_COLOR if i == optimum_step else GREY_COLOR
            for i in range(len(net_values))
        ],
        s=45,
        zorder=3,
        edgecolors="white",
        linewidths=1,
    )
    ax.scatter(
        [m_values[optimum_step]],
        [net_values[optimum_step]],
        color=ACCENT_COLOR,
        s=120,
        zorder=4,
        edgecolors="white",
        linewidths=1.5,
    )
    ax.annotate(
        f"Optimum m={m_values[optimum_step]}\nNV={net_values[optimum_step] / 1000:.1f}K €",
        xy=(m_values[optimum_step], net_values[optimum_step]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="#999999",
            linewidth=0.8,
        ),
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1),
    )

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=1)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:,.0f} €".replace(",", "."))
    )
    ax.set_xlabel("Number of OR-rules (m)")
    ax.set_ylabel("Net value (€)")
    ax.set_title(title, loc="left", fontweight="bold", pad=25)
    ax.text(
        x=0,
        y=1.02,
        s=(
            f"Review cost={review_cost:.2f} EUR, "
            f"min-reimbursement threshold={min_reimbursement_threshold:.1f}"
        ),
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="bottom",
    )
    ax.spines[["top", "right"]].set_visible(False)
    return fig


# --- Top-level entry point -------------------------------------------------------


def evaluate_composition(
    composition_rules: list[Rule],
    groups,
    data_path,
    base_filter,
    label_col: str,
    review_cost: float,
    min_reimbursement_threshold: float = 0.0,
) -> "EvaluationResult":
    """
    Evaluate a Composition on the November-2025 / 2-years-historic window.

    Reads the parquet once, applies `base_filter` and the fixed window filter,
    enriches positive-pred docs with MAX_REIMBURSEMENT in a single DB call,
    drops pairs below the pair-wise reimbursement threshold, computes per-step
    net values via union-find, picks the argmax-NV step, post-processes that
    sub-sequence with remove_subsumed + remove_redundant, and produces plots
    plus a flat metrics dict.

    The masks attached to `composition_rules` are NOT reused — they were built
    against the training data; this function re-evaluates predicates against
    the historic window.
    """
    setup_plotting()
    logging.info("evaluate_composition: reading parquet")
    pairs = pl.read_parquet(data_path).filter(base_filter & WINDOW_NOV_2YR)
    labels = pairs[label_col].cast(pl.Boolean).fill_null(False)

    rule_masks = [_evaluate_rule_predicates(pairs, rule) for rule in composition_rules]
    or_mask_full = _or_of_series(rule_masks, len(pairs))

    pos_pred_pairs = pairs.filter(or_mask_full)
    doc_reimbursements = fetch_reimbursements_for_pos_pred_docs(pos_pred_pairs)
    pairs = join_reimbursements_to_pairs(pairs, doc_reimbursements)
    pairs = filter_pairs_by_min_reimbursement(pairs, min_reimbursement_threshold)

    labels = pairs[label_col].cast(pl.Boolean).fill_null(False)
    rule_masks = [_evaluate_rule_predicates(pairs, rule) for rule in composition_rules]

    stepwise = compute_stepwise_net_value(pairs, rule_masks, labels, review_cost)
    optimum_rules, optimum_step = find_optimum_and_post_process(
        composition_rules, stepwise.net_value_total, groups, labels
    )

    or_mask_optimum = _or_of_series(rule_masks[: optimum_step + 1], len(pairs))

    plots = _build_plots(
        pairs=pairs,
        or_mask_optimum=or_mask_optimum,
        labels=labels,
        stepwise=stepwise,
        optimum_step=optimum_step,
        review_cost=review_cost,
        min_reimbursement_threshold=min_reimbursement_threshold,
    )
    metrics = _build_metrics(
        pairs=pairs,
        or_mask_optimum=or_mask_optimum,
        labels=labels,
        stepwise=stepwise,
        optimum_step=optimum_step,
    )

    return EvaluationResult(
        plots=plots,
        metrics=metrics,
        optimum_rules=optimum_rules,
        optimum_step=optimum_step,
    )


def evaluate_predicate_lists(
    predicate_lists: list[list[str]],
    groups,
    data_path,
    base_filter,
    label_col: str,
    review_cost: float,
    min_reimbursement_threshold: float = 0.0,
) -> "EvaluationResult":
    """
    Convenience entry point: evaluate a manually defined composition.

    Wraps each list of predicate names in a Rule (with empty mask, since
    `evaluate_composition` re-evaluates predicates against the historic window
    anyway) and forwards to `evaluate_composition`.
    """
    empty_mask = pl.Series("", [], dtype=pl.Boolean)
    rules = [Rule(predicates=p, mask=empty_mask) for p in predicate_lists]
    return evaluate_composition(
        composition_rules=rules,
        groups=groups,
        data_path=data_path,
        base_filter=base_filter,
        label_col=label_col,
        review_cost=review_cost,
        min_reimbursement_threshold=min_reimbursement_threshold,
    )


def _evaluate_rule_predicates(pairs: pl.DataFrame, rule: Rule) -> pl.Series:
    """Re-evaluate the predicates of a Rule against `pairs` (ignores stored mask)."""
    expression = pl.all_horizontal(
        [
            pl.col(predicate).cast(pl.Boolean).fill_null(False)
            for predicate in rule.predicates
        ]
    )
    return pairs.select(expression).to_series().cast(pl.Boolean).fill_null(False)


def _or_of_series(masks: list[pl.Series], n_rows: int) -> pl.Series:
    cumulative = pl.Series("", [False] * n_rows, dtype=pl.Boolean)
    for mask in masks:
        cumulative = cumulative | mask
    return cumulative


def _build_plots(
    pairs: pl.DataFrame,
    or_mask_optimum: pl.Series,
    labels: pl.Series,
    stepwise: StepwiseNetValue,
    optimum_step: int,
    review_cost: float,
    min_reimbursement_threshold: float,
) -> dict[str, Any]:
    precision_per_step = [
        (tp / pos_pred) if pos_pred > 0 else 0.0
        for tp, pos_pred in zip(
            stepwise.true_positive_pair_count, stepwise.pos_pred_pair_count
        )
    ]
    total_positives = int(labels.sum())
    recall_per_step = [
        (tp / total_positives) if total_positives > 0 else 0.0
        for tp in stepwise.true_positive_pair_count
    ]

    return {
        "precision_recall_curve": plot_precision_recall_curve(
            precision_per_step,
            recall_per_step,
            stepwise.pos_pred_pair_count,
            optimum_step,
        ),
        "expected_savings_total": plot_expected_savings(
            stepwise.net_value_total,
            optimum_step,
            review_cost,
            "Expected net value (all rejected reimbursements)",
            min_reimbursement_threshold,
        ),
        "expected_savings_paid_out": plot_expected_savings(
            stepwise.net_value_paid_out,
            optimum_step,
            review_cost,
            "Expected net value (paid-out only)",
            min_reimbursement_threshold,
        ),
        "lar_heatmap": plot_lar_heatmap(pairs, or_mask_optimum, labels),
        "time_difference": plot_time_difference(pairs, or_mask_optimum, labels),
        "savings_by_time_difference": plot_savings_by_time_difference(
            pairs, or_mask_optimum, labels
        ),
    }


def _build_metrics(
    pairs: pl.DataFrame,
    or_mask_optimum: pl.Series,
    labels: pl.Series,
    stepwise: StepwiseNetValue,
    optimum_step: int,
) -> dict[str, float]:
    in_out_metrics = compute_in_stack_out_stack_metrics(pairs, or_mask_optimum, labels)
    metrics = {
        "optimum_step": float(optimum_step + 1),
        "optimum_pos_pred_pair_count": float(
            stepwise.pos_pred_pair_count[optimum_step]
        ),
        "optimum_true_positive_pair_count": float(
            stepwise.true_positive_pair_count[optimum_step]
        ),
        "optimum_total_savings": stepwise.total_savings[optimum_step],
        "optimum_paid_out_savings": stepwise.paid_out_savings[optimum_step],
        "optimum_net_value_total": stepwise.net_value_total[optimum_step],
        "optimum_net_value_paid_out": stepwise.net_value_paid_out[optimum_step],
        "window_total_pairs": float(pairs.height),
        "window_total_positives": float(int(labels.sum())),
    }
    metrics.update(in_out_metrics)
    return metrics


def plot_time_difference(pairs: pl.DataFrame, or_mask: pl.Series, labels: pl.Series):
    """Stacked bar chart: pos-pred and TP counts per |Doc.1 − Doc.2| day."""
    annotated = pairs.with_columns(
        or_mask.alias("_or_mask"),
        labels.alias("_labels"),
        (pl.col("HUKIMPORTTIME") - pl.col("HUKIMPORTTIME__2"))
        .dt.total_days()
        .alias("time_difference"),
    ).filter(
        pl.col("time_difference").is_not_null(),
        pl.col("time_difference").gt(0),
        pl.col("time_difference").lt(730),
    )

    pos_pred_days = (
        annotated.filter(pl.col("_or_mask"))
        .get_column("time_difference")
        .cast(pl.Int32)
        .to_numpy()
    )
    tp_days = (
        annotated.filter(pl.col("_or_mask") & pl.col("_labels"))
        .get_column("time_difference")
        .cast(pl.Int32)
        .to_numpy()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    if len(pos_pred_days) == 0:
        ax.text(
            0.5, 0.5, "No positive predictions", transform=ax.transAxes, ha="center"
        )
        ax.set_title("Time difference between positive predictions")
        return fig

    max_day = int(max(pos_pred_days.max(), tp_days.max() if len(tp_days) > 0 else 0))
    bins = range(0, max_day + 2)
    counts_pos_pred, _ = np.histogram(pos_pred_days, bins=bins)
    counts_tp, _ = np.histogram(tp_days, bins=bins)

    ax.bar(
        bins[:-1],
        counts_pos_pred,
        width=1.0,
        color=GREY_COLOR,
        label="Positive predictions",
    )
    ax.bar(bins[:-1], counts_tp, width=1.0, color=GREEN_COLOR, label="True positives")
    ax.set_xlabel("Time difference in days (Doc. 1 − Doc. 2)")
    ax.set_ylabel("Count")
    ax.set_title(
        "Time difference of positive predictions for the optimum composition",
        loc="left",
        pad=15,
    )
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_savings_by_time_difference(
    pairs: pl.DataFrame, or_mask: pl.Series, labels: pl.Series
):
    """Average min(MAX_REIMBURSEMENT, MAX_REIMBURSEMENT__2) binned by time-diff."""
    annotated = pairs.with_columns(
        or_mask.alias("_or_mask"),
        labels.alias("_labels"),
        (pl.col("HUKIMPORTTIME") - pl.col("HUKIMPORTTIME__2"))
        .dt.total_days()
        .alias("time_difference"),
        pl.min_horizontal("MAX_REIMBURSEMENT", "MAX_REIMBURSEMENT__2").alias(
            "pair_savings"
        ),
    ).filter(
        pl.col("time_difference").is_not_null(),
        pl.col("time_difference").gt(0),
        pl.col("time_difference").lt(730),
        pl.col("pair_savings").is_not_null(),
    )

    bin_width = 60
    annotated = annotated.with_columns(
        (pl.col("time_difference") // bin_width * bin_width).alias("td_bin")
    )

    def _binned_mean(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.group_by("td_bin")
            .agg(
                pl.col("pair_savings").mean().alias("avg_savings"), pl.len().alias("n")
            )
            .sort("td_bin")
        )

    all_positives = _binned_mean(annotated.filter(pl.col("_labels")))
    true_positives = _binned_mean(
        annotated.filter(pl.col("_labels") & pl.col("_or_mask"))
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    if all_positives.is_empty():
        ax.text(0.5, 0.5, "No labeled positives", transform=ax.transAxes, ha="center")
        ax.set_title("Average savings per time difference")
        return fig

    ax.plot(
        all_positives["td_bin"].to_numpy(),
        all_positives["avg_savings"].to_numpy(),
        color=GREEN_COLOR,
        linewidth=1.4,
        marker="o",
        markersize=4,
        label="All labeled positives",
    )
    if not true_positives.is_empty():
        ax.plot(
            true_positives["td_bin"].to_numpy(),
            true_positives["avg_savings"].to_numpy(),
            color=ACCENT_COLOR,
            linewidth=1.4,
            marker="o",
            markersize=4,
            label="True positives (rule hits)",
        )

    ax.set_xlabel("Time difference in days (Doc. 1 − Doc. 2), binned")
    ax.set_ylabel("Avg. min-reimbursement (€)")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:,.0f} €".replace(",", "."))
    )
    ax.set_title(
        "Average DZ savings per time difference",
        loc="left",
        fontweight="bold",
        pad=15,
    )
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        loc="best",
    )
    ax.spines[["top", "right"]].set_visible(False)
    return fig
