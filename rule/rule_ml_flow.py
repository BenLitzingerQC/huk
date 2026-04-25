"""
Greedy Set Cover for Rule Mining — Self-Contained Version.

Finds an OR-composition of AND-rules that maximizes precision at a target
recall. Each row in the dataset has pre-computed boolean predicate columns
(e.g. eq_start_date, dist_2_end_date, overlap_0.5_TAB_BETR_POSITION).

Pipeline:
  1. Load parquet with boolean predicate columns
  2. Apriori: generate candidate AND-rules (bottom-up, anti-monotone pruning)
  3. Greedy: iteratively select rules by marginal composition precision
  4. Post-process: remove subsumed rules, remove redundant rules, merge
  5. Evaluate on train / test
"""

import logging
import sys
from collections import Counter
from itertools import combinations
from math import ceil
from typing import Any

import hydra
import matplotlib.pyplot as plt
import mlflow
import polars as pl
from evaluate_rule import evaluate_rule
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from da_hf5_dz.config import (
    OLD_DZ_RULE,
    AggregationIdentifiers,
    filepath_shared_folder,
    mlflow_tracking_uri,
)
from da_hf5_dz.configs.plotting.plotting import setup_plotting

# --- Configuration ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    stream=sys.stdout,
)

# --- Threshold hierarchy ---------------------------------------------------------
# Ordered strictest → loosest within each family.
# Looseness = index in list (0 = strictest).

THRESHOLD_FAMILIES = {
    "eq_dist": ["eq", "dist_1", "dist_2", "dist_3", "dist_4"],
    "prop": ["prop_0.99", "prop_0.95", "prop_0.9", "prop_0.8", "prop_0.7"],
    "overlap": ["overlap_0.75", "overlap_0.5", "overlap_0"],
}

# Derived lookups (built once at import time)
LOOSENESS = {}  # threshold string → int
THRESHOLD_TO_FAMILY = {}  # threshold string → family name
ALL_THRESHOLDS = []  # all thresholds, longest first (for prefix matching)

for _family, _thresholds in THRESHOLD_FAMILIES.items():
    for _i, _t in enumerate(_thresholds):
        LOOSENESS[_t] = _i
        THRESHOLD_TO_FAMILY[_t] = _family
        ALL_THRESHOLDS.append(_t)
ALL_THRESHOLDS.sort(key=len, reverse=True)


# --- Data loading ----------------------------------------------------------------


def detect_feature_groups(columns, label_col):
    """
    Auto-detect feature groups from column names like '{threshold}_{feature}'.

    Groups predicates by (feature, family) so that eq/dist, prop, and overlap
    stay in separate groups even for the same feature.
    Returns: list of (feature_name, [thresholds sorted by looseness])
    """
    ignore = {label_col, "dz_interesting", "dz", "labelled", "reclaim_list", "da_rule"}
    grouped: dict[Any, Any] = {}  # (feature, family) → [thresholds]

    for col in columns:
        if col in ignore:
            continue
        for threshold in ALL_THRESHOLDS:
            if col.startswith(threshold + "_"):
                feature = col[len(threshold) + 1 :]
                if not feature:
                    break
                family = THRESHOLD_TO_FAMILY.get(threshold, threshold)
                grouped.setdefault((feature, family), []).append(threshold)
                break

    groups = []
    for (feature, _), thresholds in sorted(grouped.items()):
        thresholds = sorted(set(thresholds), key=lambda t: LOOSENESS.get(t, 0))
        groups.append((feature, thresholds))
    return groups


def load_data(
    path,
    label_col,
    filter_expr,
    origin_col,
    origin_weights,
    origin_default_weight,
    positive_rate=None,
    test_split=None,
):
    """
    Load parquet, optionally downsample negatives and split train/test.

    Returns dict with keys: features, labels, groups,
    and optionally: test_features, test_labels.
    """

    schema_cols = pl.scan_parquet(path).collect_schema().names()
    groups = detect_feature_groups(schema_cols, label_col)
    pred_cols = [f"{t}_{feat}" for feat, thresholds in groups for t in thresholds]
    # Lar is needed because of OLD_DZ_RULE
    needed = pred_cols + [label_col, "lar", "lar__2", "HUKIMPORTTIME", "HUKIMPORTTIME__2", origin_col]

    # Load with optional negative downsampling
    if positive_rate is not None:
        stats = (
            pl.scan_parquet(path)
            .filter(filter_expr)
            .select(
                pl.col(label_col).cast(pl.Int8).sum().alias("n_pos"),
                pl.len().alias("n_total"),
            )
            .collect()
        )
        n_pos = stats["n_pos"].item()
        n_total = stats["n_total"].item()
        neg_want = int(n_pos * (1 - positive_rate) / positive_rate)
        keep_every = max(1, (n_total - n_pos) // neg_want)

        df = (
            pl.scan_parquet(path)
            .filter(filter_expr)
            .select(needed)
            .with_row_index("_idx")
            .filter(
                (pl.col(label_col).cast(pl.Int8).fill_null(0) == 1)
                | (pl.col("_idx") % keep_every == 0)
            )
            .drop("_idx")
            .collect()
        )
        # shuffle=True required — sample(fraction=1.0) alone does NOT shuffle in Polars
        df = df.sample(fraction=1.0, seed=42, shuffle=True)
        logging.info(
            f"Loaded {n_total:,} rows, sampled to {len(df):,} ({n_pos:,} positives)"
        )
    else:
        df = pl.read_parquet(path, columns=needed).filter(filter_expr)
        logging.info(f"Loaded {len(df):,} rows")

    def to_bool(frame):
        labels = frame[label_col].cast(pl.Boolean).fill_null(False)
        features = {c: frame[c].cast(pl.Boolean).fill_null(False) for c in pred_cols}
        weights = (
            frame[origin_col]
            .fill_null("")
            .map_elements(
                lambda o: origin_weights.get(o, origin_default_weight),
                return_dtype=pl.Float32,
            )
        )
        return features, labels, weights

    result = {"groups": groups}

    if test_split and test_split > 0:
        pos_df = df.filter(pl.col(label_col).cast(pl.Int8).fill_null(0) == 1)
        neg_df = df.filter(pl.col(label_col).cast(pl.Int8).fill_null(0) != 1)
        n_pos_train = int(len(pos_df) * (1 - test_split))
        n_neg_train = int(len(neg_df) * (1 - test_split))

        train = pl.concat([pos_df.head(n_pos_train), neg_df.head(n_neg_train)])
        train = train.sample(fraction=1.0, seed=42, shuffle=True)
        test = pl.concat(
            [
                pos_df.tail(len(pos_df) - n_pos_train),
                neg_df.tail(len(neg_df) - n_neg_train),
            ]
        ).sample(fraction=1.0, seed=43, shuffle=True)

        result["features"], result["labels"], result["weights"] = to_bool(train)
        result["test_features"], result["test_labels"], result["test_weights"] = (
            to_bool(test)
        )
        n_tr = int(result["labels"].sum())
        n_te = int(result["test_labels"].sum())
        logging.info(
            f"Split: train={len(train):,} ({n_tr:,} pos), test={len(test):,} ({n_te:,} pos)"
        )
    else:
        result["features"], result["labels"], result["weights"] = to_bool(df)
        logging.info(f"Positives: {int(result['labels'].sum()):,}")

    logging.info(f"Predicates: {len(pred_cols)}, Groups: {len(groups)}")
    return result


# --- Phase 1: Apriori candidate generation ---------------------------------------


def apriori_candidates(features, groups, labels, max_k, min_new_tp):
    """
    Generate AND-rules bottom-up with anti-monotone TP pruning.

    Starts with single predicates that have TP >= 1, then combines predicates
    from different feature groups up to max_k per rule.
    Anti-monotone: adding AND-predicates can only decrease TP, so any rule
    with TP=0 is pruned along with all its extensions.

    Returns: list of (predicate_names, mask, precision, tp_count)
    """
    predicates = [
        (gi, f"{t}_{feat}")
        for gi, (feat, thresholds) in enumerate(groups)
        for t in thresholds
    ]

    # Level 1: single predicates with TP >= MIN_NEW_TP
    prev_level = []
    for gi, col in predicates:
        mask = features[col]
        if int((mask & labels).sum()) >= min_new_tp:
            prev_level.append(((gi,), [col], mask))

    def score(rules):
        scored = []
        for _, names, mask in rules:
            tp = int((mask & labels).sum())
            pos = int(mask.sum())
            if tp >= min_new_tp and pos > 0:
                scored.append((names, mask, tp / pos, tp))
        return scored

    candidates = score(prev_level)

    # Extend level by level up to max_k
    for _ in tqdm(range(2, max_k + 1), desc="apriori"):
        next_level = []
        for group_ids, names, mask in prev_level:
            for gi, col in predicates:
                # Only add predicates from strictly higher group index (avoids duplicates)
                if gi <= group_ids[-1]:
                    continue
                new_mask = mask & features[col]
                if int((new_mask & labels).sum()) >= min_new_tp:
                    next_level.append((group_ids + (gi,), names + [col], new_mask))
        if not next_level:
            break
        candidates.extend(score(next_level))
        prev_level = next_level

    return candidates


def diversity_filter(candidates, max_count):
    """
    Keep top N/2 by precision + top N/2 by TP count (deduplicated).

    Preserves both precise rules (for composition quality) and high-coverage rules (for
    reaching the recall target).
    """
    if len(candidates) <= max_count:
        return candidates

    half = max_count // 2
    by_precision = sorted(range(len(candidates)), key=lambda i: -candidates[i][2])
    by_tp_count = sorted(range(len(candidates)), key=lambda i: -candidates[i][3])

    keep = set(by_precision[:half])
    for idx in by_tp_count:
        if len(keep) >= max_count:
            break
        keep.add(idx)

    return [candidates[i] for i in sorted(keep)]


# --- Phase 2: Greedy selection ---------------------------------------------------


def greedy_select(
    candidates,
    labels,
    weights,
    total_positives,
    n_rows,
    min_recall,
    max_rules,
    min_new_tp,
):
    """
    Iteratively select the rule that maximizes marginal composition precision.

    Scoring: marginal_precision = (cum_tp + new_tp) / (cum_pos + new_pos)
      where cum_tp/cum_pos are the cumulative stats of already-selected rules
      and new_tp/new_pos count only the UNCOVERED matches of the candidate.

    Adaptive threshold: each step requires new_tp >= max(min_new_tp, 1% of remaining).
      Rules below the threshold stay alive (threshold may decrease as remaining shrinks).
    Permanent pruning: rules with new_tp=0 are removed (OR mask only grows, so new_tp
      is monotonically non-increasing).
    """
    min_tp = ceil(min_recall * total_positives)

    selected_rules = []
    selected_masks = []
    prec_values = []
    rec_values = []
    num_pos_preds = []
    num_tps = []
    covered = pl.Series("", [False], dtype=pl.Boolean).new_from_index(0, n_rows)
    cum_tp = 0
    cum_pos = 0
    alive = list(range(len(candidates)))

    step = 0
    while cum_tp < min_tp and len(selected_rules) < max_rules:
        remaining = min_tp - cum_tp
        adaptive_min = max(min_new_tp, int(remaining * 0.01))

        best_score = -1.0
        best_idx = -1
        best_tp = 0
        best_pos = 0
        not_covered = ~covered
        next_alive = []

        for ci in alive:
            _, mask, _, _ = candidates[ci]
            uncovered = mask & not_covered
            new_tp = float((uncovered & labels).cast(pl.Float32).dot(weights))

            # Dead: can never contribute again (OR mask only grows)
            if new_tp < 1:
                continue
            next_alive.append(ci)

            # Below adaptive threshold: keep alive but skip this step
            if new_tp < adaptive_min:
                continue

            new_pos = float(uncovered.cast(pl.Float32).dot(weights))
            marginal_prec = (cum_tp + new_tp) / (cum_pos + new_pos)

            if marginal_prec > best_score:
                best_score = marginal_prec
                best_idx = ci
                best_tp = new_tp
                best_pos = new_pos

        alive = next_alive
        if best_idx < 0:
            break

        names, mask, _, _ = candidates[best_idx]
        selected_rules.append(names)
        selected_masks.append(mask)
        covered = covered | mask
        cum_tp += best_tp
        cum_pos += best_pos
        num_tps.append(cum_tp)
        num_pos_preds.append(cum_pos)
        prec = cum_tp / cum_pos
        rec = cum_tp / total_positives
        prec_values.append(prec)
        rec_values.append(rec)
        step += 1

        logging.info(
            f"  step {step}: +{best_tp} tp, "
            f"prec={prec:.4%}, "
            f"rec={rec:.4%}, "
            f"{len(alive)} alive"
        )
        logging.info(f"Rule: {names}")

    logging.info(f"Selected {step} rules")
    return (
        selected_rules,
        selected_masks,
        prec_values,
        rec_values,
        num_tps,
        num_pos_preds,
    )


# --- Post-processing -------------------------------------------------------------


def looseness_of(col_name):
    """Extract looseness from a predicate column name (e.g. 'dist_2_end_date' → 2)."""
    for threshold in ALL_THRESHOLDS:
        if col_name.startswith(threshold + "_"):
            return LOOSENESS.get(threshold, 0)
    return 0


def dominates(rule_a, rule_b, col_to_group):
    """
    True if rule A dominates B, meaning B is redundant when A is present.

    A dominates B iff for every predicate in A, rule B contains a predicate in
    the SAME feature group with equal or lower looseness (i.e. equal or stricter).
    B may have extra groups (making B even stricter → still dominated).
    Effect: mask(A) ⊇ mask(B), so B adds nothing to an OR with A.
    """
    # Map B's groups to their looseness
    b_looseness = {}
    for pred in rule_b:
        gi = col_to_group.get(pred)
        if gi is not None:
            b_looseness[gi] = looseness_of(pred)

    for pred in rule_a:
        gi = col_to_group.get(pred)
        if gi is None or gi not in b_looseness:
            return False
        if looseness_of(pred) < b_looseness[gi]:
            return False  # A is stricter here → A doesn't dominate
    return True


def remove_subsumed(rules, masks, groups):
    """Remove rules dominated by another rule in the set."""
    col_to_group = {}
    for gi, (feat, thresholds) in enumerate(groups):
        for t in thresholds:
            col_to_group[f"{t}_{feat}"] = gi

    n = len(rules)
    is_dominated = [False] * n
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # A can only dominate B if A has ≤ predicates (fewer = looser)
            if len(rules[i]) > len(rules[j]):
                continue
            if dominates(rules[i], rules[j], col_to_group):
                is_dominated[j] = True

    keep = [i for i in range(n) if not is_dominated[i]]
    return [rules[i] for i in keep], [masks[i] for i in keep]


def remove_redundant(rules, masks, labels, n_rows):
    """
    Iteratively remove rules whose TPs are fully covered by the other rules.

    Iterative (not batch) to avoid circular removals: if A covers B's TPs and B covers
    A's TPs, batch removal would drop both — losing coverage.
    """
    changed = True
    while changed:
        changed = False
        for i in range(len(rules)):
            others_or = pl.Series("", [False], dtype=pl.Boolean).new_from_index(
                0, n_rows
            )
            for j in range(len(rules)):
                if j != i:
                    others_or = others_or | masks[j]
            unique_tp = int((masks[i] & ~others_or & labels).sum())
            if unique_tp == 0:
                rules.pop(i)
                masks.pop(i)
                changed = True
                break
    return rules, masks


def merge_rules(rules):
    """
    Merge rules that share ≥2 common predicates into a single AND-rule.

    Greedily picks the most-frequent predicate pair, groups all rules containing that
    pair, and unions their predicates into one rule. Single pass (no cascading). May
    reduce recall.
    """
    if len(rules) <= 1:
        return [list(r) for r in rules]

    rule_sets = [set(r) for r in rules]
    used = [False] * len(rules)
    merged = []

    pair_counts: Counter[Any] = Counter()
    pair_to_rules: dict[Any, Any] = {}
    for i, preds in enumerate(rule_sets):
        for pair in combinations(sorted(preds), 2):
            pair_counts[pair] += 1
            pair_to_rules.setdefault(pair, []).append(i)

    for pair, count in pair_counts.most_common():
        if count < 2:
            break
        group = [i for i in pair_to_rules[pair] if not used[i]]
        if len(group) < 2:
            continue
        union = set()
        for i in group:
            union |= rule_sets[i]
            used[i] = True
        merged.append(sorted(union))

    for i, preds in enumerate(rule_sets):
        if not used[i]:
            merged.append(sorted(preds))

    return merged


# --- Evaluation ------------------------------------------------------------------


def evaluate(rules, features, labels):
    """Precision and recall of an OR-composition of AND-rules."""
    total_pos = int(labels.sum())
    if not rules:
        return 0.0, 0.0

    or_mask: Any = None
    for rule in rules:
        and_mask = features[rule[0]]
        for pred in rule[1:]:
            and_mask = and_mask & features[pred]
        or_mask = and_mask if or_mask is None else (or_mask | and_mask)

    tp = int((or_mask & labels).sum())
    pos = int(or_mask.sum())
    return (tp / pos if pos > 0 else 0.0), (tp / total_pos if total_pos > 0 else 0.0)


def build_runtime_config(cfg: dict) -> dict:
    """Reconstruct non-serializable objects from the raw YAML dict."""
    file_name = cfg["file_name"]
    data_path = filepath_shared_folder / cfg["data_subfolder"] / file_name

    f = cfg["filter"]
    label_col = cfg["label_col"]

    base_filter = (
        ~(OLD_DZ_RULE)
        & (
            ~pl.col("lar").str.starts_with(f["exclude_lar_prefix"])
            | ~pl.col("lar__2").str.starts_with(f["exclude_lar_prefix"])
        )
        & pl.col("lar").ne(f["exclude_lar_value"])
    )

    date_filter = pl.lit(True)
    if train_start := f.get("train_start"):
        date_filter = date_filter & pl.col("HUKIMPORTTIME").ge(
            pl.datetime(*[int(x) for x in train_start.split("-")])
        ) & pl.col("HUKIMPORTTIME__2").ge(
            pl.datetime(*[int(x) for x in train_start.split("-")])
        )
    if train_end := f.get("train_end"):
        date_filter = date_filter & pl.col("HUKIMPORTTIME").lt(
            pl.datetime(*[int(x) for x in train_end.split("-")])
        ) & pl.col("HUKIMPORTTIME__2").lt(
            pl.datetime(*[int(x) for x in train_end.split("-")])
        )

    # Positives bypass the date filter, negatives must pass it
    train_filter = base_filter & (
        pl.col(label_col).cast(pl.Boolean).fill_null(False) | date_filter
    )

    identifiers_map = {"dev": AggregationIdentifiers.dev()}
    identifiers = identifiers_map[cfg["identifiers"]]

    return {
        "FILE_NAME": file_name,
        "DATA_PATH": data_path,
        "FILTER": train_filter,
        "EVAL_FILTER": base_filter,
        "LABEL_COL": cfg["label_col"],
        "POSITIVE_RATE": cfg["positive_rate"],
        "TEST_SPLIT": cfg["test_split"],
        "FLATFILE_TYPE": cfg["flatfile_type"],
        "IDENTIFIERS": identifiers,
        "REVIEW_COST": cfg["review_cost"],
        "ORIGIN_COL": cfg["origin"]["col"],
        "ORIGIN_WEIGHTS": cfg["origin"]["weights"],
        "ORIGIN_DEFAULT_WEIGHT": cfg["origin"]["default_weight"],
        **{k.upper(): v for k, v in cfg["learning"].items()},
    }


# --- Main ------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../configs/rules", config_name="config")
def main(cfg: DictConfig):
    raw_cfg = OmegaConf.to_container(cfg, resolve=True)
    built = build_runtime_config(raw_cfg)
    FILE_NAME = built["FILE_NAME"]
    DATA_PATH = built["DATA_PATH"]
    FILTER = built["FILTER"]
    EVAL_FILTER = built["EVAL_FILTER"]
    LABEL_COL = built["LABEL_COL"]
    POSITIVE_RATE = built["POSITIVE_RATE"]
    TEST_SPLIT = built["TEST_SPLIT"]
    FLATFILE_TYPE = built["FLATFILE_TYPE"]
    IDENTIFIERS = built["IDENTIFIERS"]
    ORIGIN_COL = built["ORIGIN_COL"]
    ORIGIN_WEIGHTS = built["ORIGIN_WEIGHTS"]
    ORIGIN_DEFAULT_WEIGHT = built["ORIGIN_DEFAULT_WEIGHT"]
    MAX_PREDICATES_PER_RULE = built["MAX_PREDICATES_PER_RULE"]
    MAX_RULES = built["MAX_RULES"]
    MIN_RECALL = built["MIN_RECALL"]
    MIN_NEW_TP = built["MIN_NEW_TP"]
    MAX_CANDIDATES = built["MAX_CANDIDATES"]
    MERGE = built["MERGE"]
    REVIEW_COST = built["REVIEW_COST"]

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("DZ_test")

    with mlflow.start_run(
        run_name=f"MR:{MIN_RECALL} PR:{POSITIVE_RATE} MP:{MAX_PREDICATES_PER_RULE}",
        description=f"FN:{FILE_NAME}",
    ):
        data = load_data(
            DATA_PATH,
            LABEL_COL,
            FILTER,
            ORIGIN_COL,
            ORIGIN_WEIGHTS,
            ORIGIN_DEFAULT_WEIGHT,
            POSITIVE_RATE,
            TEST_SPLIT,
        )
        features, labels = data["features"], data["labels"]
        weights = data["weights"]
        groups = data["groups"]
        total_pos = float(labels.cast(pl.Float32).dot(weights))
        n_rows = len(labels)
        has_test = "test_features" in data

        # Phase 1: Apriori
        logging.info(f"\n--- Phase 1: Apriori (max_k={MAX_PREDICATES_PER_RULE}) ---")
        candidates = apriori_candidates(
            features, groups, labels, MAX_PREDICATES_PER_RULE, MIN_NEW_TP
        )
        n_before = len(candidates)
        candidates = diversity_filter(candidates, MAX_CANDIDATES)
        if len(candidates) < n_before:
            logging.info(
                f"{n_before:,} candidates → {len(candidates):,} after diversity filter"
            )
        else:
            logging.info(f"{len(candidates):,} candidates")

        # Phase 2: Greedy
        logging.info(
            f"\n--- Phase 2: Greedy (min_recall={MIN_RECALL}, max_rules={MAX_RULES}) ---"
        )
        (
            selected_rules,
            selected_masks,
            prec_values,
            rec_values,
            num_tps,
            num_pos_preds,
        ) = greedy_select(
            candidates,
            labels,
            weights,
            total_pos,
            n_rows,
            MIN_RECALL,
            MAX_RULES,
            MIN_NEW_TP,
        )

        # Post-processing
        logging.info("\n--- Post-processing ---")
        n0 = len(selected_rules)
        selected_rules_post_subsumed, selected_masks_post_subsumed = remove_subsumed(
            selected_rules, selected_masks, groups
        )
        if len(selected_rules_post_subsumed) < n0:
            logging.info(
                f"Subsumption: {n0} → {len(selected_rules_post_subsumed)} rules"
            )

        n1 = len(selected_rules_post_subsumed)
        selected_rules_post_redundant, selected_masks_post_redundant = remove_redundant(
            selected_rules_post_subsumed,
            selected_masks_post_subsumed,
            labels,
            n_rows,
        )
        if len(selected_rules_post_redundant) < n1:
            logging.info(
                f"Unique-TP:   {n1} → {len(selected_rules_post_redundant)} rules"
            )
        n2 = len(selected_rules_post_redundant)
        if MERGE:
            final_rules = merge_rules(selected_rules_post_redundant)
            if len(final_rules) != len(selected_rules_post_redundant):
                logging.info(
                    f"Merge:       {len(selected_rules_post_redundant)} → {len(final_rules)} rules"
                )
        else:
            final_rules = selected_rules_post_redundant
        # Evaluate
        logging.info("\n--- Results ---")
        train_prec, train_rec = evaluate(final_rules, features, labels)
        logging.info(f"Train: Precision={train_prec:.4%}, Recall={train_rec:.4%}")
        if has_test:
            test_prec, test_rec = evaluate(
                final_rules, data["test_features"], data["test_labels"]
            )
            logging.info(f"Test: Precision={test_prec:.4%}, Recall={test_rec:.4%}")

        logging.info(f"\nRules ({len(final_rules)}):")
        for i, rule in enumerate(final_rules):
            logging.info(f"  Rule {i}: {' AND '.join(rule)}")

        # ------------ Plots ------------
        # Plot precision Recall curve
        setup_plotting()
        evaluation_dict = evaluate_rule(
            selected_rules, final_rules, EVAL_FILTER, LABEL_COL, DATA_PATH, REVIEW_COST
        )

        for window_name, comp_dict in evaluation_dict.items():
            for comp_name, plot_dict in comp_dict.items():
                prefix = f"plots/{window_name}/{comp_name}"
                mlflow.log_figure(
                    plot_dict["pr_value"], f"{prefix}/prec_rec_value_.png"
                )
                mlflow.log_figure(plot_dict["lar_heatmap"], f"{prefix}/lar_heatmap.png")
                mlflow.log_figure(
                    plot_dict["time_difference"], f"{prefix}/time_difference.png"
                )

        plt.close("all")

        gw = evaluation_dict["November_2025__October_2025"]["greedy"]
        fw = evaluation_dict["November_2025__October_2025"]["final"]

        mlflow.log_params(
            {
                "DATA_PATH": DATA_PATH,
                "FILTER": str(raw_cfg.get("filter")),
                "LABEL_COL": LABEL_COL,
                "POSITIVE_RATE": POSITIVE_RATE,
                "TEST_SPLIT": TEST_SPLIT,
                "FLATFILE_TYPE": FLATFILE_TYPE,
                "IDENTIFIERS": raw_cfg.get("identifiers"),
                "ORIGIN_COL": ORIGIN_COL,
                "ORIGIN_WEIGHTS": str(ORIGIN_WEIGHTS),
                "ORIGIN_DEFAULT_WEIGHT": ORIGIN_DEFAULT_WEIGHT,
                "MAX_PREDICATES_PER_RULE": MAX_PREDICATES_PER_RULE,
                "MAX_RULES": MAX_RULES,
                "MIN_RECALL": MIN_RECALL,
                "MIN_NEW_TP": MIN_NEW_TP,
                "MAX_CANDIDATES": MAX_CANDIDATES,
                "MERGE": MERGE,
                "REVIEW_COST": REVIEW_COST,
            }
        )
        mlflow.log_metrics(
            {
                "01_NUMBER OF PAIRS": float(n_rows),
                "02_NUMBER OF RULE HITS": num_pos_preds[-1],
                "03_NUMBER OF TRUE POSITIVES": num_tps[-1],
                "04_TOTAL POSITIVES": total_pos,
                "05_TRAIN PRECISION": prec_values[-1],
                "06_TRAIN RECALL": rec_values[-1],
                "07_NUMBER OF RULES": n0,
                "08_NUMBER OF RULES POST SUBSUMED": n1,
                "09_NUMBER OF RULES POST REDUNDANT": n2,
                "10_IN_STACK_TOTAL_POS": float(gw["in_stack_stats"]["total_pos"]),
                "11_IN_STACK_POS_PRED": float(gw["in_stack_stats"]["pos_pred"]),
                "12_IN_STACK_TP": float(gw["in_stack_stats"]["tp"]),
                "13_OUT_STACK_TOTAL_POS": float(gw["out_of_stack_stats"]["total_pos"]),
                "14_OUT_STACK_TP": float(gw["out_of_stack_stats"]["tp"]),
                "15_OUT_STACK_POS_PRED": float(gw["out_of_stack_stats"]["pos_pred"]),
                # TP-only savings (was "savings" before)
                "16_ALL_POS TOTAL SAVINGS": gw["savings_all_pos"]["total_savings"],
                "17_ALL_POS AVG SAVINGS": gw["savings_all_pos"]["average_savings"],
                "18_BOTH PAID TOTAL_SAVINGS": gw["savings_all_pos"]["both_paid_total"],
                "19_BOTH PAID AVG SAVINGS": gw["savings_all_pos"]["both_paid_avg"],
                "20_TPS TOTAL SAVINGS": gw["savings_tp"]["total_savings"],
                "21_TPS_AVG_SAVINGS": gw["savings_tp"]["average_savings"],
                "22_TPS_BOTH PAID TOTAL_SAVINGS": gw["savings_tp"]["both_paid_total"],
                "23_TPS_BOTH PAID AVG SAVINGS": gw["savings_tp"]["both_paid_avg"],
            }
        )

        mlflow.log_dict(
            {
                "RULES POST GREEDY": selected_rules,
                "RULES POST SUBSUMED": selected_rules_post_subsumed,
                "RULES POST REDUNDANT": selected_rules_post_redundant,
            },
            "rules.json",
        )

        mlflow.log_dict(
            evaluation_dict,
            artifact_file="evaluation.json"
        )


if __name__ == "__main__":
    main()
