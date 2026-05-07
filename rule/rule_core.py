"""
Pure rule-mining logic — no Hydra/MLflow/external project dependencies.

Terminology:
  - predicate:   one boolean column (e.g. 'eq_start_date')
  - rule:        an AND-conjunction of predicates
  - composition: an OR-disjunction of rules (the final classifier)

Pipeline functions:
  - load_data: read parquet, optionally downsample, train/test split
  - generate_candidate_rules: bottom-up AND-rule mining with anti-monotone pruning
  - find_composition: greedily pick rules to maximize marginal precision
  - remove_subsumed / remove_redundant: post-processing
  - evaluate: precision / recall of a composition
"""

import logging
from dataclasses import dataclass, field
from math import ceil
from typing import Any

import polars as pl
from tqdm import tqdm


@dataclass
class Rule:
    """An AND-conjunction of predicates with its row mask.

    `train_precision` and `train_true_positives` are populated during candidate
    generation against the training data and remain unchanged downstream.
    """

    predicates: list[str]
    mask: pl.Series
    train_precision: float | None = None
    train_true_positives: int | None = None


@dataclass
class Composition:
    """An OR-composition of Rules with per-step statistics from selection."""

    rules: list[Rule] = field(default_factory=list)
    step_precision: list = field(default_factory=list)
    step_recall: list = field(default_factory=list)
    step_true_positives: list = field(default_factory=list)
    step_positive_predictions: list = field(default_factory=list)


# --- Group construction from predicate config ------------------------------------
# A "group" = (feature_name, [predicate_cols ordered strictest → loosest]).
# Index in the list is the predicate's "looseness" (0 = strictest).


def build_groups_from_config(cfg):
    """
    Build groups from the predicate-generation config.

    Expected cfg keys:
      calculate_equality:   list[feature]                              → 'eq_{feat}'
      calculate_lev_dist:   {feature: [{thresh: int}, ...]}            → 'dist_{t}_{feat}'
      calculate_proportion: {feature: [{thresh: float}, ...]}          → 'prop_{t}_{feat}'
      calculate_overlap:    {feature: [{thresh: float}, ...]}          → 'overlap_{t}_{feat}'

    Strictness convention: a stricter predicate is satisfied by fewer rows.
      - dist:    smaller thresh = stricter → ascending
      - prop:    larger  thresh = stricter → descending
      - overlap: larger  thresh = stricter → descending
    """
    groups: list[tuple[str, list[str]]] = []

    for feat in cfg.get("calculate_equality", []) or []:
        groups.append((feat, [f"eq_{feat}"]))

    for feat, entries in (cfg.get("calculate_lev_dist") or {}).items():
        threshs = sorted(e["thresh"] for e in entries)
        groups.append((feat, [f"dist_{t}_{feat}" for t in threshs]))

    for feat, entries in (cfg.get("calculate_proportion") or {}).items():
        threshs = sorted((e["thresh"] for e in entries), reverse=True)
        groups.append((feat, [f"prop_{t}_{feat}" for t in threshs]))

    for feat, entries in (cfg.get("calculate_overlap") or {}).items():
        threshs = sorted((e["thresh"] for e in entries), reverse=True)
        groups.append((feat, [f"overlap_{t}_{feat}" for t in threshs]))

    return groups


def build_lookups(groups):
    """Return (predicate_to_group, looseness) dicts from a groups list."""
    predicate_to_group = {
        predicate: group_index
        for group_index, (_, predicate_columns) in enumerate(groups)
        for predicate in predicate_columns
    }
    looseness = {
        predicate: i
        for _, predicate_columns in groups
        for i, predicate in enumerate(predicate_columns)
    }
    return predicate_to_group, looseness


# --- Data loading ----------------------------------------------------------------


def load_data(
    path,
    label_col,
    filter_expr,
    origin_col,
    groups,
    positive_rate=None,
    test_split=None,
):
    """
    Load parquet, optionally downsample negatives and split train/test.

    Returns dict with keys: features, labels,
    and optionally: test_features, test_labels.
    """
    predicate_columns: list[str] = [
        predicate for _, predicates_in_group in groups for predicate in predicates_in_group
    ]
    needed: list[str] = predicate_columns + [
        label_col,
        "lar",
        "lar__2",
        "HUKIMPORTTIME",
        "HUKIMPORTTIME__2",
        origin_col,
    ]

    if positive_rate is not None:
        stats: pl.DataFrame = (
            pl.scan_parquet(path)
            .filter(filter_expr)
            .select(
                pl.col(label_col).cast(pl.Int8).sum().alias("n_pos"),
                (pl.col(label_col).is_not_null() & ~pl.col(label_col).cast(pl.Boolean))
                .sum()
                .alias("n_labeled_neg"),
                pl.col(label_col).is_null().sum().alias("n_unlabeled"),
                pl.len().alias("n_total"),
            )
            .collect()
        )
        n_pos: int = stats["n_pos"].item()
        n_labeled_neg: int = stats["n_labeled_neg"].item()
        n_unlabeled: int = stats["n_unlabeled"].item()
        n_total: int = stats["n_total"].item()

        total_neg_want: int = int(n_pos * (1 - positive_rate) / positive_rate)
        n_unlabeled_want: int = max(0, total_neg_want - n_labeled_neg)
        keep_every: int | None = (
            max(1, n_unlabeled // n_unlabeled_want) if n_unlabeled_want > 0 else None
        )

        unlabeled_keep_expr: pl.Expr = (
            pl.lit(False) if keep_every is None else (pl.col("_idx") % keep_every == 0)
        )

        df = (
            pl.scan_parquet(path)
            .filter(filter_expr)
            .select(needed)
            .with_row_index("_idx")
            .filter(pl.col(label_col).is_not_null() | unlabeled_keep_expr)
            .drop("_idx")
            .collect()
            .sample(fraction=1.0, seed=42, shuffle=True)
        )

        unlabeled_kept: int = min(n_unlabeled_want, n_unlabeled)
        logging.info(
            f"Loaded {n_total:,} rows "
            f"(pos={n_pos:,}, labeled_neg={n_labeled_neg:,}, unlabeled={n_unlabeled:,}), "
            f"sampled to {len(df):,} — target neg={total_neg_want:,}, "
            f"unlabeled kept≈{unlabeled_kept:,}"
        )
    else:
        df = pl.read_parquet(path, columns=needed).filter(filter_expr)
        logging.info(f"Loaded {len(df):,} rows")

    def _to_bool(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.Series]:
        features = frame.select(
            [pl.col(c).cast(pl.Boolean).fill_null(False) for c in predicate_columns]
        )
        labels = frame[label_col].cast(pl.Boolean).fill_null(False)
        return features, labels

    result: dict[str, Any] = {}

    if test_split and test_split > 0:
        pos_df = df.filter(pl.col(label_col).cast(pl.Int8).fill_null(0) == 1)
        neg_df = df.filter(pl.col(label_col).cast(pl.Int8).fill_null(0) != 1)
        n_pos_train = int(len(pos_df) * (1 - test_split))
        n_neg_train = int(len(neg_df) * (1 - test_split))

        train = pl.concat(
            [pos_df.head(n_pos_train), neg_df.head(n_neg_train)]
        ).sample(fraction=1.0, seed=42, shuffle=True)
        test = pl.concat(
            [
                pos_df.tail(len(pos_df) - n_pos_train),
                neg_df.tail(len(neg_df) - n_neg_train),
            ]
        ).sample(fraction=1.0, seed=43, shuffle=True)

        result["features"], result["labels"] = _to_bool(train)
        result["test_features"], result["test_labels"] = _to_bool(test)
        n_tr = int(result["labels"].sum())
        n_te = int(result["test_labels"].sum())
        logging.info(
            f"Split: train={len(train):,} ({n_tr:,} pos), test={len(test):,} ({n_te:,} pos)"
        )
    else:
        result["features"], result["labels"] = _to_bool(df)
        logging.info(f"Positives: {int(result['labels'].sum()):,}")

    logging.info(f"Predicates: {len(predicate_columns)}, Groups: {len(groups)}")
    return result


# --- Phase 1: Generate candidate rules ----------------------------------------


def generate_candidate_rules(features, groups, labels, max_preds, min_new_tp):
    """
    Generate AND-rules bottom-up with anti-monotone TP pruning.

    Starts with single predicates that have TP >= 1, then combines predicates
    from different feature groups up to max_preds per rule.
    Anti-monotone: adding AND-predicates can only decrease TP, so any rule
    with TP=0 is pruned along with all its extensions.

    Returns: list of Rule (with train_precision and train_true_positives populated).
    """
    indexed_predicates: list[tuple[int, str]] = [
        (group_index, predicate)
        for group_index, (_, group_predicates) in enumerate(groups)
        for predicate in group_predicates
    ]

    previous_level = []
    for group_index, predicate in indexed_predicates:
        mask: pl.Series = features[predicate]
        if int((mask & labels).sum()) >= min_new_tp:
            previous_level.append(((group_index,), [predicate], mask))

    def score(raw_rules) -> list[Rule]:
        scored = []
        for _, predicates, mask in raw_rules:
            true_positives = int((mask & labels).sum())
            positive_predictions = int(mask.sum())
            if true_positives >= min_new_tp and positive_predictions > 0:
                scored.append(Rule(
                    predicates=predicates,
                    mask=mask,
                    train_precision=true_positives / positive_predictions,
                    train_true_positives=true_positives,
                ))
        return scored

    candidates: list[Rule] = score(previous_level)

    for _ in tqdm(range(2, max_preds + 1), desc="Generate candidate rules"):
        next_level = []
        for group_ids, predicates, mask in previous_level:
            for group_index, predicate in indexed_predicates:
                if group_index <= group_ids[-1]:
                    continue
                new_mask = mask & features[predicate]
                if int((new_mask & labels).sum()) >= min_new_tp:
                    next_level.append((
                        group_ids + (group_index,),
                        predicates + [predicate],
                        new_mask,
                    ))
        if not next_level:
            break
        candidates.extend(score(next_level))
        previous_level = next_level

    return candidates


# --- Phase 2: Composition selection ----------------------------------------------


def find_composition(
    candidates,
    labels,
    min_recall,
    max_rules,
    min_new_tp,
):
    """
    Greedily build a composition: at each step add the rule maximizing
    marginal precision (cumulative TP + new TP) / (cumulative pos. preds. + new pos. preds.),
    where "new" counts only rows not yet covered by the composition.

    Adaptive threshold: each step requires new TP >= max(min_new_tp, 1% of remaining target).
    Permanent pruning: candidates with new TP = 0 are removed forever
      (the composition mask only grows, so new TP is monotonically non-increasing).
    """
    total_positives = int(labels.sum())
    min_true_positives = ceil(min_recall * total_positives)

    composition = Composition()
    covered = pl.Series("", [False], dtype=pl.Boolean).new_from_index(0, len(labels))
    cumulative_true_positives = 0
    cumulative_positive_predictions = 0
    alive = list(range(len(candidates)))

    while (
        cumulative_true_positives < min_true_positives
        and len(composition.rules) < max_rules
    ):
        remaining = min_true_positives - cumulative_true_positives
        adaptive_min_true_positives = max(min_new_tp, int(remaining * 0.01))

        best_score = -1
        best_index = -1
        best_true_positives = 0
        best_positive_predictions = 0
        not_covered = ~covered
        next_alive = []

        for candidate_index in alive:
            uncovered = candidates[candidate_index].mask & not_covered
            new_true_positives = (uncovered & labels).sum()

            if new_true_positives < 1:
                continue
            next_alive.append(candidate_index)

            if new_true_positives < adaptive_min_true_positives:
                continue

            new_positive_predictions = uncovered.sum()
            marginal_precision = (
                (cumulative_true_positives + new_true_positives)
                / (cumulative_positive_predictions + new_positive_predictions)
            )

            if marginal_precision > best_score:
                best_score = marginal_precision
                best_index = candidate_index
                best_true_positives = new_true_positives
                best_positive_predictions = new_positive_predictions

        alive = next_alive
        if best_index < 0:
            break

        best = candidates[best_index]
        covered = covered | best.mask
        cumulative_true_positives += best_true_positives
        cumulative_positive_predictions += best_positive_predictions
        precision = cumulative_true_positives / cumulative_positive_predictions
        recall = cumulative_true_positives / total_positives

        composition.rules.append(best)
        composition.step_precision.append(precision)
        composition.step_recall.append(recall)
        composition.step_true_positives.append(cumulative_true_positives)
        composition.step_positive_predictions.append(cumulative_positive_predictions)

        logging.info(
            f"  step {len(composition.rules)}: +{best_true_positives} tp, "
            f"precision={precision:.4%}, recall={recall:.4%}, {len(alive)} alive"
        )
        logging.info(f"Rule: {best.predicates}")

    logging.info(f"Composition has {len(composition.rules)} rules")
    return composition


# --- Post-processing -------------------------------------------------------------


def dominates(rule_a, rule_b, predicate_to_group, looseness):
    """
    True if rule A dominates B, meaning B is redundant when A is present.

    A dominates B iff for every predicate in A, rule B contains a predicate in
    the SAME feature group with equal or lower looseness (i.e. equal or stricter).
    B may have extra groups (making B even stricter → still dominated).
    Effect: mask(A) ⊇ mask(B), so B adds nothing to an OR with A.
    """
    b_looseness = {
        predicate_to_group[p]: looseness[p]
        for p in rule_b
        if p in predicate_to_group
    }

    for predicate in rule_a:
        group_index = predicate_to_group.get(predicate)
        if group_index is None or group_index not in b_looseness:
            return False
        if looseness[predicate] < b_looseness[group_index]:
            return False
    return True


def remove_subsumed(rules, groups):
    """Remove rules dominated by another rule in the set."""
    predicate_to_group, looseness = build_lookups(groups)

    n = len(rules)
    is_dominated = [False] * n
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            if len(rules[i].predicates) > len(rules[j].predicates):
                continue
            if dominates(
                rules[i].predicates,
                rules[j].predicates,
                predicate_to_group,
                looseness,
            ):
                is_dominated[j] = True

    return [rules[i] for i in range(n) if not is_dominated[i]]


def remove_redundant(rules, labels):
    """
    Iteratively remove rules whose true positives are fully covered by the other rules.

    Iterative (not batch) to avoid circular removals: if A covers B's true positives
    and B covers A's true positives, batch removal would drop both — losing coverage.
    """
    changed = True
    while changed:
        changed = False
        for i in range(len(rules)):
            others_or = pl.Series("", [False], dtype=pl.Boolean).new_from_index(
                0, len(labels)
            )
            for j in range(len(rules)):
                if j != i:
                    others_or = others_or | rules[j].mask
            unique_true_positives = int(
                (rules[i].mask & ~others_or & labels).sum()
            )
            if unique_true_positives == 0:
                rules.pop(i)
                changed = True
                break
    return rules


# --- Evaluation ------------------------------------------------------------------


def evaluate(rules, features, labels):
    """
    Precision and recall of a composition (OR of AND-rules).

    `rules` is a list of Rule objects. The masks attached to those Rules are
    NOT used here — they were computed against the training data, while this
    function evaluates against whatever `features` DataFrame is passed in.
    """
    total_positives = int(labels.sum())
    if not rules:
        return 0.0, 0.0

    composition_mask: Any = None
    for rule in rules:
        rule_mask = features[rule.predicates[0]]
        for predicate in rule.predicates[1:]:
            rule_mask = rule_mask & features[predicate]
        composition_mask = (
            rule_mask if composition_mask is None else (composition_mask | rule_mask)
        )

    true_positives = int((composition_mask & labels).sum())
    positive_predictions = int(composition_mask.sum())
    precision = true_positives / positive_predictions if positive_predictions > 0 else 0.0
    recall = true_positives / total_positives if total_positives > 0 else 0.0
    return precision, recall
