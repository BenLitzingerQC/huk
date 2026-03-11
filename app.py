"""Greedy Set Cover for Rule Mining — Self-Contained Version.

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

from math import ceil
from itertools import combinations
from collections import Counter

import polars as pl
from tqdm import tqdm


# --- Configuration ---------------------------------------------------------------

DATA_PATH = "data/df_full.parquet"       # <-- adjust path
LABEL_COL = "dz_interesting"
POSITIVE_RATE = 0.001                    # downsample negatives (None = keep all)
TEST_SPLIT = 0.2                         # stratified split (None = no split)

MAX_PREDICATES_PER_RULE = 3              # max AND-predicates per rule (k)
MAX_RULES = 1000                         # max rules in OR-composition
MIN_RECALL = 0.95                        # recall target
MIN_NEW_TP = 1                           # floor for adaptive TP threshold
MAX_CANDIDATES = 50_000                  # cap after diversity filter


# --- Threshold hierarchy ---------------------------------------------------------
# Ordered strictest → loosest within each family.
# Looseness = index in list (0 = strictest).

THRESHOLD_FAMILIES = {
    "eq_dist": ["eq", "dist_1", "dist_2", "dist_3", "dist_4"],
    "prop":    ["prop_0.99", "prop_0.95", "prop_0.9", "prop_0.8", "prop_0.7"],
    "overlap": ["overlap_0.75", "overlap_0.5", "overlap_0"],
}

# Derived lookups (built once at import time)
LOOSENESS = {}               # threshold string → int
THRESHOLD_TO_FAMILY = {}     # threshold string → family name
ALL_THRESHOLDS = []          # all thresholds, longest first (for prefix matching)

for _family, _thresholds in THRESHOLD_FAMILIES.items():
    for _i, _t in enumerate(_thresholds):
        LOOSENESS[_t] = _i
        THRESHOLD_TO_FAMILY[_t] = _family
        ALL_THRESHOLDS.append(_t)
ALL_THRESHOLDS.sort(key=len, reverse=True)


# --- Data loading ----------------------------------------------------------------

def detect_feature_groups(columns):
    """Auto-detect feature groups from column names like '{threshold}_{feature}'.

    Groups predicates by (feature, family) so that eq/dist, prop, and overlap
    stay in separate groups even for the same feature.
    Returns: list of (feature_name, [thresholds sorted by looseness])
    """
    ignore = {LABEL_COL, "dz_interesting", "dz", "labelled", "reclaim_list", "da_rule"}
    grouped = {}  # (feature, family) → [thresholds]

    for col in columns:
        if col in ignore:
            continue
        for threshold in ALL_THRESHOLDS:
            if col.startswith(threshold + "_"):
                feature = col[len(threshold) + 1:]
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


def load_data(path, label_col, positive_rate=None, test_split=None):
    """Load parquet, optionally downsample negatives and split train/test.

    Returns dict with keys: features, labels, groups,
    and optionally: test_features, test_labels.
    """
    schema_cols = pl.scan_parquet(path).collect_schema().names()
    groups = detect_feature_groups(schema_cols)
    pred_cols = [f"{t}_{feat}" for feat, thresholds in groups for t in thresholds]
    needed = pred_cols + [label_col]

    # Load with optional negative downsampling
    if positive_rate is not None:
        stats = pl.scan_parquet(path).select(
            pl.col(label_col).fill_null(0).cast(pl.Int8).sum().alias("n_pos"),
            pl.len().alias("n_total"),
        ).collect()
        n_pos = stats["n_pos"].item()
        n_total = stats["n_total"].item()
        neg_want = int(n_pos * (1 - positive_rate) / positive_rate)
        keep_every = max(1, (n_total - n_pos) // neg_want)

        df = (
            pl.scan_parquet(path).select(needed)
            .with_row_index("_idx")
            .filter(
                (pl.col(label_col).fill_null(0) == 1)
                | (pl.col("_idx") % keep_every == 0)
            )
            .drop("_idx").collect()
        )
        # shuffle=True required — sample(fraction=1.0) alone does NOT shuffle in Polars
        df = df.sample(fraction=1.0, seed=42, shuffle=True)
        print(f"Loaded {n_total:,} rows, sampled to {len(df):,} ({n_pos:,} positives)")
    else:
        df = pl.read_parquet(path, columns=needed)
        print(f"Loaded {len(df):,} rows")

    def to_bool(frame):
        labels = frame[label_col].fill_null(0).cast(pl.Boolean)
        features = {c: frame[c].fill_null(0).cast(pl.Boolean) for c in pred_cols}
        return features, labels

    result = {"groups": groups}

    if test_split and test_split > 0:
        pos_df = df.filter(pl.col(label_col).fill_null(0) == 1)
        neg_df = df.filter(pl.col(label_col).fill_null(0) != 1)
        n_pos_train = int(len(pos_df) * (1 - test_split))
        n_neg_train = int(len(neg_df) * (1 - test_split))

        train = pl.concat([pos_df.head(n_pos_train), neg_df.head(n_neg_train)])
        train = train.sample(fraction=1.0, seed=42, shuffle=True)
        test = pl.concat([
            pos_df.tail(len(pos_df) - n_pos_train),
            neg_df.tail(len(neg_df) - n_neg_train),
        ]).sample(fraction=1.0, seed=43, shuffle=True)

        result["features"], result["labels"] = to_bool(train)
        result["test_features"], result["test_labels"] = to_bool(test)
        n_tr = int(result["labels"].sum())
        n_te = int(result["test_labels"].sum())
        print(f"Split: train={len(train):,} ({n_tr:,} pos), test={len(test):,} ({n_te:,} pos)")
    else:
        result["features"], result["labels"] = to_bool(df)
        print(f"Positives: {int(result['labels'].sum()):,}")

    print(f"Predicates: {len(pred_cols)}, Groups: {len(groups)}")
    return result


# --- Phase 1: Apriori candidate generation ---------------------------------------

def apriori_candidates(features, groups, labels, max_k):
    """Generate AND-rules bottom-up with anti-monotone TP pruning.

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

    # Level 1: single predicates with TP >= 1
    prev_level = []
    for gi, col in predicates:
        mask = features[col]
        if int((mask & labels).sum()) >= 1:
            prev_level.append(((gi,), [col], mask))

    def score(rules):
        scored = []
        for _, names, mask in rules:
            tp = int((mask & labels).sum())
            pos = int(mask.sum())
            if tp > 0 and pos > 0:
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
                if int((new_mask & labels).sum()) >= 1:
                    next_level.append((group_ids + (gi,), names + [col], new_mask))
        if not next_level:
            break
        candidates.extend(score(next_level))
        prev_level = next_level

    return candidates


def diversity_filter(candidates, max_count):
    """Keep top N/2 by precision + top N/2 by TP count (deduplicated).

    Preserves both precise rules (for composition quality) and
    high-coverage rules (for reaching the recall target).
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

def greedy_select(candidates, labels, total_positives, n_rows,
                  min_recall, max_rules, min_new_tp):
    """Iteratively select the rule that maximizes marginal composition precision.

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
            new_tp = int((uncovered & labels).sum())

            # Dead: can never contribute again (OR mask only grows)
            if new_tp < 1:
                continue
            next_alive.append(ci)

            # Below adaptive threshold: keep alive but skip this step
            if new_tp < adaptive_min:
                continue

            new_pos = int(uncovered.sum())
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
        step += 1

        print(f"  step {step}: +{best_tp} tp, "
              f"prec={cum_tp / cum_pos:.4%}, "
              f"rec={cum_tp / total_positives:.4%}, "
              f"{len(alive)} alive")

    print(f"Selected {step} rules")
    return selected_rules, selected_masks


# --- Post-processing -------------------------------------------------------------

def looseness_of(col_name):
    """Extract looseness from a predicate column name (e.g. 'dist_2_end_date' → 2)."""
    for threshold in ALL_THRESHOLDS:
        if col_name.startswith(threshold + "_"):
            return LOOSENESS.get(threshold, 0)
    return 0


def dominates(rule_a, rule_b, col_to_group):
    """True if rule A dominates B, meaning B is redundant when A is present.

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
    """Iteratively remove rules whose TPs are fully covered by the other rules.

    Iterative (not batch) to avoid circular removals: if A covers B's TPs and
    B covers A's TPs, batch removal would drop both — losing coverage.
    """
    changed = True
    while changed:
        changed = False
        for i in range(len(rules)):
            others_or = pl.Series("", [False], dtype=pl.Boolean).new_from_index(0, n_rows)
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
    """Merge rules that share ≥2 common predicates into a single AND-rule.

    Greedily picks the most-frequent predicate pair, groups all rules
    containing that pair, and unions their predicates into one rule.
    Single pass (no cascading). May reduce recall.
    """
    if len(rules) <= 1:
        return [list(r) for r in rules]

    rule_sets = [set(r) for r in rules]
    used = [False] * len(rules)
    merged = []

    pair_counts = Counter()
    pair_to_rules = {}
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

    or_mask = None
    for rule in rules:
        and_mask = features[rule[0]]
        for pred in rule[1:]:
            and_mask = and_mask & features[pred]
        or_mask = and_mask if or_mask is None else (or_mask | and_mask)

    tp = int((or_mask & labels).sum())
    pos = int(or_mask.sum())
    return (tp / pos if pos > 0 else 0.0), (tp / total_pos if total_pos > 0 else 0.0)


# --- Main ------------------------------------------------------------------------

def main():
    data = load_data(DATA_PATH, LABEL_COL, POSITIVE_RATE, TEST_SPLIT)
    features, labels = data["features"], data["labels"]
    groups = data["groups"]
    total_pos = int(labels.sum())
    n_rows = len(labels)
    has_test = "test_features" in data

    # Phase 1: Apriori
    print(f"\n--- Phase 1: Apriori (max_k={MAX_PREDICATES_PER_RULE}) ---")
    candidates = apriori_candidates(features, groups, labels, MAX_PREDICATES_PER_RULE)
    n_before = len(candidates)
    candidates = diversity_filter(candidates, MAX_CANDIDATES)
    if len(candidates) < n_before:
        print(f"{n_before:,} candidates → {len(candidates):,} after diversity filter")
    else:
        print(f"{len(candidates):,} candidates")

    # Phase 2: Greedy
    print(f"\n--- Phase 2: Greedy (min_recall={MIN_RECALL}, max_rules={MAX_RULES}) ---")
    selected_rules, selected_masks = greedy_select(
        candidates, labels, total_pos, n_rows,
        MIN_RECALL, MAX_RULES, MIN_NEW_TP,
    )

    # Post-processing
    print(f"\n--- Post-processing ---")
    n0 = len(selected_rules)
    selected_rules, selected_masks = remove_subsumed(selected_rules, selected_masks, groups)
    if len(selected_rules) < n0:
        print(f"Subsumption: {n0} → {len(selected_rules)} rules")

    n1 = len(selected_rules)
    selected_rules, selected_masks = remove_redundant(
        selected_rules, selected_masks, labels, n_rows,
    )
    if len(selected_rules) < n1:
        print(f"Unique-TP:   {n1} → {len(selected_rules)} rules")

    merged = merge_rules(selected_rules)
    if len(merged) != len(selected_rules):
        print(f"Merge:       {len(selected_rules)} → {len(merged)} rules")

    # Evaluate
    print(f"\n--- Results ---")
    train_prec, train_rec = evaluate(merged, features, labels)
    line = f"Train: Precision={train_prec:.4%}, Recall={train_rec:.4%}"
    if has_test:
        test_prec, test_rec = evaluate(merged, data["test_features"], data["test_labels"])
        line += f"  |  Test: Precision={test_prec:.4%}, Recall={test_rec:.4%}"
    print(line)

    print(f"\nRules ({len(merged)}):")
    for i, rule in enumerate(merged):
        print(f"  Rule {i}: {' AND '.join(rule)}")

    print(f"\nYAML format:")
    print("rules:")
    for rule in merged:
        print(f"  - {' AND '.join(rule)}")


if __name__ == "__main__":
    main()
