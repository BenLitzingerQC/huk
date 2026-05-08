import pytest
import polars as pl
from rule_core import (
    Rule,
    build_groups_from_config,
    build_lookups,
    dominates,
    evaluate,
    find_composition,
    generate_candidate_rules,
    remove_redundant,
    remove_subsumed,
)


# --- Helpers ---------------------------------------------------------------------


def _bool_series(values):
    return pl.Series("", values, dtype=pl.Boolean)


def _make_features(spec):
    """spec: dict[str, list[bool]] -> pl.DataFrame of boolean columns."""
    return pl.DataFrame({name: _bool_series(vals) for name, vals in spec.items()})


def _rule(*predicates):
    """Build a Rule with a dummy mask (mask is unused by `evaluate`)."""
    return Rule(list(predicates), _bool_series([]))


# --- build_groups_from_config ----------------------------------------------------


def test_build_groups_empty_config_returns_empty():
    assert build_groups_from_config({}) == []


def test_build_groups_equality_one_predicate_per_feature():
    cfg = {"calculate_equality": ["start_date", "end_date"]}
    assert build_groups_from_config(cfg) == [
        ("start_date", ["eq_start_date"]),
        ("end_date", ["eq_end_date"]),
    ]


def test_build_groups_dist_sorted_ascending_strictest_first():
    # Smaller dist threshold = stricter → ascending.
    cfg = {"calculate_lev_dist": {"date": [{"thresh": 2}, {"thresh": 1}]}}
    assert build_groups_from_config(cfg) == [("date", ["dist_1_date", "dist_2_date"])]


def test_build_groups_prop_sorted_descending_strictest_first():
    # Larger prop threshold = stricter → descending.
    cfg = {"calculate_proportion": {"betrag": [{"thresh": 0.95}, {"thresh": 0.99}]}}
    assert build_groups_from_config(cfg) == [
        ("betrag", ["prop_0.99_betrag", "prop_0.95_betrag"])
    ]


def test_build_groups_overlap_sorted_descending_strictest_first():
    cfg = {"calculate_overlap": {"pos": [{"thresh": 0}, {"thresh": 0.5}]}}
    assert build_groups_from_config(cfg) == [
        ("pos", ["overlap_0.5_pos", "overlap_0_pos"])
    ]


def test_build_groups_separates_families_for_same_feature():
    # eq + prop on same feature → two separate groups, predicates don't mix.
    cfg = {
        "calculate_equality": ["betrag"],
        "calculate_proportion": {"betrag": [{"thresh": 0.99}]},
    }
    groups = build_groups_from_config(cfg)
    assert len(groups) == 2
    assert ("betrag", ["eq_betrag"]) in groups
    assert ("betrag", ["prop_0.99_betrag"]) in groups


def test_build_groups_handles_none_sections():
    # Hydra often produces None for missing sections, not absent keys.
    cfg = {"calculate_equality": None, "calculate_lev_dist": None}
    assert build_groups_from_config(cfg) == []


# --- build_lookups ---------------------------------------------------------------


def test_build_lookups_assigns_group_index_per_predicate():
    groups = [
        ("date", ["eq_date", "dist_1_date"]),
        ("betrag", ["eq_betrag"]),
    ]
    predicate_to_group, looseness = build_lookups(groups)
    assert predicate_to_group == {"eq_date": 0, "dist_1_date": 0, "eq_betrag": 1}
    assert looseness == {"eq_date": 0, "dist_1_date": 1, "eq_betrag": 0}


# --- dominates -------------------------------------------------------------------


@pytest.fixture
def lookups():
    # group 0 = 'date' (eq stricter than dist_1 stricter than dist_2)
    # group 1 = 'betrag' (eq stricter than prop_0.95)
    groups = [
        ("date", ["eq_date", "dist_1_date", "dist_2_date"]),
        ("betrag", ["eq_betrag", "prop_0.95_betrag"]),
    ]
    return build_lookups(groups)


def test_dominates_identical_rules(lookups):
    predicate_to_group, looseness = lookups
    rule = ["eq_date", "eq_betrag"]
    assert dominates(rule, rule, predicate_to_group, looseness) is True


def test_dominates_looser_dominates_stricter(lookups):
    # A is looser on 'date' (dist_2 > eq), so mask(A) ⊇ mask(B).
    predicate_to_group, looseness = lookups
    a = ["dist_2_date"]
    b = ["eq_date"]
    assert dominates(a, b, predicate_to_group, looseness) is True


def test_dominates_stricter_does_not_dominate_looser(lookups):
    predicate_to_group, looseness = lookups
    a = ["eq_date"]
    b = ["dist_2_date"]
    assert dominates(a, b, predicate_to_group, looseness) is False


def test_dominates_b_with_extra_group_is_still_dominated(lookups):
    # B has an extra predicate (extra AND-condition makes B stricter).
    predicate_to_group, looseness = lookups
    a = ["eq_date"]
    b = ["eq_date", "eq_betrag"]
    assert dominates(a, b, predicate_to_group, looseness) is True


def test_dominates_a_with_extra_group_does_not_dominate(lookups):
    # A requires 'betrag' but B doesn't constrain it → A is stricter on betrag.
    predicate_to_group, looseness = lookups
    a = ["eq_date", "eq_betrag"]
    b = ["eq_date"]
    assert dominates(a, b, predicate_to_group, looseness) is False


def test_dominates_unknown_predicate_breaks_domination(lookups):
    predicate_to_group, looseness = lookups
    a = ["unknown_col"]
    b = ["eq_date"]
    assert dominates(a, b, predicate_to_group, looseness) is False


# --- evaluate --------------------------------------------------------------------


def test_evaluate_perfect_rule():
    features = _make_features({"a": [True, True, False, False]})
    labels = _bool_series([True, True, False, False])
    prec, rec = evaluate([_rule("a")], features, labels)
    assert prec == 1.0
    assert rec == 1.0


def test_evaluate_returns_zero_for_empty_rule_list():
    features = _make_features({"a": [True, False]})
    labels = _bool_series([True, False])
    assert evaluate([], features, labels) == (0.0, 0.0)


def test_evaluate_no_positives_in_data_gives_zero_recall():
    features = _make_features({"a": [True, True]})
    labels = _bool_series([False, False])
    prec, rec = evaluate([_rule("a")], features, labels)
    # Mask hits 2 rows but 0 are TP → precision 0, recall 0 (no positives).
    assert prec == 0.0
    assert rec == 0.0


def test_evaluate_rule_never_fires_returns_zero_precision():
    features = _make_features({"a": [False, False, False]})
    labels = _bool_series([True, False, True])
    prec, rec = evaluate([_rule("a")], features, labels)
    assert prec == 0.0
    assert rec == 0.0


def test_evaluate_and_within_rule():
    # Rule = a AND b. Only row 0 satisfies both.
    features = _make_features(
        {
            "a": [True, True, False],
            "b": [True, False, True],
        }
    )
    labels = _bool_series([True, True, False])
    prec, rec = evaluate([_rule("a", "b")], features, labels)
    assert prec == 1.0  # 1 TP out of 1 hit
    assert rec == 0.5   # 1 TP out of 2 positives


def test_evaluate_or_across_rules():
    # Rule 1: a, Rule 2: b. OR = a | b.
    features = _make_features(
        {
            "a": [True, False, False, False],
            "b": [False, True, False, False],
        }
    )
    labels = _bool_series([True, True, True, False])
    prec, rec = evaluate([_rule("a"), _rule("b")], features, labels)
    assert prec == 1.0           # 2 TPs out of 2 hits
    assert rec == pytest.approx(2 / 3)


def test_evaluate_mixed_precision():
    features = _make_features({"a": [True, True, True, False]})
    labels = _bool_series([True, True, False, False])
    prec, rec = evaluate([_rule("a")], features, labels)
    assert prec == pytest.approx(2 / 3)
    assert rec == 1.0


# --- remove_subsumed -------------------------------------------------------------


def test_remove_subsumed_drops_dominated_rule():
    # Group 0: 'date' family. dist_2 dominates eq.
    groups = [("date", ["eq_date", "dist_1_date", "dist_2_date"])]
    rules = [
        Rule(["dist_2_date"], _bool_series([True, True, False])),
        Rule(["eq_date"], _bool_series([True, False, False])),
    ]

    kept = remove_subsumed(rules, groups)
    assert len(kept) == 1
    assert kept[0].predicates == ["dist_2_date"]


def test_remove_subsumed_keeps_independent_rules():
    groups = [("date", ["eq_date", "dist_1_date"]), ("betrag", ["eq_betrag"])]
    rules = [
        Rule(["eq_date"], _bool_series([True])),
        Rule(["eq_betrag"], _bool_series([True])),
    ]

    kept = remove_subsumed(rules, groups)
    kept_predicates = sorted(sorted(r.predicates) for r in kept)
    assert kept_predicates == [["eq_betrag"], ["eq_date"]]


def test_remove_subsumed_empty_input():
    assert remove_subsumed([], []) == []


# --- remove_redundant ------------------------------------------------------------


def test_remove_redundant_drops_rule_with_no_unique_tps():
    # Rule 0 covers TPs at index 0,1. Rule 1 covers same TPs (subset) → redundant.
    rules = [
        Rule(["a"], _bool_series([True, True, False])),
        Rule(["b"], _bool_series([True, False, False])),
    ]
    labels = _bool_series([True, True, False])
    kept = remove_redundant(rules, labels)
    assert len(kept) == 1
    assert kept[0].predicates == ["a"]


def test_remove_redundant_keeps_rules_with_unique_tps():
    rules = [
        Rule(["a"], _bool_series([True, False, False])),
        Rule(["b"], _bool_series([False, True, False])),
    ]
    labels = _bool_series([True, True, False])
    kept = remove_redundant(rules, labels)
    kept_predicates = sorted(sorted(r.predicates) for r in kept)
    assert kept_predicates == [["a"], ["b"]]


def test_remove_redundant_keeps_one_when_two_rules_cover_same_tp():
    rules = [
        Rule(["a"], _bool_series([True])),
        Rule(["b"], _bool_series([True])),
    ]
    labels = _bool_series([True])
    kept = remove_redundant(rules, labels)
    assert len(kept) == 1


def test_remove_redundant_empty_input_is_noop():
    assert remove_redundant([], _bool_series([])) == []


# --- generate_candidate_rules ----------------------------------------------------


def test_generate_candidates_single_predicate_meeting_min_tp():
    features = _make_features({"eq_a": [True, True, False]})
    labels = _bool_series([True, True, False])
    groups = [("a", ["eq_a"])]

    cands = generate_candidate_rules(
        features, groups, labels, max_preds=1, min_new_tp=1
    )
    assert len(cands) == 1
    assert cands[0].predicates == ["eq_a"]
    assert cands[0].train_precision == 1.0
    assert cands[0].train_true_positives == 2


def test_generate_candidates_prunes_below_min_tp():
    features = _make_features({"eq_a": [True, False, False]})
    labels = _bool_series([False, False, False])  # 0 TPs
    groups = [("a", ["eq_a"])]

    cands = generate_candidate_rules(
        features, groups, labels, max_preds=1, min_new_tp=1
    )
    assert cands == []


def test_generate_candidates_combines_across_groups():
    features = _make_features(
        {
            "eq_a": [True, True, False, True],
            "eq_b": [True, False, True, True],
        }
    )
    labels = _bool_series([True, True, False, False])
    groups = [("a", ["eq_a"]), ("b", ["eq_b"])]

    cands = generate_candidate_rules(
        features, groups, labels, max_preds=2, min_new_tp=1
    )
    rule_sets = {tuple(sorted(c.predicates)) for c in cands}
    assert ("eq_a",) in rule_sets
    assert ("eq_b",) in rule_sets
    assert ("eq_a", "eq_b") in rule_sets


def test_generate_candidates_does_not_combine_within_same_group():
    # Two thresholds in the same group → never AND'ed together (same group_id).
    features = _make_features(
        {
            "eq_a": [True, True, False],
            "dist_1_a": [True, True, True],
        }
    )
    labels = _bool_series([True, True, False])
    groups = [("a", ["eq_a", "dist_1_a"])]

    cands = generate_candidate_rules(
        features, groups, labels, max_preds=2, min_new_tp=1
    )
    # Only single-predicate rules; no rule combines eq_a AND dist_1_a.
    for c in cands:
        assert len(c.predicates) == 1


def test_generate_candidates_respects_max_preds():
    features = _make_features(
        {
            "eq_a": [True, True],
            "eq_b": [True, True],
            "eq_c": [True, True],
        }
    )
    labels = _bool_series([True, True])
    groups = [("a", ["eq_a"]), ("b", ["eq_b"]), ("c", ["eq_c"])]

    cands = generate_candidate_rules(
        features, groups, labels, max_preds=2, min_new_tp=1
    )
    for c in cands:
        assert len(c.predicates) <= 2


# --- find_composition ------------------------------------------------------------


def test_find_composition_picks_highest_precision_first():
    candidates = [
        Rule(["a"], _bool_series([True, True, False, False]), 1.0, 2),
        Rule(["b"], _bool_series([True, True, True, True]), 0.5, 2),
    ]
    labels = _bool_series([True, True, False, False])

    composition = find_composition(
        candidates,
        labels,
        min_recall=1.0,
        min_new_tp=1,
    )
    assert composition.rules[0].predicates == ["a"]
    assert composition.step_precision[0] == 1.0
    assert composition.step_recall[0] == 1.0


def test_find_composition_stops_when_recall_target_reached():
    # min_recall=0.5 → 1 TP suffices. Search should stop after one rule.
    candidates = [
        Rule(["a"], _bool_series([True, False, False, False]), 1.0, 1),
        Rule(["b"], _bool_series([False, True, False, False]), 1.0, 1),
    ]
    labels = _bool_series([True, True, False, False])
    composition = find_composition(
        candidates,
        labels,
        min_recall=0.5,
        min_new_tp=1,
    )
    assert len(composition.rules) == 1


def test_find_composition_no_candidates_returns_empty():
    labels = _bool_series([True, False])
    composition = find_composition(
        [],
        labels,
        min_recall=1.0,
        min_new_tp=1,
    )
    assert composition.rules == []
    assert (
        composition.step_precision
        == composition.step_recall
        == composition.step_true_positives
        == composition.step_positive_predictions
        == []
    )


def test_find_composition_ignores_candidates_with_zero_marginal_true_positives():
    # 'b' covers only what 'a' already covers → after picking 'a', 'b' is dead.
    candidates = [
        Rule(["a"], _bool_series([True, True, False]), 1.0, 2),
        Rule(["b"], _bool_series([True, True, False]), 1.0, 2),
    ]
    labels = _bool_series([True, True, False])
    composition = find_composition(
        candidates,
        labels,
        min_recall=1.0,
        min_new_tp=1,
    )
    assert len(composition.rules) == 1
