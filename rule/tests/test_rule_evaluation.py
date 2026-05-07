import polars as pl
import pytest

from rule_evaluation import (
    _build_doc_lookups,
    compute_component_savings,
    compute_stepwise_net_value,
)


def _pairs(rows):
    """Build a pair DataFrame from (a, b) tuples where a, b are doc-id triples."""
    return pl.DataFrame(
        rows,
        schema=[
            "StackID",
            "DocID",
            "SubDocID",
            "StackID__2",
            "DocID__2",
            "SubDocID__2",
        ],
        orient="row",
    )


def _enriched_pairs(rows):
    """
    Build an enriched pair DataFrame including reimbursements and paid_out flags.

    Each row: (doc_a_triple, doc_b_triple, reimb_a, reimb_b, paid_a, paid_b).
    """
    expanded = [
        (*a, *b, reimb_a, reimb_b, paid_a, paid_b)
        for (a, b, reimb_a, reimb_b, paid_a, paid_b) in rows
    ]
    return pl.DataFrame(
        expanded,
        schema=[
            "StackID",
            "DocID",
            "SubDocID",
            "StackID__2",
            "DocID__2",
            "SubDocID__2",
            "MAX_REIMBURSEMENT",
            "MAX_REIMBURSEMENT__2",
            "PAID_OUT",
            "PAID_OUT__2",
        ],
        orient="row",
    )


def test_savings_empty_pairs():
    pairs = _pairs([])
    total, paid = compute_component_savings(pairs, {}, {})
    assert total == 0.0
    assert paid == 0.0


def test_savings_single_pair_rejects_cheaper_doc():
    # Two-doc component: keep doc with reimbursement 100, reject doc with 30.
    a = ("S1", "D1", "1")
    b = ("S2", "D2", "1")
    pairs = _pairs([(*a, *b)])
    reimbursements = {a: 30.0, b: 100.0}
    paid_out = {a: True, b: False}
    total, paid = compute_component_savings(pairs, reimbursements, paid_out)
    assert total == 30.0
    assert paid == 30.0  # rejected doc was paid out


def test_savings_paid_out_subset_excludes_unpaid_rejected():
    # Reject the cheaper doc (50), but it was NOT paid out → total counts it,
    # paid-out subset does not.
    a = ("S1", "D1", "1")
    b = ("S2", "D2", "1")
    pairs = _pairs([(*a, *b)])
    reimbursements = {a: 50.0, b: 200.0}
    paid_out = {a: False, b: True}
    total, paid = compute_component_savings(pairs, reimbursements, paid_out)
    assert total == 50.0
    assert paid == 0.0


def test_savings_chain_of_three_docs_keeps_most_expensive():
    # a—b—c: one connected component of size 3. Reimbursements 10, 20, 30.
    # Reject the two cheaper (10 + 20 = 30); keep the 30.
    a, b, c = ("S", "A", "1"), ("S", "B", "1"), ("S", "C", "1")
    pairs = _pairs([(*a, *b), (*b, *c)])
    reimbursements = {a: 10.0, b: 20.0, c: 30.0}
    paid_out = {a: True, b: True, c: True}
    total, paid = compute_component_savings(pairs, reimbursements, paid_out)
    assert total == 30.0
    assert paid == 30.0


def test_savings_branching_component_six_pairs_three_docs():
    # 4 docs fully connected: 6 pairs, 1 component, reject 3 cheapest.
    # Reimbursements 5, 10, 20, 100 → reject 5+10+20 = 35.
    a, b, c, d = ("S", "A", "1"), ("S", "B", "1"), ("S", "C", "1"), ("S", "D", "1")
    pairs = _pairs(
        [
            (*a, *b),
            (*a, *c),
            (*a, *d),
            (*b, *c),
            (*b, *d),
            (*c, *d),
        ]
    )
    reimbursements = {a: 5.0, b: 10.0, c: 20.0, d: 100.0}
    paid_out = {a: False, b: True, c: True, d: True}
    total, paid = compute_component_savings(pairs, reimbursements, paid_out)
    assert total == pytest.approx(35.0)
    # Paid-out among rejected: b (10) and c (20). a was NOT paid.
    assert paid == pytest.approx(30.0)


def test_savings_two_independent_components_summed():
    a, b = ("S", "A", "1"), ("S", "B", "1")
    c, d = ("S", "C", "1"), ("S", "D", "1")
    pairs = _pairs([(*a, *b), (*c, *d)])
    reimbursements = {a: 10.0, b: 50.0, c: 30.0, d: 200.0}
    paid_out = {a: True, b: True, c: True, d: True}
    total, paid = compute_component_savings(pairs, reimbursements, paid_out)
    assert total == 10.0 + 30.0
    assert paid == 10.0 + 30.0


def test_savings_missing_reimbursement_treated_as_zero():
    a, b = ("S", "A", "1"), ("S", "B", "1")
    pairs = _pairs([(*a, *b)])
    # b not in reimbursement dict → treated as 0; cheaper of (50, 0) is b.
    total, paid = compute_component_savings(pairs, {a: 50.0}, {a: True})
    assert total == 0.0
    assert paid == 0.0


def test_savings_self_loop_pair_is_no_op():
    # Pair where both sides reference the same doc → 1-doc component → skipped.
    a = ("S", "A", "1")
    pairs = _pairs([(*a, *a)])
    total, paid = compute_component_savings(pairs, {a: 99.0}, {a: True})
    assert total == 0.0
    assert paid == 0.0


# --- _build_doc_lookups: data-quality edge cases --------------------------------


def test_build_doc_lookups_extracts_both_sides():
    a, b = ("S", "A", "1"), ("S", "B", "1")
    pairs = _enriched_pairs([(a, b, 10.0, 20.0, True, False)])
    reimbursements, paid_out = _build_doc_lookups(pairs)
    assert reimbursements == {a: 10.0, b: 20.0}
    assert paid_out == {a: True, b: False}


def test_build_doc_lookups_null_reimbursement_becomes_zero():
    a, b = ("S", "A", "1"), ("S", "B", "1")
    pairs = _enriched_pairs([(a, b, None, 50.0, True, True)])
    reimbursements, _ = _build_doc_lookups(pairs)
    assert reimbursements[a] == 0.0
    assert reimbursements[b] == 50.0


def test_build_doc_lookups_null_paid_out_becomes_false():
    a, b = ("S", "A", "1"), ("S", "B", "1")
    pairs = _enriched_pairs([(a, b, 10.0, 20.0, None, None)])
    _, paid_out = _build_doc_lookups(pairs)
    assert paid_out[a] is False
    assert paid_out[b] is False


def test_build_doc_lookups_empty_pairs():
    pairs = _enriched_pairs([])
    reimbursements, paid_out = _build_doc_lookups(pairs)
    assert reimbursements == {}
    assert paid_out == {}


def test_build_doc_lookups_doc_appearing_twice_is_deduplicated():
    # Doc 'a' appears as Doc.1 in pair-1 and as Doc.2 in pair-2 — it should
    # produce a single entry in the lookup, not duplicate.
    a, b, c = ("S", "A", "1"), ("S", "B", "1"), ("S", "C", "1")
    pairs = _enriched_pairs(
        [
            (a, b, 10.0, 20.0, True, True),
            (c, a, 30.0, 10.0, False, True),
        ]
    )
    reimbursements, _ = _build_doc_lookups(pairs)
    assert reimbursements[a] == 10.0
    assert len(reimbursements) == 3  # a, b, c — no duplicate of a


# --- compute_stepwise_net_value: integration over the full loop ------------------


def test_stepwise_net_value_two_rules_two_pairs():
    # Two pairs, two rules. Pair 1 caught by rule 0, pair 2 caught by rule 1.
    # Both pairs are TPs. After rule 0: 1 pair flagged, 1 TP, savings = cheaper
    # of pair 1. After rule 1: 2 pairs flagged, 2 TPs, two independent components.
    a, b = ("S", "A", "1"), ("S", "B", "1")
    c, d = ("S", "C", "1"), ("S", "D", "1")
    pairs = _enriched_pairs(
        [
            (a, b, 10.0, 100.0, True, True),
            (c, d, 30.0, 200.0, True, True),
        ]
    )
    labels = pl.Series("", [True, True], dtype=pl.Boolean)
    rule_masks = [
        pl.Series("", [True, False], dtype=pl.Boolean),
        pl.Series("", [False, True], dtype=pl.Boolean),
    ]

    result = compute_stepwise_net_value(pairs, rule_masks, labels, review_cost=2.0)

    assert result.pos_pred_pair_count == [1, 2]
    assert result.true_positive_pair_count == [1, 2]
    assert result.total_savings == [10.0, 40.0]
    assert result.paid_out_savings == [10.0, 40.0]
    # NV = savings - n_pos_pred * review_cost
    assert result.net_value_total == [10.0 - 2.0, 40.0 - 4.0]
    assert result.net_value_paid_out == [10.0 - 2.0, 40.0 - 4.0]


def test_stepwise_net_value_false_positive_pair_costs_review_but_no_savings():
    # One pair is FP (label=False) → counts toward pos_pred but not toward TP
    # union-find. Savings should stay 0 while review_cost is paid.
    a, b = ("S", "A", "1"), ("S", "B", "1")
    pairs = _enriched_pairs([(a, b, 100.0, 200.0, True, True)])
    labels = pl.Series("", [False], dtype=pl.Boolean)
    rule_masks = [pl.Series("", [True], dtype=pl.Boolean)]

    result = compute_stepwise_net_value(pairs, rule_masks, labels, review_cost=5.0)

    assert result.pos_pred_pair_count == [1]
    assert result.true_positive_pair_count == [0]
    assert result.total_savings == [0.0]
    assert result.net_value_total == [-5.0]


def test_stepwise_net_value_paid_out_diverges_from_total():
    # All TPs, but cheapest doc was NOT paid out → it counts toward total
    # savings but not toward paid_out_savings.
    a, b = ("S", "A", "1"), ("S", "B", "1")
    pairs = _enriched_pairs([(a, b, 10.0, 100.0, False, True)])
    labels = pl.Series("", [True], dtype=pl.Boolean)
    rule_masks = [pl.Series("", [True], dtype=pl.Boolean)]

    result = compute_stepwise_net_value(pairs, rule_masks, labels, review_cost=1.0)

    assert result.total_savings == [10.0]
    assert result.paid_out_savings == [0.0]
    assert result.net_value_total == [9.0]
    assert result.net_value_paid_out == [-1.0]


def test_stepwise_net_value_handles_null_reimbursement_gracefully():
    # Doc 'a' has NULL reimbursement → treated as 0; doc 'b' has 50.
    # Cheapest is 'a' (0), reject 'a', savings = 0 (it's the cheaper one rejected).
    a, b = ("S", "A", "1"), ("S", "B", "1")
    pairs = _enriched_pairs([(a, b, None, 50.0, True, True)])
    labels = pl.Series("", [True], dtype=pl.Boolean)
    rule_masks = [pl.Series("", [True], dtype=pl.Boolean)]

    result = compute_stepwise_net_value(pairs, rule_masks, labels, review_cost=1.0)

    assert result.total_savings == [0.0]
    assert result.net_value_total == [-1.0]


def test_stepwise_net_value_components_grow_across_rules():
    # 3 pairs forming a chain a—b—c—d. After rule 0: only (a,b) → reject 1 doc.
    # After rule 1: (a,b) + (b,c) → component {a,b,c} → reject 2 docs.
    # After rule 2: full chain → reject 3 docs.
    a, b, c, d = ("S", "A", "1"), ("S", "B", "1"), ("S", "C", "1"), ("S", "D", "1")
    pairs = _enriched_pairs(
        [
            (a, b, 10.0, 20.0, True, True),
            (b, c, 20.0, 30.0, True, True),
            (c, d, 30.0, 40.0, True, True),
        ]
    )
    labels = pl.Series("", [True, True, True], dtype=pl.Boolean)
    rule_masks = [
        pl.Series("", [True, False, False], dtype=pl.Boolean),
        pl.Series("", [False, True, False], dtype=pl.Boolean),
        pl.Series("", [False, False, True], dtype=pl.Boolean),
    ]

    result = compute_stepwise_net_value(pairs, rule_masks, labels, review_cost=0.0)

    # Step 1: {a,b}, reject cheapest (a=10) → 10
    # Step 2: {a,b,c}, reject 2 cheapest (a=10, b=20) → 30
    # Step 3: {a,b,c,d}, reject 3 cheapest (a, b, c) → 60
    assert result.total_savings == [10.0, 30.0, 60.0]
    assert result.true_positive_pair_count == [1, 2, 3]
