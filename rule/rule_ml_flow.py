"""
Hydra + MLflow entry point for the rule-mining pipeline.

Builds an OR-composition of AND-rules over pre-computed boolean predicate
columns (e.g. eq_start_date, dist_2_end_date, overlap_0.5_TAB_BETR_POSITION),
optimised for net value on the historic window.

Pipeline:
  1. Load parquet, optionally downsample negatives, train/test split (rule_core.load_data)
  2. Generate candidate AND-rules bottom-up with anti-monotone pruning
     (rule_core.generate_candidate_rules)
  3. Find composition: greedily pick rules maximising marginal precision
     (rule_core.find_composition)
  4. Train/test evaluation for sanity logging (rule_core.evaluate)
  5. Historic-window evaluation: per-step net value via union-find on TP pairs,
     pick the economic optimum, post-process with remove_subsumed +
     remove_redundant on that sub-sequence (rule_evaluation.evaluate_composition)
  6. Log params, metrics, plots, rule sets to MLflow
"""

import logging
import sys

import hydra
import matplotlib.pyplot as plt
import mlflow
import polars as pl
from omegaconf import DictConfig, OmegaConf

from da_hf5_dz.config import (
    OLD_DZ_RULE,
    AggregationIdentifiers,
    filepath_shared_folder,
    mlflow_tracking_uri,
)
from rule_core import (
    build_groups_from_config,
    evaluate,
    find_composition,
    generate_candidate_rules,
    load_data,
)
from rule_evaluation import evaluate_composition

# --- Configuration ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    stream=sys.stdout,
)


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
        date_filter = (
            date_filter
            & pl.col("HUKIMPORTTIME").ge(
                pl.datetime(*[int(x) for x in train_start.split("-")])
            )
            & pl.col("HUKIMPORTTIME__2").ge(
                pl.datetime(*[int(x) for x in train_start.split("-")])
            )
        )
    if train_end := f.get("train_end"):
        date_filter = (
            date_filter
            & pl.col("HUKIMPORTTIME").lt(
                pl.datetime(*[int(x) for x in train_end.split("-")])
            )
            & pl.col("HUKIMPORTTIME__2").lt(
                pl.datetime(*[int(x) for x in train_end.split("-")])
            )
        )

    # Labelled rows (True or False) bypass the date filter; unlabelled rows
    # must fall inside the training window.
    train_filter = base_filter & (pl.col(label_col).is_not_null() | date_filter)

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
        "MIN_REIMBURSEMENT_THRESHOLD": cfg.get("min_reimbursement_threshold", 0),
        "ORIGIN_COL": cfg["origin"]["col"],
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
    MAX_PREDICATES_PER_RULE = built["MAX_PREDICATES_PER_RULE"]
    MAX_CANDIDATE_RULES = built["MAX_CANDIDATE_RULES"]
    MIN_RECALL = built["MIN_RECALL"]
    MIN_NEW_TP = built["MIN_NEW_TP"]
    REVIEW_COST = built["REVIEW_COST"]
    MIN_REIMBURSEMENT_THRESHOLD = built["MIN_REIMBURSEMENT_THRESHOLD"]

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("DZ_test")

    with mlflow.start_run(
        run_name=f"MR:{MIN_RECALL} PR:{POSITIVE_RATE} MP:{MAX_PREDICATES_PER_RULE}",
        description=f"FN:{FILE_NAME}",
    ):
        groups = build_groups_from_config(raw_cfg["predicates"])
        data = load_data(
            DATA_PATH,
            LABEL_COL,
            FILTER,
            ORIGIN_COL,
            groups,
            POSITIVE_RATE,
            TEST_SPLIT,
        )
        features, labels = data["features"], data["labels"]
        has_test: bool = "test_features" in data

        # Phase 1: Candidate rule generation
        logging.info(
            f"\n--- Phase 1: Candidate rule generation (max_preds={MAX_PREDICATES_PER_RULE}) ---"
        )
        candidates = generate_candidate_rules(
            features, groups, labels, MAX_PREDICATES_PER_RULE, MIN_NEW_TP
        )
        if len(candidates) > MAX_CANDIDATE_RULES:
            n_before = len(candidates)
            candidates = sorted(
                candidates, key=lambda r: -r.train_true_positives
            )[:MAX_CANDIDATE_RULES]
            logging.info(
                f"{n_before:,} candidates → {len(candidates):,} after TP cutoff"
            )
        else:
            logging.info(f"{len(candidates):,} candidates")

        # Phase 2: Find composition
        logging.info(f"\n--- Phase 2: Find composition (min_recall={MIN_RECALL}) ---")
        composition = find_composition(
            candidates,
            labels,
            MIN_RECALL,
            MIN_NEW_TP,
        )

        # Train evaluation (just for logs)
        logging.info("\n--- Train results ---")
        train_precision, train_recall = evaluate(composition.rules, features, labels)
        logging.info(
            f"Train: precision={train_precision:.4%}, recall={train_recall:.4%}"
        )
        if has_test:
            test_precision, test_recall = evaluate(
                composition.rules, data["test_features"], data["test_labels"]
            )
            logging.info(
                f"Test: precision={test_precision:.4%}, recall={test_recall:.4%}"
            )

        logging.info(f"\nGreedy composition ({len(composition.rules)} rules):")
        for i, rule in enumerate(composition.rules):
            logging.info(f"  Rule {i}: {' AND '.join(rule.predicates)}")

        # ------------ Historic-window evaluation ------------
        evaluation_result = evaluate_composition(
            composition_rules=composition.rules,
            groups=groups,
            data_path=DATA_PATH,
            base_filter=EVAL_FILTER,
            label_col=LABEL_COL,
            review_cost=REVIEW_COST,
            min_reimbursement_threshold=MIN_REIMBURSEMENT_THRESHOLD,
        )

        logging.info(
            f"\nOptimum composition ({len(evaluation_result.optimum_rules)} rules "
            f"at step {evaluation_result.optimum_step + 1}):"
        )
        for i, rule in enumerate(evaluation_result.optimum_rules):
            logging.info(f"  Rule {i}: {' AND '.join(rule.predicates)}")

        for plot_name, figure in evaluation_result.plots.items():
            mlflow.log_figure(figure, f"plots/{plot_name}.png")
        plt.close("all")

        mlflow.log_params(
            {
                "DATA_PATH": DATA_PATH,
                "FILTER": FILTER,
                "LABEL_COL": LABEL_COL,
                "POSITIVE_RATE": POSITIVE_RATE,
                "TEST_SPLIT": TEST_SPLIT,
                "FLATFILE_TYPE": FLATFILE_TYPE,
                "IDENTIFIERS": IDENTIFIERS,
                "ORIGIN_COL": ORIGIN_COL,
                "MAX_PREDICATES_PER_RULE": MAX_PREDICATES_PER_RULE,
                "MAX_CANDIDATE_RULES": MAX_CANDIDATE_RULES,
                "MIN_RECALL": MIN_RECALL,
                "MIN_NEW_TP": MIN_NEW_TP,
                "REVIEW_COST": REVIEW_COST,
                "MIN_REIMBURSEMENT_THRESHOLD": MIN_REIMBURSEMENT_THRESHOLD,
            }
        )
        mlflow.log_metrics(
            {
                "train_pair_count": float(len(labels)),
                "train_total_positives": float(int(labels.sum())),
                "train_pos_pred_pair_count": float(
                    composition.step_positive_predictions[-1]
                ),
                "train_true_positive_count": float(
                    composition.step_true_positives[-1]
                ),
                "train_precision": composition.step_precision[-1],
                "train_recall": composition.step_recall[-1],
                "greedy_rule_count": float(len(composition.rules)),
                "optimum_rule_count": float(len(evaluation_result.optimum_rules)),
                **evaluation_result.metrics,
            }
        )

        mlflow.log_dict(
            {
                "greedy_composition": [r.predicates for r in composition.rules],
                "optimum_composition": [
                    r.predicates for r in evaluation_result.optimum_rules
                ],
            },
            "rules.json",
        )


if __name__ == "__main__":
    main()
