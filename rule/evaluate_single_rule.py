"""
Stand-alone entry point: evaluate a manually defined composition (list[list[str]])
on the historic window using the production config (filter, paths, predicate groups).

Edit the `RULES` list below, then run via Hydra:
    python evaluate_single_rule.py
"""

import logging
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from rule_evaluation import evaluate_predicate_lists
from rule_core import build_groups_from_config
from rule_ml_flow import build_runtime_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)

OUTPUT_DIR = Path("single_rule_evaluation")

RULES: list[list[str]] = [
    # paste your composition here
]


@hydra.main(version_base=None, config_path="../configs/rules", config_name="config")
def main(cfg: DictConfig):
    raw_cfg = OmegaConf.to_container(cfg, resolve=True)
    built = build_runtime_config(raw_cfg)
    groups = build_groups_from_config(raw_cfg["predicates"])

    result = evaluate_predicate_lists(
        predicate_lists=RULES,
        groups=groups,
        data_path=built["DATA_PATH"],
        base_filter=built["EVAL_FILTER"],
        label_col=built["LABEL_COL"],
        review_cost=built["REVIEW_COST"],
        min_reimbursement_threshold=built["MIN_REIMBURSEMENT_THRESHOLD"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for plot_name, figure in result.plots.items():
        figure.savefig(OUTPUT_DIR / f"{plot_name}.png", bbox_inches="tight")
    plt.close("all")

    logging.info("\n=== Metrics ===")
    for name, value in result.metrics.items():
        logging.info(f"  {name}: {value}")
    logging.info(
        f"\nOptimum composition ({len(result.optimum_rules)} rules at step "
        f"{result.optimum_step + 1})"
    )


if __name__ == "__main__":
    main()
