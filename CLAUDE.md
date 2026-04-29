# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this code does

This repo contains the rule-mining pipeline for detecting duplicate payments (Doppelzahlungen, DZ) between invoice pairs. The working code lives entirely under `rule/`.

A pair consists of two documents (Doc.1 and Doc.2) identified by (StackID, DocID, SubDocID). The pipeline learns OR-compositions of AND-rules over pre-computed boolean predicates (e.g. `eq_start_date`, `overlap_0.5_TAB_BETR_POSITION`) that maximize precision at a target recall.

## Key files

- `rule/rule_ml_flow.py` — Hydra + MLflow entry point. Reads `config.yaml`, loads data, runs Apriori → Greedy → post-processing → evaluation. Config path in the `@hydra.main` decorator points to `../configs/rules` — the YAML in this repo is a local copy for reference.
- `rule/evaluate_rule.py` — Evaluates greedy and final rule sets on two fixed time windows (`November_2025__October_2025` and `November_2025__2_years_historic`). Builds PR-value-landscape plots, LAR heatmaps, time-diff plots. `savings_analysis` joins reimbursement data from `CUR.VW_MF_PKL_DA_HANDLER_DATA` via a temp table in `DA00249.TEMP_DZ_SAVINGS`.
- `rule/historic_sampling.py` — Draws a stratified sample from the unlabeled historic window for manual labelling. Strata = index of the first greedy rule that fires.
- `rule/historic_precision.py` — After labels come back, estimates precision on the historic (unlabeled) window via the stratified sample, combines it with the exact labeled-window precision into a hits-weighted estimator with SE.
- `rule/config.yaml` — Data file, filter dates, learning hyperparameters.
- `rule/plotting.py` — HUK-colored matplotlib styling (imported into the main flow via `da_hf5_dz.configs.plotting.plotting`).

## Domain-specific concepts

**Labeled vs. historic window.** Labels only exist for pairs with Doc.2 in `[2025-10-01, 2025-12-25)`. For the 2-year historic window (`Doc.2 ∈ [2023-11-01, 2025-12-01)`) the remaining ~11M unlabeled rows have `null` labels that are cast to `False` — this is intentional, confirmed with the user.

**Strata.** A pair is assigned to stratum `k` = index of the first greedy rule in the sequence that fires. Strata are disjoint by construction. `N_k` = population size (deterministic), `n_k` = sample size (labels), `p̂_k = tp_k / n_k`, `se_k` = Bernoulli SE.

**Cumulative metrics over "first m rules as OR".** `N_1_to_m = Σ_{k<m} N_k`. `P_hat_m` is the hits-weighted average of `p̂_k` over sampled strata. The combined estimator blends the exact labeled-window precision with the historic estimate:
```
P_combined_m = (N_lab_m · P_lab_m + N_1_to_m · P_hat_m) / (N_lab_m + N_1_to_m)
SE_combined_m = (N_1_to_m / N_total_m) · SE_m
```
Only the historic term contributes variance; labeled counts are exact.

**`HISTORIC_UNLABELED_FILTER` lives in `historic_sampling.py`.** `LABELED_DOC2` in `historic_precision.py` is the intra-window complement (the 2-year filter is already applied upstream by `WINDOW_NOV_2YR` in `evaluate_rule.py`).

## Running the flow

The pipeline is run via Hydra from the `da-hf5-dz` project directory (not this repo directly) — `rule_ml_flow.py` imports from `da_hf5_dz.*` and expects its config at `../configs/rules/config.yaml`. Overrides use standard Hydra syntax, e.g. `learning.max_predicates_per_rule=3 positive_rate=0.01`.

MLflow runs are logged under the `DZ_test` experiment at `mlflow_tracking_uri` (from `da_hf5_dz.config`). Plots, rule lists, and the full `evaluation_dict` are logged as artifacts.

## Things that trip people up

- **`max_rules` can reach 1000.** Avoid deeply nested Polars expressions over rule lists — they blow the stack. Iterate with numpy (see `_first_hit_from_masks` in `historic_precision.py`).
- **Parquet is ~12M rows.** Working RAM on the run machine is ~180 GB, so eager loading is fine, but pass already-loaded DataFrames through helpers instead of re-reading parquet.
- **Greedy rules change between runs.** The historic sample was drawn against an older rule set; new runs will produce many strata with `n_k=0`. These are marked visually (blood-red dots) in the PR-value plot — they are *not* a bug.
- **`savings_analysis` writes only positives** (filtered with `cast(pl.Boolean).fill_null(False)`) to `DA00249.TEMP_DZ_SAVINGS`. The log line states the positive count, not total rows.
