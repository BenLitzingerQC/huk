import logging

import numpy as np
import polars as pl

from historic_sampling import HISTORIC_UNLABELED_FILTER


def _assign_first_hit(data, greedy_rules):
    first_hit = pl.lit(-1, dtype=pl.Int32)
    for k, rule in enumerate(greedy_rules):
        mask = pl.all_horizontal(
            [pl.col(c).cast(pl.Boolean).fill_null(False) for c in rule]
        )
        first_hit = pl.when((first_hit < 0) & mask).then(k).otherwise(first_hit)
    return data.with_columns(first_hit.alias("stratum_k"))


def estimate_historic_precision(greedy_rules, data_path, base_filter, label_col):
    """Returns DataFrame: stratum_k | N_k | n_k | tp_k | p_hat_k | se_k."""
    data = (
        pl.scan_parquet(data_path)
        .filter(base_filter & HISTORIC_UNLABELED_FILTER)
        .collect()
    )
    data = _assign_first_hit(data, greedy_rules).filter(pl.col("stratum_k").ge(0))

    N_per_k = data.group_by("stratum_k").agg(pl.len().alias("N_k"))
    labelled = data.filter(pl.col(label_col).is_not_null())

    per_k = (
        labelled.group_by("stratum_k")
        .agg(
            pl.len().alias("n_k"),
            pl.col(label_col).cast(pl.Boolean).sum().alias("tp_k"),
        )
        .join(N_per_k, on="stratum_k", how="right")
        .with_columns(pl.col("n_k").fill_null(0), pl.col("tp_k").fill_null(0))
        .with_columns(
            pl.when(pl.col("n_k") > 0)
            .then(pl.col("tp_k") / pl.col("n_k"))
            .alias("p_hat_k")
        )
        .with_columns(
            (pl.col("p_hat_k") * (1 - pl.col("p_hat_k")) / pl.col("n_k"))
            .sqrt()
            .alias("se_k")
        )
        .sort("stratum_k")
    )
    logging.info(f"historic_precision per stratum:\n{per_k}")
    return per_k


def cumulative_precision(per_k):
    """For each m in 1..K: P_hat_m = Σ w_i p_i, SE_m = sqrt(Σ w_i² se_i²)."""
    per_k = per_k.sort("stratum_k")
    rows = []
    for m in range(1, per_k.height + 1):
        head = per_k.head(m).filter(pl.col("n_k") > 0)
        if head.is_empty():
            rows.append({"m": m, "N_1_to_m": 0, "P_hat_m": None, "SE_m": None})
            continue
        N = head["N_k"].sum()
        w = (head["N_k"] / N).to_numpy()
        p = head["p_hat_k"].to_numpy()
        se = head["se_k"].to_numpy()
        rows.append({
            "m": m,
            "N_1_to_m": int(per_k.head(m)["N_k"].sum()),
            "P_hat_m": float(np.sum(w * p)),
            "SE_m": float(np.sqrt(np.sum(w**2 * se**2))),
        })
    return pl.DataFrame(rows)
