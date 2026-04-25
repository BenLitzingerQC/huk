import logging

import numpy as np
import polars as pl

from historic_sampling import HISTORIC_UNLABELED_FILTER


def _first_hit_from_masks(and_masks):
    """stratum_k = index of first True mask per row, -1 if none. Operates on numpy."""
    stratum = np.full(len(and_masks[0]), -1, dtype=np.int32)
    for k, mask in enumerate(and_masks):
        m = mask.to_numpy() if isinstance(mask, pl.Series) else mask
        stratum = np.where((stratum < 0) & m, k, stratum)
    return stratum


def estimate_historic_precision(and_masks, data, label_col):
    """Returns DataFrame: stratum_k | N_k | n_k | tp_k | p_hat_k | se_k.

    `data` and each mask in `and_masks` must be aligned (same length).
    Restricts to HISTORIC_UNLABELED_FILTER rows before stratifying.
    """
    historic_mask = (
        data.select(HISTORIC_UNLABELED_FILTER.alias("m")).to_series().to_numpy()
    )
    logging.info(f"historic_precision: {historic_mask.sum()} rows in unlabeled historic window")
    and_masks_hist = [m.to_numpy()[historic_mask] for m in and_masks]
    stratum = _first_hit_from_masks(and_masks_hist)

    data = (
        data.filter(pl.Series("_m", historic_mask))
        .with_columns(pl.Series("stratum_k", stratum))
        .filter(pl.col("stratum_k").ge(0))
    )

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
