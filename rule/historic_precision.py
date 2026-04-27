import logging

import numpy as np
import polars as pl

LABELED_DOC2 = pl.col("HUKIMPORTTIME__2").ge(pl.datetime(2025, 10, 1)) & pl.col(
    "HUKIMPORTTIME__2"
).lt(pl.datetime(2025, 12, 25))


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
    historic_mask = data.select((~LABELED_DOC2).alias("m")).to_series().to_numpy()
    logging.info(f"historic_precision: {historic_mask.sum()} rows in unlabeled historic window")
    and_masks_hist = [m.to_numpy()[historic_mask] for m in and_masks]
    stratum = _first_hit_from_masks(and_masks_hist)

    data = (
        data.filter(pl.Series("_m", historic_mask))
        .with_columns(pl.Series("stratum_k", stratum))
        .filter(pl.col("stratum_k").ge(0))
    )

    all_strata = pl.DataFrame(
        {"stratum_k": np.arange(len(and_masks), dtype=np.int32)}
    )
    N_per_k = (
        data.group_by("stratum_k")
        .agg(pl.len().alias("N_k"))
        .join(all_strata, on="stratum_k", how="right")
        .with_columns(pl.col("N_k").fill_null(0))
    )
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


def labeled_cumulative_precision(and_masks, data, label_col):
    """Exact cumulative precision on the labeled subset (complement of HISTORIC_UNLABELED_FILTER).

    Returns DataFrame: m | N_lab_m | TP_lab_m | P_lab_m.
    Null labels are treated as False.
    """
    labeled_mask = data.select(LABELED_DOC2.alias("m")).to_series().to_numpy()
    labels = (
        data.select(pl.col(label_col).cast(pl.Boolean).fill_null(False))
        .to_series()
        .to_numpy()
    )
    labels_lab = labels[labeled_mask]
    logging.info(f"labeled_cumulative_precision: {labeled_mask.sum()} labeled rows")

    cum_or = np.zeros(labeled_mask.sum(), dtype=bool)
    rows = []
    for m, mask in enumerate(and_masks, start=1):
        m_arr = mask.to_numpy() if isinstance(mask, pl.Series) else mask
        cum_or = cum_or | m_arr[labeled_mask]
        n_lab = int(cum_or.sum())
        tp_lab = int((cum_or & labels_lab).sum())
        rows.append({
            "m": m,
            "N_lab_m": n_lab,
            "TP_lab_m": tp_lab,
            "P_lab_m": tp_lab / n_lab if n_lab > 0 else None,
        })
    return pl.DataFrame(rows)


def combined_precision(labeled_cum, historic_cum):
    """Combines exact labeled precision with estimated historic precision.

    P_combined_m = (N_lab_m · P_lab_m + N_hist_m · P_hat_hist_m) / (N_lab_m + N_hist_m)
    SE_combined_m = (N_hist_m / (N_lab_m + N_hist_m)) · SE_hat_hist_m

    Labeled counts are exact (zero variance); only the historic estimator contributes SE.
    If a stratum has no historic sample (SE_hat is null), we fall back to labeled-only
    for that m — i.e. P_combined = P_lab, SE = 0 — which is honest about what we know.
    """
    joined = labeled_cum.join(historic_cum, on="m", how="inner").with_columns(
        (pl.col("N_lab_m") + pl.col("N_1_to_m")).alias("N_total_m"),
    )
    joined = joined.with_columns(
        pl.when(pl.col("P_hat_m").is_not_null())
        .then(
            (
                pl.col("N_lab_m") * pl.col("P_lab_m")
                + pl.col("N_1_to_m") * pl.col("P_hat_m")
            )
            / pl.col("N_total_m")
        )
        .otherwise(pl.col("P_lab_m"))
        .alias("P_combined_m"),
        pl.when(pl.col("SE_m").is_not_null())
        .then((pl.col("N_1_to_m") / pl.col("N_total_m")) * pl.col("SE_m"))
        .otherwise(0.0)
        .alias("SE_combined_m"),
        pl.col("P_hat_m").is_not_null().alias("has_historic_evidence"),
    )
    return joined
