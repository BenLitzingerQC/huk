import logging
import sys
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    stream=sys.stdout,
)

# Historic window: Doc.1 in November 2025, Doc.2 in [2023-11-01, 2025-12-01)
# but excluding pairs where Doc.2 is already labelled (01.10.2025–25.12.2025)
HISTORIC_UNLABELED_FILTER = (
    pl.col("HUKIMPORTTIME").dt.year().eq(2025)
    & pl.col("HUKIMPORTTIME").dt.month().eq(11)
    & pl.col("HUKIMPORTTIME__2").ge(pl.datetime(2023, 11, 1))
    & pl.col("HUKIMPORTTIME__2").lt(pl.datetime(2025, 12, 1))
    & ~(
        pl.col("HUKIMPORTTIME__2").ge(pl.datetime(2025, 10, 1))
        & pl.col("HUKIMPORTTIME__2").lt(pl.datetime(2025, 12, 25))
    )
)


def sample_historic_unlabeled(
    greedy_rules: list[list[str]],
    data_path: Path,
    base_filter: pl.Expr,
    total_sample_size: int,
    output_path: Path,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Draws a stratified sample from the unlabeled historic window for manual labelling.

    Strata are defined by first_hit_idx — the index of the first greedy rule
    that fires for a pair. Pairs with no hit are discarded.

    Parameters
    ----------
    greedy_rules      : list of AND-rules (each rule = list of predicate column names)
    data_path         : path to the feature parquet
    base_filter       : base Polars filter expression (EVAL_FILTER from build_runtime_config)
    total_sample_size : desired total sample size (proportionally allocated across strata)
    output_path       : where to write the sampled parquet
    seed              : random seed for reproducibility

    Returns
    -------
    Polars DataFrame with all original columns plus stratum_k, N_k, n_k.
    """
    logging.info("historic_sampling: loading data")
    data = pl.read_parquet(data_path).filter(base_filter & HISTORIC_UNLABELED_FILTER)
    logging.info(f"historic_sampling: {data.height} rows after filters")

    # --- Compute first_hit_idx per row ---
    rule_masks = []
    for rule in greedy_rules:
        mask = pl.lit(True)
        for col in rule:
            mask = mask & pl.col(col).cast(pl.Boolean).fill_null(False)
        rule_masks.append(mask)

    # Add a column per rule, then derive first_hit_idx
    rule_col_names = [f"_rule_{k}" for k in range(len(greedy_rules))]
    data = data.with_columns(
        [mask.alias(name) for mask, name in zip(rule_masks, rule_col_names)]
    )

    # first_hit_idx: index of first True rule column, or -1
    data = data.with_columns(
        pl.concat_list(rule_col_names)
        .list.arg_max()  # index of first True (arg_max on booleans)
        .alias("_first_hit_raw")
    ).with_columns(
        pl.when(pl.concat_list(rule_col_names).list.any())
        .then(pl.col("_first_hit_raw"))
        .otherwise(pl.lit(-1))
        .cast(pl.Int32)
        .alias("stratum_k")
    ).drop(rule_col_names + ["_first_hit_raw"])

    # Discard pairs with no hit
    data = data.filter(pl.col("stratum_k").ge(0))
    logging.info(f"historic_sampling: {data.height} rows with at least one rule hit")

    # --- Stratum sizes ---
    stratum_counts = (
        data.group_by("stratum_k")
        .agg(pl.len().alias("N_k"))
        .sort("stratum_k")
    )
    total_hits = stratum_counts["N_k"].sum()
    logging.info(f"historic_sampling: stratum sizes (N_k):")
    for row in stratum_counts.iter_rows(named=True):
        logging.info(f"  stratum {row['stratum_k']}: N_k={row['N_k']}")

    # --- Proportional allocation ---
    stratum_counts = stratum_counts.with_columns(
        (pl.col("N_k") / total_hits * total_sample_size)
        .round()
        .cast(pl.Int32)
        .clip(lower_bound=1)
        .alias("n_k_raw")
    ).with_columns(
        pl.min_horizontal("N_k", "n_k_raw").alias("n_k")
    ).drop("n_k_raw")

    logging.info(f"historic_sampling: sample allocation (n_k):")
    for row in stratum_counts.iter_rows(named=True):
        logging.info(f"  stratum {row['stratum_k']}: n_k={row['n_k']} / N_k={row['N_k']}")

    # --- Attach N_k and n_k, then sample per stratum ---
    data = data.join(stratum_counts, on="stratum_k", how="left")

    sampled_frames = []
    for row in stratum_counts.iter_rows(named=True):
        k, n_k = row["stratum_k"], row["n_k"]
        stratum_df = data.filter(pl.col("stratum_k").eq(k))
        sampled_frames.append(
            stratum_df.sample(n=n_k, seed=seed, shuffle=True)
        )

    result = pl.concat(sampled_frames).sort("stratum_k")
    logging.info(f"historic_sampling: total sampled rows={result.height}")

    result.write_parquet(output_path)
    logging.info(f"historic_sampling: written to {output_path}")

    return result
