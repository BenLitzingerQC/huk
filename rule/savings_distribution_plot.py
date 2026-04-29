import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import polars as pl
from da_hf5_utils.db2 import get_engine
from sqlalchemy import text
from sqlalchemy.types import VARCHAR

from da_hf5_dz.configs.plotting.plotting import setup_plotting

setup_plotting()

conn = get_engine(database="spielwiese")
raw_ids = ["StackID", "DocID", "SubDocID", "StackID__2", "DocID__2", "SubDocID__2"]
data_path = "/domino/edv/pvc-hf5health/outpatient/dz/labelling/full_data_sets/25_12_25_2y_historic_all_labels_without_unclear__27_04_2026_.parquet"

with conn.begin() as c:
    try:
        c.execute(text("DROP TABLE DA00249.TEMP_DZ_LORENZ"))
    except:
        pass

df = (
    pl.scan_parquet(data_path)
    .filter(pl.col("dz_interesting"))
    .select(raw_ids + ["HUKIMPORTTIME", "HUKIMPORTTIME__2"])
    .collect()
)

df.write_database(
    "DA00249.TEMP_DZ_LORENZ",
    connection=conn,
    engine_options={"dtype": {c: VARCHAR(50) for c in raw_ids}},
)

q = """
SELECT LEAST(h1.MAX_REIMBURSEMENT, h2.MAX_REIMBURSEMENT) AS S, HUKIMPORTTIME, HUKIMPORTTIME__2
FROM DA00249.TEMP_DZ_LORENZ t
LEFT JOIN CUR.VW_MF_PKL_DA_HANDLER_DATA h1
  ON t."StackID"=h1.STACKID AND t."DocID"=h1.DOCID AND t."SubDocID"=h1.SUBDOCID
LEFT JOIN CUR.VW_MF_PKL_DA_HANDLER_DATA h2
  ON t."StackID__2"=h2.STACKID AND t."DocID__2"=h2.DOCID AND t."SubDocID__2"=h2.SUBDOCID
"""
result = pl.read_database(q, connection=conn)
s_col = next(c for c in result.columns if c.upper() == "S")
s_all = result[s_col].drop_nulls().to_numpy().astype(float)
s = np.sort(s_all)[::-1]

n_total = len(s)
total_reimbursement = s.sum()
average_reimbursement = total_reimbursement/n_total

x = np.arange(1, n_total + 1) / n_total * 100
y = np.cumsum(s) / total_reimbursement * 100

#%%
fig, ax_lorenz = plt.subplots(figsize=(7, 6), dpi=300)

# --- Lorenz-style cumulative savings curve ---
ax_lorenz.plot(x, y, color="#333333", linewidth=1.4)
ax_lorenz.set_xlabel("% der Paare (absteigend nach Savings sortiert)")
ax_lorenz.set_ylabel("% der kumulierten Savings")
ax_lorenz.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax_lorenz.spines[["top", "right"]].set_visible(False)
ax_lorenz.grid(alpha=0.3)

# # --- Histogram ---
# ax_hist.hist(s_all, bins=50, range=(0, 1000), color="#333333", edgecolor="white", linewidth=0.4)
# ax_hist.set_xlabel("min(Erstattungsbeträge) in EUR")
# ax_hist.set_ylabel("Anzahl DZ-Paare")
# ax_hist.xaxis.set_major_formatter(
#     plt.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", "."))
# )
# ax_hist.spines[["top", "right"]].set_visible(False)

# --- Shared title & subtitle ---
fig.suptitle(
    "Savings-Verteilung der gelabelten DZ-Positiven",
    fontsize=13,
    fontweight="bold",
    x=0.0,
    ha="left",
)
fmt = lambda x: f"{x:,.2f}".translate(str.maketrans(",.", ".,"))
fig.text(
    x=0.0,
    y=0.93,
    s=f"Durchschnitt={fmt(total_reimbursement/n_total)} EUR, Median={fmt(np.median(s))} EUR",
    fontsize=10,
    ha="left",
    va="top",
)

fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
plt.show()


# %%
time_diff = (
    result
    .with_columns(
        (pl.col("HUKIMPORTTIME") - pl.col("HUKIMPORTTIME__2")).dt.total_days().mul(1/7).alias("weeks")
    )
    .select(pl.col("weeks"))
    .drop_nulls()
    .to_series()
    .to_numpy()
)

n_total_time = len(time_diff)
mean_days = np.mean(time_diff)
median_days = np.median(time_diff)
valid_diff = time_diff[(time_diff > 0) & (time_diff <= 2*52)]

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

ax.hist(valid_diff, bins=int(2*52), color="#333333", edgecolor="white", linewidth=0.4)
ax.set_xlabel("Zeitdifferenz in Wochen (Doc 1 − Doc 2)")
ax.set_ylabel("Anzahl DZ-Paare")
ax.set_xlim(left=0, right=2*52)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{int(x)}")
)

fig.suptitle(
    "Zeitdifferenz-Verteilung bei DZ-positiven Paaren",
    fontsize=13,
    fontweight="bold",
    x=0.0,
    ha="left",
)
fig.text(
    x=0.0,
    y=0.93,
    s=f"Durchschnitt={mean_days:.1f} Tage, Median={median_days:.1f} Tage",
    fontsize=10,
    ha="left",
    va="top",
)

fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
plt.show()
# %%
