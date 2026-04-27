import numpy as np
import polars as pl
from sqlalchemy import text
from sqlalchemy.types import VARCHAR
from da_hf5_utils.db2 import get_engine

setup_plotting()

conn = get_engine(database="spielwiese")
raw_ids = ["StackID", "DocID", "SubDocID", "StackID__2", "DocID__2", "SubDocID__2"]

with conn.begin() as c:
    try: c.execute(text("DROP TABLE DA00249.TEMP_DZ_LORENZ"))
    except: pass
df.select(raw_ids).write_database(
    "DA00249.TEMP_DZ_LORENZ", connection=conn,
    engine_options={"dtype": {c: VARCHAR(50) for c in raw_ids}},
)

q = """
SELECT LEAST(h1.MAX_REIMBURSEMENT, h2.MAX_REIMBURSEMENT) AS s
FROM DA00249.TEMP_DZ_LORENZ t
LEFT JOIN CUR.VW_MF_PKL_DA_HANDLER_DATA h1
  ON t."StackID"=h1.STACKID AND t."DocID"=h1.DOCID AND t."SubDocID"=h1.SUBDOCID
LEFT JOIN CUR.VW_MF_PKL_DA_HANDLER_DATA h2
  ON t."StackID__2"=h2.STACKID AND t."DocID__2"=h2.DOCID AND t."SubDocID__2"=h2.SUBDOCID
"""
s_col = next(iter(pl.read_database(q, connection=conn).columns))
s = pl.read_database(q, connection=conn)[s_col].drop_nulls().to_numpy()
s = np.sort(s.astype(float))[::-1]

x = np.arange(1, len(s) + 1) / len(s) * 100
y = np.cumsum(s) / s.sum() * 100

plt.plot(x, y)
plt.xlabel("% der Paare (sortiert nach Savings absteigend)")
plt.ylabel("% der kumulierten Savings")
plt.grid(alpha=0.3)
plt.show()
