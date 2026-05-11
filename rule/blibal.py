def build_components(pairs_df: pl.DataFrame) -> pl.DataFrame:
      """
      One row per connected component with its docs and per-doc stats packed in lists,
      plus the n-1-cheapest aggregate (we keep the most expensive doc per component).

      Returns a DataFrame with columns:
        component_id          Struct[StackID, DocID, SubDocID]   -- root doc of the component
        doc_ids               List[Struct]
        reimbursements        List[Float64]                       -- aligned with doc_ids
        paid_outs             List[Boolean]                       -- aligned with doc_ids
        total_reimbursement   Float64                             -- sum of n-1 cheapest
        total_paid_out        Float64                             -- sum of n-1 cheapest where paid_out
      """
      union_find = _UnionFind()
      for row in pairs_df.iter_rows(named=False):
          doc_a = row[: len(DOC_ID_COLUMNS)]
          doc_b = row[len(DOC_ID_COLUMNS) :]
          union_find.union(doc_a, doc_b)

      left = pairs_df.select(
          pl.struct(DOC_ID_COLUMNS).alias("doc_id"),
          pl.col("MAX_REIMBURSEMENT").cast(pl.Float64).fill_null(0.0).alias("reimbursement"),
          pl.col("PAID_OUT").cast(pl.Boolean).fill_null(False).alias("paid_out"),
      )
      right = pairs_df.select(
          pl.struct(DOC2_ID_COLUMNS).alias("doc_id"),
          pl.col("MAX_REIMBURSEMENT__2").cast(pl.Float64).fill_null(0.0).alias("reimbursement"),
          pl.col("PAID_OUT__2").cast(pl.Boolean).fill_null(False).alias("paid_out"),
      )
      docs = pl.concat([left, right]).unique(subset=["doc_id"])

      def _root_struct(doc_struct: dict) -> dict:
          root_tuple = union_find.find(
              (doc_struct["StackID"], doc_struct["DocID"], doc_struct["SubDocID"])
          )
          return {
              "StackID": root_tuple[0],
              "DocID": root_tuple[1],
              "SubDocID": root_tuple[2],
          }

      docs = docs.with_columns(
          pl.col("doc_id")
          .map_elements(
              _root_struct,
              return_dtype=pl.Struct({
                  "StackID": pl.String,
                  "DocID": pl.String,
                  "SubDocID": pl.String,
              }),
          )
          .alias("component_id")
      )

      # Sort by reimbursement DESC inside each component, so we can drop the head
      # (the most expensive doc) and aggregate the tail = the n-1 cheapest.
      docs = docs.sort(["component_id", "reimbursement"], descending=[False, True])

      return docs.group_by("component_id").agg(
          pl.col("doc_id").alias("doc_ids"),
          pl.col("reimbursement").alias("reimbursements"),
          pl.col("paid_out").alias("paid_outs"),
          pl.col("reimbursement").slice(1).sum().alias("total_reimbursement"),
          pl.col("reimbursement")
              .slice(1)
              .filter(pl.col("paid_out").slice(1))
              .sum()
              .alias("total_paid_out"),
      )
