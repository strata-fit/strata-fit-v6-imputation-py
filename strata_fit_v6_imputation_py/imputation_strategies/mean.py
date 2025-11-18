from typing import Any, Dict, List
import pandas as pd
import polars as pl
import polars.selectors as cs
from .base import ImputationStrategy

class MeanImputer(ImputationStrategy):

    def compute(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        dfpl = pl.from_pandas(df)

        dfpl = dfpl.group_by("pat_ID").agg(
            cs.by_name(columns).mean(),
            pl.col("pat_ID").count().alias("n")
        )
        # df = df.groupby("pat_ID")[columns].mean().reset_index()
        return dfpl.to_pandas()
    
    def impute(self, df: pd.DataFrame, global_metric: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index("pat_ID")
        global_metric = global_metric.set_index("pat_ID")

        df = df.combine_first(global_metric).reset_index()

        return df
    
    def aggregate(self, results: List[Dict[Any, Any]], columns: List[str]) -> Any:
        
        dfs = []
        for result in results:
            dfs.append(pl.DataFrame({col: list(inner_dict.values()) for col, inner_dict in result.items()}))

        dfpl = pl.concat(dfs, how="vertical")

        df = dfpl.group_by("pat_ID").agg(
            cs.by_name(columns).mul("n").sum().truediv(pl.col("n").sum())
        ).to_pandas()

        return df.to_dict()