from typing import Any, Dict, List
import pandas as pd
import polars as pl
import polars.selectors as cs
from .base import ImputationStrategy
from strata_fit_v6_imputation_py.utils import stack_results

class MeanImputer(ImputationStrategy):

    def compute(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        dfpl = pl.from_pandas(df)

        dfpl = dfpl.group_by("pat_ID").agg(
            cs.by_name(columns).mean(),
            pl.col("pat_ID").count().alias("n")
        )
        # df = df.groupby("pat_ID")[columns].mean().reset_index()
        return dfpl.to_pandas()
    
    def impute(self, df: pd.DataFrame, global_metric: Dict) -> pd.DataFrame:
        impute_vals = {col: list(v.values())[0] for col, v in global_metric.items()}
        df = df.fillna(impute_vals)
        return df
    
    def aggregate(self, node_metrics: List[Dict[Any, Any]], columns: List[str]) -> Any:
        dfpl = pl.from_pandas(stack_results(node_metrics))
        # simplest averaging
        global_weighted_mean = cs.by_name(columns).mul("n").sum().truediv(pl.col("n").sum())

        df = dfpl.with_columns(
            global_weighted_mean
        ).slice(0, 1).drop("pat_ID", "n").to_pandas()

        return df.to_dict()