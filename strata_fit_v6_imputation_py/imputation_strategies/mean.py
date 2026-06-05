from typing import Any, Dict, List

import pandas as pd

from .base import ImputationStrategy, ImputationStrategyEnum, register_imputation_strategy
from strata_fit_v6_imputation_py.utils import stack_results


@register_imputation_strategy(ImputationStrategyEnum.MEAN_IMPUTER)
class MeanImputer(ImputationStrategy):
    def compute(
        self,
        df: pd.DataFrame,
        columns: List[str],
        global_state: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        del global_state
        available_columns = [column for column in columns if column in df.columns]
        if not available_columns:
            return {"n": {0: 0}}

        if "pat_ID" in df.columns:
            grouped = df.groupby("pat_ID", dropna=False)[available_columns].mean()
            counts = df.groupby("pat_ID", dropna=False).size().rename("n")
            result = grouped.join(counts).reset_index()
        else:
            payload = {column: [float(df[column].mean())] for column in available_columns}
            payload["n"] = [int(len(df.index))]
            result = pd.DataFrame(payload)

        return result.to_dict()

    def impute(self, df: pd.DataFrame, global_metric: Dict) -> Dict[str, Any]:
        impute_vals = {}
        for column, values in global_metric.items():
            if isinstance(values, dict) and values:
                impute_vals[column] = float(next(iter(values.values())))
        return df.fillna(impute_vals).to_dict()

    def aggregate(
        self,
        node_metrics: List[Dict[Any, Any]],
        columns: List[str],
        global_means: Dict[str, Any] | None = None,
    ) -> Dict:
        del global_means
        stacked = stack_results(node_metrics)
        if stacked.empty:
            return {column: {0: 0.0} for column in columns}

        weights = pd.to_numeric(stacked.get("n"), errors="coerce").fillna(0.0)
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            return {column: {0: 0.0} for column in columns}

        aggregated: Dict[str, Dict[int, float]] = {}
        for column in columns:
            if column not in stacked.columns:
                aggregated[column] = {0: 0.0}
                continue
            series = pd.to_numeric(stacked[column], errors="coerce")
            weighted_mean = float((series.fillna(0.0) * weights).sum() / total_weight)
            aggregated[column] = {0: weighted_mean}
        return aggregated
