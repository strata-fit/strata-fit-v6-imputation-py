from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from .base import ImputationStrategy, ImputationStrategyEnum, register_imputation_strategy


@register_imputation_strategy(ImputationStrategyEnum.MICE_IMPUTER)
class MiceImputer(ImputationStrategy):
    def compute(
        self,
        df: pd.DataFrame,
        columns: List[str],
        global_state: Dict[str, Any] | None = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        state = global_state or {}
        data_work = df[columns].copy()

        means = state.get("initial_means", {})
        for column in columns:
            data_work[column] = data_work[column].fillna(float(means.get(column, 0.0)))

        for estimate in state.get("global_estimates", []):
            feat_idx = estimate.get("feat_idx")
            if not isinstance(feat_idx, int) or feat_idx < 0 or feat_idx >= len(columns):
                continue

            target_column = columns[feat_idx]
            missing_mask = df[target_column].isna()
            if not missing_mask.any():
                continue

            neighbor_indices = estimate.get("neighbor_indices", [])
            predictor_columns = [
                columns[idx]
                for idx in neighbor_indices
                if isinstance(idx, int) and 0 <= idx < len(columns)
            ]
            if not predictor_columns:
                continue

            x_missing = data_work.loc[missing_mask, predictor_columns].values
            x_missing = np.hstack([np.ones((x_missing.shape[0], 1)), x_missing])
            beta = np.array(
                [estimate.get("intercept", 0.0)] + list(estimate.get("coef", [])),
                dtype=float,
            )
            if beta.shape[0] != x_missing.shape[1]:
                continue

            data_work.loc[missing_mask, target_column] = x_missing @ beta

        stats_per_column: List[Dict[str, Any]] = []
        for feat_idx, target_column in enumerate(columns):
            observed_mask = df[target_column].notna()
            if not observed_mask.any():
                continue

            y = df.loc[observed_mask, target_column].values
            predictor_columns = [column for column in columns if column != target_column]
            x = data_work.loc[observed_mask, predictor_columns].values
            x = np.hstack([np.ones((x.shape[0], 1)), x])

            stats_per_column.append(
                {
                    "feat_idx": feat_idx,
                    "xtx": (x.T @ x).tolist(),
                    "xty": (x.T @ y).tolist(),
                    "n_obs": int(len(y)),
                }
            )

        return {"column_stats": stats_per_column}

    def aggregate(
        self,
        node_metrics: List[Dict[str, Any]],
        columns: List[str],
        global_means: Dict[str, Any] | None = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        del global_means

        global_estimates: List[Dict[str, Any]] = []
        processed_indices = sorted(
            {
                stat.get("feat_idx")
                for node in node_metrics
                for stat in node.get("column_stats", [])
                if isinstance(stat.get("feat_idx"), int)
            }
        )

        for feat_idx in processed_indices:
            global_xtx = None
            global_xty = None

            for node in node_metrics:
                stat = next(
                    (
                        item
                        for item in node.get("column_stats", [])
                        if item.get("feat_idx") == feat_idx
                    ),
                    None,
                )
                if stat is None:
                    continue

                xtx_local = np.array(stat["xtx"], dtype=float)
                xty_local = np.array(stat["xty"], dtype=float)
                global_xtx = xtx_local if global_xtx is None else global_xtx + xtx_local
                global_xty = xty_local if global_xty is None else global_xty + xty_local

            if global_xtx is None or global_xty is None:
                continue

            ridge = 1e-6
            global_xtx = global_xtx + np.eye(global_xtx.shape[0]) * ridge
            beta = np.linalg.solve(global_xtx, global_xty)

            global_estimates.append(
                {
                    "feat_idx": feat_idx,
                    "neighbor_indices": [i for i in range(len(columns)) if i != feat_idx],
                    "coef": beta[1:].tolist(),
                    "intercept": float(beta[0]),
                }
            )

        return {"global_estimates": global_estimates}

    def impute(self, df: pd.DataFrame, global_metric: Dict[str, Any]) -> pd.DataFrame:
        if not global_metric:
            return df

        model_config = global_metric[0] if isinstance(global_metric, list) else global_metric
        if not isinstance(model_config, dict):
            return df

        state = model_config.get("state", {})
        params = model_config.get("parameters", {})
        columns = params.get("columns")
        if not isinstance(columns, list) or not columns:
            return df

        global_estimates = state.get("global_estimates", [])
        max_iter = int(params.get("max_iter", 5))
        data = df[columns].copy().values

        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=max_iter,
            random_state=42,
        )
        imputer.fit(data)

        fed_map = {
            estimate.get("feat_idx"): estimate
            for estimate in global_estimates
            if isinstance(estimate.get("feat_idx"), int)
        }

        for step in imputer.imputation_sequence_:
            target_feat_idx = step[0]
            estimator = step[2]
            estimate = fed_map.get(target_feat_idx)
            if estimate is None:
                continue

            fed_coef = np.array(estimate.get("coef", []), dtype=float)
            local_coef = np.array(getattr(estimator, "coef_", []), dtype=float)
            if local_coef.size and fed_coef.shape != local_coef.shape:
                continue

            estimator.coef_ = fed_coef
            estimator.intercept_ = float(estimate.get("intercept", 0.0))
            estimator.fitted_ = True

        imputed_values = imputer.transform(data)
        return pd.DataFrame(imputed_values, columns=columns, index=df.index)
