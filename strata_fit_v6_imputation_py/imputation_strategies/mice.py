# mice.py

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import (
    ImputationStrategy,
    ImputationStrategyEnum,
    register_imputation_strategy,
)


@register_imputation_strategy(ImputationStrategyEnum.MICE_IMPUTER)
class MiceImputer(ImputationStrategy):
    """
    Federated deterministic MICE using pooled ridge regression
    with TRUE sequential chained updates (Gauss-Seidel style).

    IMPORTANT:
    - updates are performed feature-by-feature
    - each feature update immediately modifies the working matrix
    - later features in the same iteration see earlier updates
    - local matrices persist through the chain
    """

    RIDGE = 1e-6

    # >>> NEW (A): two helper methods <<<
    @staticmethod
    def _apply_estimate(
        df: pd.DataFrame,
        data_work: pd.DataFrame,
        columns: List[str],
        estimate: Dict[str, Any],
    ) -> None:
        """Overwrite the originally-missing cells of one feature, in place."""
        col = columns[estimate["feat_idx"]]
        missing_mask = df[col].isna()
        if not missing_mask.any():
            return
        predictor_columns = [columns[i] for i in estimate["neighbor_indices"]]
        X_missing = data_work.loc[missing_mask, predictor_columns].to_numpy(float)
        X_missing = np.hstack([np.ones((X_missing.shape[0], 1)), X_missing])
        beta = np.array([estimate["intercept"]] + estimate["coef"], dtype=float)
        data_work.loc[missing_mask, col] = X_missing @ beta

    @staticmethod
    def _sequence(state: Dict[str, Any]) -> List[Dict[str, Any]] | None:
        """Ordered list of every fitted step (round by round), or None for old models."""
        seq = state.get("imputation_sequence")
        return list(seq) if seq else None

    # ============================================================
    # LOCAL PARTIAL COMPUTE
    # ============================================================

    def compute(
        self,
        df: pd.DataFrame,
        columns: List[str],
        global_state: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:

        state = global_state or {}

        target_feat_idx = state.get("target_feat_idx")

        if target_feat_idx is None:
            raise ValueError("target_feat_idx missing from global_state")

        target_column = columns[target_feat_idx]

        # --------------------------------------------------------
        # Reconstruct CURRENT chained matrix
        # --------------------------------------------------------

        data_work = df[columns].copy()

        # Initial mean fill
        initial_means = state.get("initial_means", {})

        for col in columns:
            mean_val = float(initial_means.get(col, 0.0))
            data_work[col] = data_work[col].fillna(mean_val)

        # --------------------------------------------------------
        # APPLY CURRENT GLOBAL CHAIN SEQUENTIALLY
        # --------------------------------------------------------

        # >>> CHANGED (B): replay every step fitted so far, in fitting order <<<
        # Older models without a stored sequence fall back to one pass
        # over the latest estimate per feature.
        sequence = self._sequence(state)
        if sequence is None:
            sequence = sorted(
                state.get("global_estimates", []),
                key=lambda x: x["feat_idx"],
            )

        for estimate in sequence:
            self._apply_estimate(df, data_work, columns, estimate)

        # --------------------------------------------------------
        # Compute sufficient statistics ONLY for target feature
        # --------------------------------------------------------

        observed_mask = df[target_column].notna()

        if not observed_mask.any():
            return {"column_stats": []}

        predictor_columns = [
            c for c in columns if c != target_column
        ]

        X = data_work.loc[
            observed_mask,
            predictor_columns,
        ].values

        y = df.loc[
            observed_mask,
            target_column,
        ].values

        X = np.hstack([np.ones((X.shape[0], 1)), X])

        xtx = X.T @ X
        xty = X.T @ y

        return {
            "column_stats": [
                {
                    "feat_idx": target_feat_idx,
                    "xtx": xtx.tolist(),
                    "xty": xty.tolist(),
                    "n_obs": int(len(y)),
                }
            ]
        }

    # ============================================================
    # GLOBAL AGGREGATION
    # ============================================================

    def aggregate(
        self,
        node_metrics: List[Dict[str, Any]],
        columns: List[str],
        global_means: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:

        del global_means

        # only ONE feature processed each call
        feat_idx = None

        global_xtx = None
        global_xty = None

        for node in node_metrics:

            stats = node.get("column_stats", [])

            if not stats:
                continue

            stat = stats[0]

            feat_idx = stat["feat_idx"]

            xtx_local = np.array(stat["xtx"], dtype=float)
            xty_local = np.array(stat["xty"], dtype=float)

            if global_xtx is None:
                global_xtx = xtx_local
                global_xty = xty_local
            else:
                global_xtx += xtx_local
                global_xty += xty_local

        if global_xtx is None:
            return {"global_estimates": []}

        # pooled ridge solve
        global_xtx += (
            np.eye(global_xtx.shape[0]) * self.RIDGE
        )

        beta = np.linalg.solve(global_xtx, global_xty)

        return {
            "global_estimates": [
                {
                    "feat_idx": feat_idx,
                    "neighbor_indices": [
                        i
                        for i in range(len(columns))
                        if i != feat_idx
                    ],
                    "coef": beta[1:].tolist(),
                    "intercept": float(beta[0]),
                }
            ]
        }

    # ============================================================
    # FINAL IMPUTATION
    # ============================================================

    def impute(
        self,
        df: pd.DataFrame,
        global_metric: Dict[str, Any],
    ) -> pd.DataFrame:

        if not global_metric:
            return df.copy()

        model_config = (
            global_metric[0]
            if isinstance(global_metric, list)
            else global_metric
        )

        state = model_config["state"]
        params = model_config["parameters"]

        columns = params["columns"]
        max_iter = int(params.get("max_iter", 10))

        initial_means = state["initial_means"]

        # --------------------------------------------------------
        # Initialize matrix
        # --------------------------------------------------------

        data_work = df.copy()

        for col in columns:
            data_work[col] = data_work[col].fillna(
                float(initial_means[col])
            )

        # >>> CHANGED (C): replay the chain exactly as it was fitted <<<
        sequence = self._sequence(state)

        if sequence is not None:
            for estimate in sequence:
                self._apply_estimate(df, data_work, columns, estimate)
            return data_work

        # Legacy models (no stored sequence): iterate the final estimates.
        estimates = sorted(
            state["global_estimates"],
            key=lambda x: x["feat_idx"],
        )
        for _ in range(max_iter):
            for estimate in estimates:
                self._apply_estimate(df, data_work, columns, estimate)
        return data_work