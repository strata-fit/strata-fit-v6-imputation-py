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
    Federated deterministic MICE (pooled OLS/ridge per feature) that follows
    the algorithm of sklearn's ``IterativeImputer`` (imputation_order="ascending").

    Model state
    -----------
    ``initial_means``      global per-column means (used for the initial fill)
    ``feature_order``      order in which the columns are visited in every
                           iteration (ascending fraction of missing values)
    ``estimate_sequence``  ALL fitted regressions in the order they were fitted:
                           ``max_iter * n_columns`` entries, each with
                           ``iteration``, ``feat_idx``, ``neighbor_indices``,
                           ``coef`` and ``intercept``

    The working matrix of a node is never stored. It is reconstructed by
    *replaying* the estimate sequence on the mean-filled local data
    (``_replay``). This reproduces the persistent state of the sequential
    chained equations: the regression for feature ``t`` in iteration ``k`` is fitted
    on exactly the imputations produced by all earlier regressions, and the final
    imputation replays the complete sequence once.

    Only sufficient statistics (X'X, X'y, n) and regression coefficients leave
    a node/the aggregator - never row-level data.

    Backward compatibility: models created by older versions only contain
    ``global_estimates`` (one final estimate per feature); ``impute`` and
    ``compute`` still accept them (legacy behaviour).
    """

    RIDGE = 1e-6

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def imputation_order(
        sum_results: List[Dict[str, Dict[str, float | int]]],
        columns: List[str],
    ) -> List[int]:
        """
        Order of the features: ascending fraction of missing values, i.e.
        descending number of observed values (all columns share the same number
        of rows). Ties keep the column order (stable), like sklearn.
        """
        counts = np.array(
            [
                sum(int(node.get(column, {}).get("count", 0)) for node in sum_results)
                for column in columns
            ]
        )
        return [int(i) for i in np.argsort(-counts, kind="stable")]

    @staticmethod
    def _sequence(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        sequence = state.get("estimate_sequence")
        if sequence is not None:
            return list(sequence)
        # legacy state: one final estimate per feature, applied in column order
        return sorted(
            state.get("global_estimates", []),
            key=lambda estimate: estimate["feat_idx"],
        )

    @staticmethod
    def _replay(
        df: pd.DataFrame,
        columns: List[str],
        state: Dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Rebuild the current working matrix: mean fill, then apply every stored
        regression in the order it was fitted (only to the originally missing cells).
        Returns (working matrix, boolean mask of originally missing cells).
        """
        data = df[columns].to_numpy(dtype=float, na_value=np.nan)
        missing = np.isnan(data)

        initial_means = state.get("initial_means", {})
        means = np.array([float(initial_means.get(c, 0.0)) for c in columns])
        work = np.where(missing, means, data)

        for estimate in MiceImputer._sequence(state):
            target = estimate["feat_idx"]
            rows = missing[:, target]
            if not rows.any():
                continue
            predictors = estimate["neighbor_indices"]
            coef = np.asarray(estimate["coef"], dtype=float)
            work[rows, target] = (
                float(estimate["intercept"]) + work[rows][:, predictors] @ coef
            )

        return work, missing

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

        # current working matrix = replay of all estimates fitted so far
        work, missing = self._replay(df, columns, state)

        observed_mask = ~missing[:, target_feat_idx]

        if not observed_mask.any():
            return {"column_stats": []}

        predictor_indices = [
            j for j in range(len(columns)) if j != target_feat_idx
        ]

        X = work[observed_mask][:, predictor_indices]
        y = work[observed_mask, target_feat_idx]

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

        if "estimate_sequence" not in state:
            return self._impute_legacy(df, columns, params, state)

        # replay the complete fitted sequence once
        work, missing = self._replay(df, columns, state)

        result = df.copy()
        for j, column in enumerate(columns):
            if missing[:, j].any():
                result[column] = work[:, j]

        return result

    def _impute_legacy(
        self,
        df: pd.DataFrame,
        columns: List[str],
        params: Dict[str, Any],
        state: Dict[str, Any],
    ) -> pd.DataFrame:
        """Models from older versions: final estimates re-applied max_iter times."""

        max_iter = int(params.get("max_iter", 10))
        initial_means = state["initial_means"]

        data_work = df.copy()

        for col in columns:
            data_work[col] = data_work[col].fillna(float(initial_means[col]))

        estimates = sorted(
            state["global_estimates"],
            key=lambda x: x["feat_idx"],
        )

        for _ in range(max_iter):

            for estimate in estimates:

                target_column = columns[estimate["feat_idx"]]

                missing_mask = df[target_column].isna()

                if not missing_mask.any():
                    continue

                predictor_columns = [
                    columns[i] for i in estimate["neighbor_indices"]
                ]

                X_missing = data_work.loc[
                    missing_mask,
                    predictor_columns,
                ].values

                X_missing = np.hstack(
                    [np.ones((X_missing.shape[0], 1)), X_missing]
                )

                beta = np.array(
                    [estimate["intercept"]] + estimate["coef"],
                    dtype=float,
                )

                data_work.loc[missing_mask, target_column] = X_missing @ beta

        return data_work