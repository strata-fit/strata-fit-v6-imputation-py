import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from typing import List, Dict, Any
from .base import ImputationStrategy, register_imputation_strategy, ImputationStrategyEnum

@register_imputation_strategy(ImputationStrategyEnum.MICE_IMPUTER)
class MiceImputer(ImputationStrategy):

    def compute(self, df: pd.DataFrame, columns: List[str], global_state: Dict = None) -> Dict:
        data_work = df[columns].copy()
        
        # 1. Base Initialization (Global Means)
        # This prevents NaNs from poisoning the matrix math
        means = global_state.get("initial_means", {})
        for col in columns:
            if col in means:
                data_work[col] = data_work[col].fillna(means[col])
            else:
                # Fallback safety: fill with 0 if mean is somehow missing
                data_work[col] = data_work[col].fillna(0.0)

        # 2. Refine Imputations (Feedback Loop)
        # If we have coefficients from the PREVIOUS round, use them to update our X
        if "global_estimates" in global_state:
            for est in global_state["global_estimates"]:
                target_col = columns[est["feat_idx"]]
                missing_mask = df[target_col].isna()
                
                if missing_mask.any():
                    # Predict values using previous round's Beta
                    X_cols = [columns[idx] for idx in est["neighbor_indices"]]
                    X_missing = data_work.loc[missing_mask, X_cols].values
                    X_missing = np.hstack([np.ones((X_missing.shape[0], 1)), X_missing])
                    
                    beta = np.array([est["intercept"]] + est["coef"])
                    data_work.loc[missing_mask, target_col] = X_missing @ beta

        # 3. Compute Sufficient Statistics for the NEXT round
        stats_per_column = []
        for i, target_col in enumerate(columns):
            observed_mask = df[target_col].notna()
            if not observed_mask.any():
                continue

            y = df.loc[observed_mask, target_col].values
            X_cols = [c for c in columns if c != target_col]
            X = data_work.loc[observed_mask, X_cols].values
            X = np.hstack([np.ones((X.shape[0], 1)), X])

            stats_per_column.append({
                "feat_idx": i,
                "xtx": (X.T @ X).tolist(),
                "xty": (X.T @ y).tolist(),
                "n_obs": int(len(y))
            })

        return {"column_stats": stats_per_column}

    def aggregate(self, node_metrics: List[Dict], columns: List[str]) -> Dict:
        global_estimates = []
        
        # Identify which columns were actually processed
        processed_indices = {s["feat_idx"] for n in node_metrics for s in n["column_stats"]}
        
        for feat_idx in sorted(list(processed_indices)):
            # Sum up stats across nodes for this specific column
            dim = len(columns) # Total columns including intercept
            global_xtx = None
            global_xty = None
            
            for node in node_metrics:
                stat = next((s for s in node["column_stats"] if s["feat_idx"] == feat_idx), None)
                if stat:
                    xtx_local = np.array(stat["xtx"])
                    xty_local = np.array(stat["xty"])
                    global_xtx = xtx_local if global_xtx is None else global_xtx + xtx_local
                    global_xty = xty_local if global_xty is None else global_xty + xty_local

            # Solve (XTX + ridge) * beta = XTy
            ridge = 1e-6
            global_xtx += np.eye(global_xtx.shape[0]) * ridge
            beta = np.linalg.solve(global_xtx, global_xty)

            global_estimates.append({
                "feat_idx": feat_idx,
                "neighbor_indices": [i for i in range(len(columns)) if i != feat_idx],
                "coef": beta[1:].tolist(),
                "intercept": float(beta[0])
            })

        return {"global_estimates": global_estimates}

    def impute(self, df: pd.DataFrame, global_metric: Dict) -> pd.DataFrame:
        """Standard MICE transform using global coefficients."""
        if not global_metric or "global_estimates" not in global_metric:
            return df 

        # 1. Initialize the imputer
        # We use a small max_iter because we are manually overriding the logic
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=1, random_state=42)
        
        # 2. "Prime" the imputer so it builds the internal imputation_sequence_
        # It needs to see the data structure to know how many estimators to create
        imputer.fit(df.values) 
        
        # 3. Surgical injection of global weights
        for i, step_meta in enumerate(global_metric["global_estimates"]):
            # Index [2] is the estimator (the BayesianRidge model)
            target_estimator = imputer.imputation_sequence_[i][2]
            
            target_estimator.coef_ = np.array(step_meta["coef"])
            target_estimator.intercept_ = step_meta["intercept"]
            
            # Sklearn estimators often need this flag to realize they are "fitted"
            # though BayesianRidge usually works fine without it after coef_ is set
            target_estimator.fitted_ = True 

        # 4. Perform the actual imputation
        imputed_values = imputer.transform(df.values)
        
        return pd.DataFrame(imputed_values, columns=df.columns, index=df.index)