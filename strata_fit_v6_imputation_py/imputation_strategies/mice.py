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
        if not global_metric or not isinstance(global_metric, List):
            return df
        
        if isinstance(global_metric, List):
            model_config = global_metric[0]
        else:
            model_config = global_metric

        state = model_config['state']
        params = model_config['parameters']
        columns = params['columns']
        global_estimates = state['global_estimates']
        
        # Use the same max_iter as the training/central run
        max_iter = params['max_iter']
        
        data = df[columns].copy().values
        
        # 1. Initialize with the same parameters as central
        # We must match max_iter to ensure the chain converges similarly
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=max_iter, random_state=42)
        
        # 2. Prime the imputer
        imputer.fit(data)
        
        # 3. Corrected Weight Injection: Match by feature index
        # Create a lookup map for our federated estimates
        fed_map = {est['feat_idx']: est for est in global_estimates}
        
        for i in range(len(imputer.imputation_sequence_)):
            # The first element of the sequence tuple is the target feature index
            target_feat_idx = imputer.imputation_sequence_[i][0]
            target_estimator = imputer.imputation_sequence_[i][2]
            
            if target_feat_idx in fed_map:
                est_data = fed_map[target_feat_idx]
                target_estimator.coef_ = np.array(est_data['coef'])
                target_estimator.intercept_ = est_data['intercept']
                # BayesianRidge specifically needs this flag set to True
                target_estimator.fitted_ = True 
        
        # 4. Transform
        imputed_values = imputer.transform(data)
        
        return pd.DataFrame(imputed_values, columns=columns, index=df.index)