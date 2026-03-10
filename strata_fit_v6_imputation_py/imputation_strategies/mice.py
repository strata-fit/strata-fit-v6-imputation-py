import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from typing import List, Dict, Any
from .base import ImputationStrategy, register_imputation_strategy, ImputationStrategyEnum

@register_imputation_strategy(ImputationStrategyEnum.MICE_IMPUTER)
class MiceImputer(ImputationStrategy):

    def compute(self, df: pd.DataFrame, columns: List[str], global_state: Dict = None) -> Dict:
        data_work = df[columns].copy()
        
        # MANDATORY: Fill NaNs with Global Means before fitting
        # This prevents "Local Mean Bias" from ruining the first round of regressions
        if global_state and "initial_means" in global_state:
            for col, mean_val in global_state["initial_means"].items():
                data_work[col] = data_work[col].fillna(mean_val)
        
        # Now, even with max_iter=1, the regression starts from a 'Globally' 
        # consistent baseline.
        imputer = IterativeImputer(
            estimator=BayesianRidge(), 
            max_iter=1, 
            random_state=42
        )
        
        imputer.fit(data_work.values)
        
        # 3. EXTRACT (Same as before, with type casting)
        estimates = []
        for step in imputer.imputation_sequence_:
            estimates.append({
                "feat_idx": int(step[0]),
                "neighbor_indices": [int(idx) for idx in step[1]],
                "coef": [float(c) for c in step[2].coef_],
                "intercept": float(step[2].intercept_),
            })
            
        return {"estimates": estimates, "n_samples": int(len(df))}

    def aggregate(self, node_metrics: List[Dict], columns: List[str]) -> Dict:
        """Averages coefficients from all nodes."""
        total_n = sum(node['n_samples'] for node in node_metrics)
        first_node_estimates = node_metrics[0]['estimates']
        global_estimates = []

        for i in range(len(first_node_estimates)):
            avg_coef = np.zeros_like(first_node_estimates[i]['coef'], dtype=float)
            avg_intercept = 0.0
            
            for node in node_metrics:
                weight = node['n_samples'] / total_n
                avg_coef += np.array(node['estimates'][i]['coef']) * weight
                avg_intercept += node['estimates'][i]['intercept'] * weight
            
            global_estimates.append({
                "feat_idx": first_node_estimates[i]['feat_idx'],
                "neighbor_indices": first_node_estimates[i]['neighbor_indices'],
                "coef": avg_coef.tolist(),
                "intercept": avg_intercept
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