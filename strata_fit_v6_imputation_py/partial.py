import pandas as pd
from typing import Any, List, Dict, Hashable
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import data
from .imputation_strategies.base import ImputationStrategyEnum
from .imputation_strategies.base import STRATEGY_REGISTRY


@data(1)
def partial_compute(
    df1: pd.DataFrame,
    columns: List[str],
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN_IMPUTER,
    global_state: Dict = None
) -> Dict[Hashable, Any]:
    """compute the node specific imputation metrics

    Args:
        df1 (pd.DataFrame): local node data
        columns (List[str]): columns for imputation
        imputation_strategy (Enum, optional): imputation method to use. Defaults to ImputationStrategyEnum.MeanImputer.

    Returns:
        Dict[Hashable, Any]:
    """
    # imputer = STRATEGY_REGISTRY[imputation_strategy]()
    # info(f"Computing imputation metrics with strategy: {imputation_strategy.value}")
    # result = imputer.compute(df1, columns)

    imputer = STRATEGY_REGISTRY[imputation_strategy]()
    # Pass global_state to the compute method
    result = imputer.compute(df1, columns, global_state=global_state)
    return result

@data(1)
def get_local_sums(df: pd.DataFrame, columns: List[str]) -> Dict:
    """
    Calculates the sum and count of non-null values for each column 
    to facilitate global mean calculation in the central node.
    """
    results = {}
    for col in columns:
        if col in df.columns:
            # Drop NaNs for the calculation
            series = df[col].dropna()
            results[col] = {
                "sum": float(series.sum()),
                "count": int(series.count())
            }
        else:
            # Handle cases where a node might be missing a column entirely
            results[col] = {"sum": 0.0, "count": 0}
            
    return results
    
    