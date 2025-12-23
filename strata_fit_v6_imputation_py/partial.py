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
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN_IMPUTER
) -> Dict[Hashable, Any]:
    """compute the node specific imputation metrics

    Args:
        df1 (pd.DataFrame): local node data
        columns (List[str]): columns for imputation
        imputation_strategy (Enum, optional): imputation method to use. Defaults to ImputationStrategyEnum.MeanImputer.

    Returns:
        Dict[Hashable, Any]:
    """
    return _partial_compute(df1, columns, imputation_strategy)

def _partial_compute(
    df1: pd.DataFrame,
    columns: List[str],
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN_IMPUTER
) -> Dict[Hashable, Any]:
    
    imputer = STRATEGY_REGISTRY[imputation_strategy]()
    info(f"Computing imputation metrics with strategy: {imputation_strategy.value}")
    result = imputer.compute(df1, columns)

    return result.to_dict()