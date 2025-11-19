import pandas as pd
from typing import Any, List, Dict

from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import data
from .utils import stack_results
from .types import ImputationStrategyEnum
from .types import STRATEGY_MAP
import polars as pl


@data(1)
def partial_compute(
    df1: pd.DataFrame,
    columns: List[str],
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN
) -> Any:

    """ Decentral part of the algorithm """
    imputer = STRATEGY_MAP[imputation_strategy]()
    info(f"Computing imputation metrics with strategy: {imputation_strategy.value}")
    result = imputer.compute(df1, columns)

    return result.to_dict()


@data(1)
def partial_impute(
    df1: pd.DataFrame,
    global_metrics: Dict,
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN
) -> Any:
    """impute global metrics into node data

    Args:
        df1 (pd.DataFrame): node data
        columns (List[str]): columns to impute
        global_metrics (List[Dict[Any, Any]]): global imputation values

    Returns:
        List[Dict[Any, Any]]: imputed data
    """
    imputer = STRATEGY_MAP[imputation_strategy]()
    info("imputing global metrics into local node data")

    result = imputer.impute(df1, global_metrics)

    return result.to_dict()