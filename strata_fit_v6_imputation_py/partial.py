import pandas as pd
from typing import Any, List, Type

from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import data
from .types import ImputationStrategyEnum
from .types import STRATEGY_MAP
import polars as pl


@data(1)
def partial_compute(
    df1: pd.DataFrame,
    columns: List[str],
    imputation_strategy: ImputationStrategyEnum
) -> Any:

    """ Decentral part of the algorithm """
    imputer = STRATEGY_MAP[imputation_strategy]()
    info(f"Computing imputation metrics with strategy: {imputation_strategy.value}")
    result = imputer.compute(df1, columns)

    return result.to_dict()
