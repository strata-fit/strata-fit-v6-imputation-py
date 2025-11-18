"""
This file contains all partial algorithm functions, that are normally executed
on all nodes for which the algorithm is executed.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled). From there, they are sent to the partial task
or directly to the user (if they requested partial results).
"""
import pandas as pd
from typing import Any, List, Type

from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import data
from imputation_strategies.base import ImputationStrategy


@data(1)
def partial_compute(
    df1: pd.DataFrame,
    columns: List[str],
    imputation_strategy: Type[ImputationStrategy]
) -> Any:

    """ Decentral part of the algorithm """
    imputer = imputation_strategy()
    info(f"Computing imputation metrics with strategy: {imputation_strategy.__class__}")
    result = imputer.compute(df1, columns)

    return result.to_dict()
