from typing import Any, Dict, Hashable, List

import pandas as pd
from vantage6.algorithm.tools.decorators import data
from v6_federated_core import MethodContext, dispatch_registered_method, to_v6_result

from .methods import METHOD_REGISTRY

@data(1)
def partial_compute(
    df1: pd.DataFrame,
    columns: List[str],
    imputation_strategy: Any,
    global_state: Dict[str, Any] | None = None,
) -> Dict[Hashable, Any]:
    """Thin V6 adapter for the typed imputation partial method."""
    method_context = MethodContext(
        method="partial_compute",
        meta={"df": df1},
    )
    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "partial_compute",
        {
            "columns": columns,
            "imputation_strategy": imputation_strategy,
            "global_state": global_state,
        },
        context=method_context,
    )
    return to_v6_result(envelope)

@data(1)
def get_local_sums(
    df1: pd.DataFrame,
    columns: List[str],
) -> Dict[str, Dict[str, float | int]]:
    """Thin V6 adapter for local column sums used by MICE initialization."""
    method_context = MethodContext(
        method="get_local_sums",
        meta={"df": df1},
    )
    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "get_local_sums",
        {
            "columns": columns,
        },
        context=method_context,
    )
    return to_v6_result(envelope)
