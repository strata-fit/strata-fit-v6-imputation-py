import pandas as pd
from typing import Any, List, Dict, Hashable
from vantage6.algorithm.tools.decorators import data
from v6_federated_core import MethodContext, dispatch_registered_method, to_v6_result

from .methods import METHOD_REGISTRY

@data(1)
def partial_compute(
    df1: pd.DataFrame,
    columns: List[str],
    imputation_strategy: Any
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
        },
        context=method_context,
    )
    return to_v6_result(envelope)
