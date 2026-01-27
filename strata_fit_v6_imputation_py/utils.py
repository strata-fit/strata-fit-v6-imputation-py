from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import polars as pl
import pandas as pd

def stack_results(results: List[Dict[Any, Any]]) -> pd.DataFrame:
    """Stack local node data into a single dataframe

    Args:
        results (List[Dict[Any, Any]]): Output from partial/central functions

    Returns:
        pd.DataFrame: combined data
    """
    dfs = []
    for df_dict in results:
        dfs.append(pl.DataFrame({col: list(inner_dict.values()) for col, inner_dict in df_dict.items()}))

    return pl.concat(dfs, how="vertical").to_pandas()


def build_imputation_model_config(
    strategy: str,
    parameters: Dict[str, Any],
    state: Dict[str, Any],
    n_organizations: Optional[int] = None,
    schema_version: int = 1,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build a standardized imputation model config for federated pipelines.

    Args:
        strategy: Name of the imputation strategy, e.g., "mean", "mice".
        parameters: User-specified or algorithm parameters (columns, hyperparams).
        state: Aggregated model outputs from nodes (strategy-specific).
        n_organizations: Optional number of contributing organizations.
        schema_version: Schema version for the config.
        metadata: Optional extra metadata.

    Returns:
        JSON-serializable dictionary representing the global imputation model.
    """
    if metadata is None:
        metadata = {}

    # always include creation timestamp
    metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat() + "Z")
    if n_organizations is not None:
        metadata["n_organizations"] = n_organizations

    config = {
        "schema_version": schema_version,
        "type": "imputation",
        "strategy": strategy,
        "fitted": True,
        "parameters": parameters,
        "state": state,
        "metadata": metadata,
    }

    return config