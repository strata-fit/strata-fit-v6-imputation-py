from typing import Any, Dict, List
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