from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .io import normalize_payload, write_output
from .runtime import run_context
from .service import run_partial_method


def _load_dataframe(dataset_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(dataset_path))


def partial_compute_frame(
    df: pd.DataFrame,
    *,
    columns: List[str],
    imputation_strategy: Any,
    global_state: Dict[str, Any] | None = None,
    client: Any = None,
) -> Dict[str, Any]:
    return run_partial_method(
        "partial_compute",
        df=df,
        raw_input={
            "columns": columns,
            "imputation_strategy": imputation_strategy,
            "global_state": global_state,
        },
        client=client,
    )


def get_local_sums_frame(
    df: pd.DataFrame,
    *,
    columns: List[str],
    client: Any = None,
) -> Dict[str, Dict[str, float | int]]:
    return run_partial_method(
        "get_local_sums",
        df=df,
        raw_input={"columns": columns},
        client=client,
    )


@run_context(
    input_uris="dataset_path",
    output_uris="output_path",
    named_arguments=["columns", "imputation_strategy", "global_state"],
)
def partial_compute(
    dataset_path: str | Path,
    columns: List[str],
    imputation_strategy: Any,
    global_state: Dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    result = normalize_payload(
        partial_compute_frame(
            _load_dataframe(dataset_path),
            columns=columns,
            imputation_strategy=imputation_strategy,
            global_state=global_state,
        )
    )
    write_output(output_path, result)
    return result


@run_context(
    input_uris="dataset_path",
    output_uris="output_path",
    named_arguments=["columns"],
)
def get_local_sums(
    dataset_path: str | Path,
    columns: List[str],
    output_path: str | Path | None = None,
) -> Dict[str, Dict[str, float | int]]:
    result = normalize_payload(
        get_local_sums_frame(
            _load_dataframe(dataset_path),
            columns=columns,
        )
    )
    write_output(output_path, result)
    return result
