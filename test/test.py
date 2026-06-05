from __future__ import annotations

from io import StringIO
from pathlib import Path

import json
import pandas as pd

from strata_fit_v6_imputation_py import run_local_imputation
from strata_fit_v6_imputation_py.imputation_strategies.mean import MeanImputer
from strata_fit_v6_imputation_py.imputation_strategies.mice import MiceImputer
from strata_fit_v6_imputation_py.partial import partial_compute
from strata_fit_v6_imputation_py.runtime import RunContext


def _build_dataset_frames() -> list[pd.DataFrame]:
    return [
        pd.DataFrame(
            {
                "pat_ID": [1, 1, 2, 2],
                "DAS28": [1.0, None, 3.0, 5.0],
                "CRP": [2.0, 4.0, None, 8.0],
                "ESR": [10.0, 12.0, 8.0, None],
                "SJC28": [1.0, 2.0, None, 3.0],
                "TJC28": [0.0, None, 2.0, 1.0],
            }
        ),
        pd.DataFrame(
            {
                "pat_ID": [5, 5, 6, 6],
                "DAS28": [6.0, 4.0, None, 2.0],
                "CRP": [8.0, None, 4.0, 2.0],
                "ESR": [None, 8.0, 12.0, 10.0],
                "SJC28": [3.0, None, 2.0, 1.0],
                "TJC28": [1.0, 2.0, None, 0.0],
            }
        ),
        pd.DataFrame(
            {
                "pat_ID": [7, 7, 8, 8],
                "DAS28": [2.0, None, 4.0, 6.0],
                "CRP": [1.0, None, 7.0, 9.0],
                "ESR": [9.0, 11.0, None, 13.0],
                "SJC28": [1.0, None, 2.0, 4.0],
                "TJC28": [1.0, 2.0, None, 3.0],
            }
        ),
    ]


def test_compute_returns_dict_for_supported_strategies() -> None:
    frame = _build_dataset_frames()[0]
    columns = ["DAS28", "CRP", "ESR", "SJC28", "TJC28"]

    mean_payload = MeanImputer().compute(frame, columns)
    mice_payload = MiceImputer().compute(
        frame,
        columns,
        global_state={
            "initial_means": {column: float(frame[column].dropna().mean()) for column in columns}
        },
    )

    assert isinstance(mean_payload, dict)
    assert isinstance(mice_payload, dict)


def test_imputation_central_mean_end_to_end() -> None:
    columns = ["DAS28", "CRP", "ESR", "SJC28", "TJC28"]
    result = run_local_imputation(
        _build_dataset_frames(),
        organizations_to_include=[0, 1, 2],
        imputation_config={
            "schema_version": 1,
            "strategy": "mean",
            "parameters": {"columns": columns},
        },
    )

    assert result["type"] == "imputation"
    assert result["strategy"] == "mean"
    assert result["fitted"] is True
    assert result["schema_version"] == 1
    assert result["parameters"]["columns"] == columns
    assert result["metadata"]["n_organizations"] == 3
    assert "state" in result and result["state"]


def test_imputation_central_mice_end_to_end() -> None:
    columns = ["DAS28", "CRP", "ESR", "SJC28", "TJC28"]
    result = run_local_imputation(
        _build_dataset_frames(),
        organizations_to_include=[0, 1, 2],
        imputation_config={
            "schema_version": 1,
            "strategy": "mice",
            "parameters": {"columns": columns, "max_iter": 3},
        },
    )

    assert result["type"] == "imputation"
    assert result["strategy"] == "mice"
    assert result["fitted"] is True
    assert result["schema_version"] == 1
    assert result["parameters"]["columns"] == columns
    assert result["parameters"]["max_iter"] == 3
    assert result["metadata"]["n_organizations"] == 3
    assert "initial_means" in result["state"]
    assert "global_estimates" in result["state"]
    assert isinstance(result["state"]["global_estimates"], list)


def test_run_context_partial_writes_output(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_path = tmp_path / "out.json"
    _build_dataset_frames()[0].to_csv(dataset_path, index=False)

    context = RunContext(
        source=tmp_path / "run_context.json",
        payload={
            "entrypoint": {"name": "partial_compute"},
            "arguments": {
                "named": {
                    "columns": ["DAS28", "CRP"],
                    "imputation_strategy": "mean",
                }
            },
            "inputs": [{"uri": str(dataset_path)}],
            "outputs": [{"uri": str(output_path)}],
        },
    )

    result = partial_compute(run_context=context)
    assert json.loads(output_path.read_text(encoding="utf-8")) == result


if __name__ == "__main__":
    test_imputation_central_mean_end_to_end()
    test_imputation_central_mice_end_to_end()
