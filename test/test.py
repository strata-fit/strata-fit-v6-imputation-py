import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strata_fit_v6_imputation_py.imputation_strategies.mean import MeanImputer
from strata_fit_v6_imputation_py.imputation_strategies.mice import MiceImputer


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


def build_client() -> MockAlgorithmClient:
    temp_dir = TemporaryDirectory()
    tmp_path = Path(temp_dir.name)
    datasets = []

    for idx, frame in enumerate(_build_dataset_frames(), start=1):
        csv_path = tmp_path / f"org_{idx}.csv"
        frame.to_csv(csv_path, index=False)
        datasets.append([{"database": csv_path, "db_type": "csv"}])

    client = MockAlgorithmClient(
        datasets=datasets,
        organization_ids=[1, 2, 3],
        module="strata_fit_v6_imputation_py",
    )

    # Persist the temporary files for the lifetime of the client.
    client._test_tmpdir = temp_dir  # type: ignore[attr-defined]
    return client


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


def test_imputation_central_end_to_end() -> None:
    client = build_client()
    org_ids = [org["id"] for org in client.organization.list()]
    columns = ["DAS28", "CRP", "ESR", "SJC28", "TJC28"]

    central_task = client.task.create(
        input_={
            "method": "central",
            "kwargs": {
                "organizations_to_include": org_ids,
                "imputation_config": {
                    "schema_version": 1,
                    "strategy": "mean",
                    "parameters": {
                        "columns": columns,
                    },
                },
            },
        },
        organizations=[org_ids[0]],
    )

    result = client.result.get(central_task["id"])

    assert result["type"] == "imputation"
    assert result["strategy"] == "mean"
    assert result["fitted"] is True
    assert result["schema_version"] == 1
    assert result["parameters"]["columns"] == columns
    assert result["metadata"]["n_organizations"] == 3
    assert "state" in result and result["state"]

def test_imputation_central_mice_end_to_end() -> None:
    client = build_client()
    org_ids = [org["id"] for org in client.organization.list()]
    columns = ["DAS28", "CRP", "ESR", "SJC28", "TJC28"]

    central_task = client.task.create(
        input_={
            "method": "central",
            "kwargs": {
                "organizations_to_include": org_ids,
                "imputation_config": {
                    "schema_version": 1,
                    "strategy": "mice",
                    "parameters": {
                        "columns": columns,
                        "max_iter": 3,
                    },
                },
            },
        },
        organizations=[org_ids[0]],
    )

    result = client.result.get(central_task["id"])

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

if __name__ == "__main__":
    test_imputation_central_end_to_end()
    test_imputation_central_mice_end_to_end()
