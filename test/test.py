from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
from strata_fit_v6_imputation_py.imputation_strategies import ImputationStrategyEnum
from pathlib import Path

# get path of current directory
data_directory = Path(__file__).parent
org_ids = [1,2,3]

## Mock client
client = MockAlgorithmClient(
    datasets=[
        [{
            "database": data_directory / "data_times/alpha.csv",
            "db_type": "csv"
        }],
        [{
            "database": data_directory / "data_times/beta.csv",
            "db_type": "csv"
        }],
        [{
            "database": data_directory / "data_times/gamma.csv",
            "db_type": "csv"
        }]
    ],
    organization_ids=org_ids,
    module="strata_fit_v6_imputation_py"
)

organizations = client.organization.list()
columns = ["DAS28", "CRP", "ESR", "SJC28", "TJC28"]

# Run the central method on 1 node and get the results
central_task = client.task.create(
    input_={
        "method":"central",
        "kwargs": {
            "columns" : columns,
            "organizations_to_include" : org_ids,
            "imputation_strategy" : ImputationStrategyEnum.MeanImputer
        }
    },
    organizations=[org_ids[0]],
)

results = client.wait_for_results(central_task.get("id"))


