"""
Run this script to test your algorithm locally (without building a Docker
image) using the mock client.

Run as:

    python test.py

Make sure to do so in an environment where `vantage6-algorithm-tools` is
installed. This can be done by running:

    pip install vantage6-algorithm-tools
"""
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
from strata_fit_v6_imputation_py.types import ImputationStrategyEnum
from pathlib import Path

# get path of current directory
data_directory = Path(__file__).parent
org_ids = [1,2,3]

## Mock client
client = MockAlgorithmClient(
    datasets=[
        # Data for first organization
        [{
            "database": data_directory / "data_times/alpha.csv",
            "db_type": "csv"
        }],
        # Data for second organization
        [{
            "database": data_directory / "data_times/beta.csv",
            "db_type": "csv"
        }],
        # Data for second organization
        [{
            "database": data_directory / "data_times/gamma.csv",
            "db_type": "csv"
        }]
    ],
    organization_ids=org_ids,
    module="strata_fit_v6_imputation_py"
)


# list mock organizations
organizations = client.organization.list()
print(organizations)
# org_ids = [organization["id"] for organization in organizations]
columns = ["DAS28", "CRP", "ESR", "SJC28", "TJC28"]

# Run the central method on 1 node and get the results
central_task = client.task.create(
    input_={
        "method":"central",
        "kwargs": {
            "columns" : columns,
            "organizations_to_include" : org_ids,
            "imputation_strategy" : ImputationStrategyEnum.MEAN
        }
    },
    organizations=[org_ids[0]],
)

results = client.wait_for_results(central_task.get("id"))

# import polars as pl
# for result in results[0]:
#     print(pl.DataFrame({col: list(inner_dict.values()) for col, inner_dict in result.items()}))

# for idx, result in enumerate(results):
#     print(len(result[0]))
    # print(pl.DataFrame({col: list(inner_dict.values()) for col, inner_dict in result.items()}))

# # Run the partial method for all organizations
# task = client.task.create(
#     input_={
#         "method":"partial",
#         "kwargs": {
#             # TODO add sensible values
#             "arg1": "some_value",

#         }
#     },
#     organizations=org_ids
# )
# # print(task)

# # Get the results from the task
# results = client.wait_for_results(task.get("id"))
# print(results)
