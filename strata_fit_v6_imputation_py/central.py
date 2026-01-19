from typing import Any, Optional, List, Dict
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.exceptions import PrivacyThresholdViolation
from .imputation_strategies.base import STRATEGY_REGISTRY
from strata_fit_v6_imputation_py.imputation_strategies.base import ImputationStrategyEnum

MINIMUM_ORGANIZATIONS = 3

@algorithm_client
def central(
    client: AlgorithmClient,
    # columns: List[str],
    # imputation_strategy: Enum,
    imputation_config: Dict[str, Any],
    organizations_to_include: Optional[List[int]] = None
) -> List[Dict[Any, Any]]:
    """central orchestration of federated imputation

    Args:
        client (AlgorithmClient): Vantage6 client object injected via the `@algorithm_client` decorator.
        columns (List[str]): Data columns for which imputation metrics will be computed
        imputation_strategy (Enum): The imputation method to use
        organizations_to_include (Optional[List[int]], optional): List of organization IDs to include in the computation. If not provided,
        all organizations in the collaboration will be used. Defaults to None.

    Raises:
        PrivacyThresholdViolation: If the number of organizations included is less than the minimum threshold required for privacy.

    Returns:
        List[Dict[Any, Any]]: A list of dicts with imputed local data
    """
    if not organizations_to_include:
        organizations_to_include = [org["id"] for org in client.organization.list()]

    if len(organizations_to_include) < MINIMUM_ORGANIZATIONS:
        raise PrivacyThresholdViolation(f"Minimum number of organizations not met (required: {MINIMUM_ORGANIZATIONS}).")
    
    imputation_strategy = ImputationStrategyEnum(imputation_config["strategy"])
    columns = imputation_config["parameters"]["columns"]

    node_metrics = _start_partial_and_collect_results(
        client, 
        input = {
        "method": "partial_compute",
            "kwargs": {
                "columns" : columns,
                "imputation_strategy" : imputation_strategy
            }
        },
        organizations_to_include=organizations_to_include)
    
    info("Results obtained!")

    info("Computing global metrics")
    global_metrics = STRATEGY_REGISTRY[imputation_strategy]().aggregate(node_metrics=node_metrics, columns=columns)

    return [global_metrics]

def _start_partial_and_collect_results(
    client: AlgorithmClient,
    input: Dict[str, Any],
    organizations_to_include: List[int]
) -> List[Dict]:
    info(f"""Starting partial task '{input.get('method')}' with {len(organizations_to_include)} organizations.""")
    task = client.task.create(
        input_=input,
        organizations=organizations_to_include,
    )
    info("Waiting for results...")
    results = client.wait_for_results(task_id=task["id"])
    info(f"""Results for '{input.get('method')}' received.""")
    return results
