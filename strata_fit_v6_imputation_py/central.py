"""
This file contains all central algorithm functions. It is important to note
that the central method is executed on a node, just like any other method.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled).
"""
from typing import Any, Optional, List, Dict

from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.exceptions import PrivacyThresholdViolation
from .imputation_strategies.base import STRATEGY_REGISTRY
from .imputation_strategies.base import ImputationStrategyEnum
from .types import MINIMUM_ORGANIZATIONS

@algorithm_client
def central(
    client: AlgorithmClient,
    columns: List[str],
    imputation_strategy: ImputationStrategyEnum,
    organizations_to_include: Optional[List[int]] = None
) -> List[Dict[Any, Any]]:
    

    if not organizations_to_include:
        organizations_to_include = [org["id"] for org in client.organization.list()]

    if len(organizations_to_include) < MINIMUM_ORGANIZATIONS:
        raise PrivacyThresholdViolation(f"Minimum number of organizations not met (required: {MINIMUM_ORGANIZATIONS}).")

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
    
    info("Sending global metrics for imputation")
    imputed_results = _start_partial_and_collect_results(
        client,
        input = {
        "method" : "partial_impute",
            "kwargs" : {
                "imputation_strategy" : imputation_strategy,
                "global_metrics" : global_metrics
            }
        },
        organizations_to_include = organizations_to_include)
    
    info("Imputation done!")

    return imputed_results

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
