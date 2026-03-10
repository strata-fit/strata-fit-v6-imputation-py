from typing import Any, Optional, List, Dict
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.exceptions import PrivacyThresholdViolation
from .imputation_strategies.base import STRATEGY_REGISTRY
from strata_fit_v6_imputation_py.imputation_strategies.base import ImputationStrategyEnum
from .utils import build_imputation_model_config

MINIMUM_ORGANIZATIONS = 3

@algorithm_client
def central(
    client: AlgorithmClient,
    # columns: List[str],
    # imputation_strategy: Enum,
    imputation_config: Dict[str, Any],
    organizations_to_include: Optional[List[int]] = None
) -> Dict[Any, Any]:
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

    # --- ROUND 0: Global Mean Calculation ---
    info("Round 0: Calculating Global Means for initialization...")
    
    sum_results = _start_partial_and_collect_results(
        client,
        input={
            "method": "get_local_sums",
            "kwargs": {"columns": columns}
        },
        organizations_to_include=organizations_to_include
    )

    global_means = {}
    for col in columns:
        total_sum = sum(node[col]["sum"] for node in sum_results)
        total_count = sum(node[col]["count"] for node in sum_results)
        
        if total_count > 0:
            global_means[col] = total_sum / total_count
        else:
            global_means[col] = 0.0
            
    info(f"Global Means calculated: {global_means}")

    # --- START ITERATIONS ---
    # Store the means in our state dictionary so they can be passed to the nodes
    global_metrics = {"initial_means": global_means}

    # node_metrics = _start_partial_and_collect_results(
    #     client, 
    #     input = {
    #     "method": "partial_compute",
    #         "kwargs": {
    #             "columns" : columns,
    #             "imputation_strategy" : imputation_strategy
    #         }
    #     },
    #     organizations_to_include=organizations_to_include)
    
    # info("Results obtained!")

    # info("Computing global metrics")
    # global_metrics = STRATEGY_REGISTRY[imputation_strategy]().aggregate(node_metrics=node_metrics, columns=columns)

    # n_orgs = len(node_metrics)

    # imputation_model_config = build_imputation_model_config(
    #     strategy=imputation_strategy.value,
    #     parameters={
    #         "columns" : columns
    #     },
    #     state=global_metrics,
    #     n_organizations=n_orgs
    # )



    max_rounds = imputation_config["parameters"].get("max_iter", 5)

    for round_num in range(max_rounds):
        info(f"--- Starting Round {round_num + 1}/{max_rounds} ---")
        
        node_results = _start_partial_and_collect_results(
            client, 
            input={
                "method": "partial_compute",
                "kwargs": {
                    "columns": columns,
                    "imputation_strategy": imputation_strategy,
                    "global_state": global_metrics # Pass the previous round's results
                }
            },
            organizations_to_include=organizations_to_include
        )
        
        # Aggregate local models into a new global model
        global_metrics = STRATEGY_REGISTRY[imputation_strategy]().aggregate(
            node_metrics=node_results, 
            columns=columns,
            # global_means=global_metrics
        )


    imputation_model_config = build_imputation_model_config(
        strategy=imputation_strategy.value,
        parameters=imputation_config["parameters"],
        state=global_metrics,
        n_organizations=len(organizations_to_include)
    )


    return imputation_model_config

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
