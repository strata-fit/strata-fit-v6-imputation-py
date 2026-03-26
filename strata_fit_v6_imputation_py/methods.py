from typing import Any, Dict, List, Optional

import pandas as pd
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info

from v6_federated_core import (
    MethodContext,
    MethodRegistry,
    MethodSpec,
    MinOrganizationsPolicy,
    PartialFailureError,
    PolicyContext,
    PolicyScope,
)

from .contracts import (
    CentralInput,
    CentralOutput,
    LocalSumsInput,
    LocalSumsOutput,
    PartialComputeInput,
    PartialComputeOutput,
)
from .imputation_strategies.base import ImputationStrategyEnum, STRATEGY_REGISTRY
from .utils import build_imputation_model_config

MINIMUM_ORGANIZATIONS = 3
DEFAULT_MICE_MAX_ITER = 5


def _get_client(context: MethodContext) -> AlgorithmClient:
    client = context.meta.get("client")
    if client is None:
        raise RuntimeError("Method context is missing the AlgorithmClient")
    return client


def _get_dataframe(context: MethodContext) -> pd.DataFrame:
    df = context.meta.get("df")
    if df is None:
        raise RuntimeError("Method context is missing the dataframe")
    return df


def _ensure_partial_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    failures = [
        result
        for result in results
        if isinstance(result, dict) and result.get("ok") is False
    ]
    if failures:
        raise PartialFailureError(
            "One or more partial imputation tasks failed",
            meta={"failure_count": len(failures)},
        )
    return results


def _run_partial_task(
    client: AlgorithmClient,
    input_: Dict[str, Any],
    organization_ids: List[int],
) -> List[Dict[str, Any]]:
    info(
        f"Starting partial task '{input_.get('method')}' with "
        f"{len(organization_ids)} organizations."
    )
    task = client.task.create(
        input_=input_,
        organizations=organization_ids,
    )
    info("Waiting for results...")
    results = client.wait_for_results(task_id=task["id"])
    info(f"Results for '{input_.get('method')}' received.")
    return _ensure_partial_results(results)


def _compute_global_means(
    sum_results: List[Dict[str, Dict[str, float | int]]],
    columns: List[str],
) -> Dict[str, float]:
    global_means: Dict[str, float] = {}
    for column in columns:
        total_sum = sum(
            float(node.get(column, {}).get("sum", 0.0)) for node in sum_results
        )
        total_count = sum(
            int(node.get(column, {}).get("count", 0)) for node in sum_results
        )
        global_means[column] = (total_sum / total_count) if total_count > 0 else 0.0
    return global_means


def _build_model_parameters(
    columns: List[str],
    max_iter: Optional[int] = None,
) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {"columns": columns}
    if max_iter is not None:
        parameters["max_iter"] = max_iter
    return parameters


def central_handler(
    data: CentralInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for the central handler")

    client = _get_client(context)
    organization_ids = context.organization_ids

    imputation_strategy = data.imputation_config.strategy
    columns = data.imputation_config.parameters.columns

    if imputation_strategy == ImputationStrategyEnum.MICE_IMPUTER:
        info("Round 0: calculating global means for MICE initialization")
        sum_results = _run_partial_task(
            client,
            input_={
                "method": "get_local_sums",
                "kwargs": {
                    "columns": columns,
                },
            },
            organization_ids=organization_ids,
        )
        global_state: Dict[str, Any] = {
            "initial_means": _compute_global_means(sum_results, columns)
        }
        max_rounds = (
            data.imputation_config.parameters.max_iter
            or DEFAULT_MICE_MAX_ITER
        )
        imputer = STRATEGY_REGISTRY[imputation_strategy]()

        for round_num in range(max_rounds):
            info(f"Starting MICE round {round_num + 1}/{max_rounds}")
            node_metrics = _run_partial_task(
                client,
                input_={
                    "method": "partial_compute",
                    "kwargs": {
                        "columns": columns,
                        "imputation_strategy": imputation_strategy,
                        "global_state": global_state,
                    },
                },
                organization_ids=organization_ids,
            )
            global_state.update(
                imputer.aggregate(
                    node_metrics=node_metrics,
                    columns=columns,
                )
            )

        return build_imputation_model_config(
            strategy=imputation_strategy.value,
            parameters=_build_model_parameters(columns, max_iter=max_rounds),
            state=global_state,
            n_organizations=len(organization_ids),
            schema_version=data.imputation_config.schema_version,
        )

    node_metrics = _run_partial_task(
        client,
        input_={
            "method": "partial_compute",
            "kwargs": {
                "columns": columns,
                "imputation_strategy": imputation_strategy,
            },
        },
        organization_ids=organization_ids,
    )

    info("Results obtained!")
    info("Computing global metrics")
    global_metrics = STRATEGY_REGISTRY[imputation_strategy]().aggregate(
        node_metrics=node_metrics,
        columns=columns,
    )

    return build_imputation_model_config(
        strategy=imputation_strategy.value,
        parameters=_build_model_parameters(
            columns,
            max_iter=data.imputation_config.parameters.max_iter,
        ),
        state=global_metrics,
        n_organizations=len(node_metrics),
        schema_version=data.imputation_config.schema_version,
    )


def partial_compute_handler(
    data: PartialComputeInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for the partial handler")

    df = _get_dataframe(context)
    imputer = STRATEGY_REGISTRY[data.imputation_strategy]()
    info(f"Computing imputation metrics with strategy: {data.imputation_strategy.value}")
    return imputer.compute(df, data.columns, global_state=data.global_state)


def get_local_sums_handler(
    data: LocalSumsInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Dict[str, float | int]]:
    if context is None:
        raise RuntimeError("Method context is required for the local sums handler")

    df = _get_dataframe(context)
    results: Dict[str, Dict[str, float | int]] = {}
    for column in data.columns:
        if column in df.columns:
            series = df[column].dropna()
            results[column] = {
                "sum": float(series.sum()),
                "count": int(series.count()),
            }
        else:
            results[column] = {"sum": 0.0, "count": 0}
    return results


METHOD_REGISTRY = MethodRegistry(
    [
        MethodSpec(
            name="central",
            input_model=CentralInput,
            output_model=CentralOutput,
            handler=central_handler,
        ),
        MethodSpec(
            name="partial_compute",
            input_model=PartialComputeInput,
            output_model=PartialComputeOutput,
            handler=partial_compute_handler,
        ),
        MethodSpec(
            name="get_local_sums",
            input_model=LocalSumsInput,
            output_model=LocalSumsOutput,
            handler=get_local_sums_handler,
        ),
    ]
)


def build_policy_context(method_name: str, organization_ids: List[int]) -> PolicyContext:
    return PolicyContext(
        scope=PolicyScope.ELIGIBILITY,
        method=method_name,
        organization_count=len(organization_ids),
    )


def build_min_organization_policies() -> List[MinOrganizationsPolicy]:
    return [MinOrganizationsPolicy(minimum=MINIMUM_ORGANIZATIONS)]
