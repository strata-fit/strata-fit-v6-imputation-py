from typing import Any, Optional, List, Dict
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient
from v6_federated_core import MethodContext, dispatch_registered_method, to_v6_result

from .methods import (
    METHOD_REGISTRY,
    build_min_organization_policies,
    build_policy_context,
)

@algorithm_client
def central(
    client: AlgorithmClient,
    # columns: List[str],
    # imputation_strategy: Enum,
    imputation_config: Dict[str, Any],
    organizations_to_include: Optional[List[int]] = None
) -> Dict[Any, Any]:
    """Thin V6 adapter for the typed imputation central method."""
    resolved_organization_ids = (
        organizations_to_include
        or [org["id"] for org in client.organization.list()]
    )
    method_context = MethodContext(
        method="central",
        organization_ids=resolved_organization_ids,
        meta={"client": client},
    )
    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "central",
        {
            "imputation_config": imputation_config,
            "organizations_to_include": organizations_to_include,
        },
        context=method_context,
        policies=build_min_organization_policies(),
        policy_context=build_policy_context("central", resolved_organization_ids),
    )
    return to_v6_result(envelope)
