from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .io import normalize_payload, write_output
from .local_client import run_local_imputation as _run_local_imputation
from .runtime import run_context
from .service import run_central_method


@run_context(
    output_uris="output_path",
    named_arguments=["imputation_config", "organizations_to_include"],
)
def central(
    *,
    imputation_config: Dict[str, Any],
    organizations_to_include: Optional[List[int]] = None,
    output_path: str | Path | None = None,
    client: Any = None,
) -> Dict[str, Any]:
    if client is None:
        from .client import AlgorithmProxyClient

        resolved_client = AlgorithmProxyClient.from_env()
    else:
        resolved_client = client
    resolved_org_ids = organizations_to_include or [
        org["id"] for org in resolved_client.organization.list()
    ]
    result = normalize_payload(
        run_central_method(
            raw_input={
                "imputation_config": imputation_config,
                "organizations_to_include": organizations_to_include,
            },
            client=resolved_client,
            organization_ids=resolved_org_ids,
        )
    )
    write_output(output_path, result)
    return result


def run_local_imputation(
    datasets: list[pd.DataFrame],
    **kwargs: Any,
) -> Dict[str, Any]:
    return _run_local_imputation(datasets, **kwargs)
