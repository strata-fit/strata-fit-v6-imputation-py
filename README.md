# strata-fit-v6-imputation-py

Standalone federated imputation algorithm for STRATA-FIT v6.

## Runtime

- Base image: `python:3.11-slim`
- Container entrypoint: `python -m strata_fit_v6_imputation_py.container`
- Runtime contract: `RUN_CONTEXT_FILE`
- Public methods:
  - `central`
  - `partial_compute`
  - `get_local_sums`

This repo no longer depends on `vantage6-algorithm-tools`, Harbor `algorithm-base`, or `polars`.

## Supported strategies

- `mean`
- `mice`

The strategy registry is lazy-loaded so importing the package root does not eagerly import every strategy module.

## Install

```bash
python -m pip install -e .[dev]
```

## Example task input

```python
input_ = {
    "method": "central",
    "kwargs": {
        "organizations_to_include": [1, 2, 3],
        "imputation_config": {
            "schema_version": 1,
            "strategy": "mean",
            "parameters": {
                "columns": ["DAS28", "CRP", "ESR"],
            },
        },
    },
}
```

For MICE:

```python
"imputation_config": {
    "schema_version": 1,
    "strategy": "mice",
    "parameters": {
        "columns": ["DAS28", "CRP", "ESR"],
        "max_iter": 3,
    },
}
```

## Local development

Pure local execution is available through `run_local_imputation(...)`.

Repo-local verification:

```bash
/tmp/strata-imputation-verify/bin/python -m pytest test/test.py -q
```

Equivalent package checks should verify:

- package root import is safe
- mean strategy works without `polars`
- MICE central orchestration still returns the expected config payload
- `run_context` partial execution writes JSON output correctly
