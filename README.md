# strata-fit-v6-imputation-py

## Update for testing

### Federated vs. local MICE comparison script

A script (test/compare_fed_vs_local_mice.py) so other sites can test the federated imputation on their own data.

Status

- The standalone MICE imputation is similar to local sklearn MICE (single imputation) but not yet identical. In my tests it was clearly better than mean imputation and gave the same output however the data was split across sites. It was slightly worse than local MICE, and the gap grew with more columns and more iterations.
- For a quick first test inside the meta-algo it may be usable, but it first needs to be ported into the meta-algo repo properly. The meta-algo currently has a simplified single-round version that is much less accurate.
- It needs more work to be robust and closer to local MICE, for example how the iterations and the column order are handled.

What the script does
It hides 20% of your observed values, imputes them, and compares the errors against the true values. It does this for local sklearn MICE (pooled) and for the federated MICE, where your data is split into simulated sites inside one process. Nothing leaves your machine.

## How to run

```python
[uv run] python test/compare_fed_vs_local_mice.py \
  --csv path/to/your_data.csv \
  --cols DAS28 CRP ESR SJC28 TJC28 ... \
  --n-sites 3 --max-iter 5 \
  --group-col pat_ID \
  [--save-dir out/] # saves both local and federated imputed datasets
```

- --cols: numeric columns to impute.
- --n-sites: number of simulated sites (at least 3).
- --max-iter: federated and local iterations.
- --group-col: keeps all rows of a patient in one simulated site. Use a center column if you have one.
- --save-dir: optional; saves the global federated model (fed_model.json) and both imputed datasets for downstream comparison (e.g. Cox). The CSVs contain patient data, so don’t share them.
---
<br>

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
