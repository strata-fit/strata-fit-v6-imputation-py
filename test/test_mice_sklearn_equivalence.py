from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strata_fit_v6_imputation_py import run_local_imputation
from strata_fit_v6_imputation_py.imputation_strategies.mice import MiceImputer

sklearn = pytest.importorskip("sklearn")
from sklearn.experimental import enable_iterative_imputer  # noqa: E402,F401
from sklearn.impute import IterativeImputer  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

COLUMNS = ["A", "B", "C", "D", "E"]


def _split(frame: pd.DataFrame, n_sites: int):
    return [
        frame.iloc[idx].reset_index(drop=True)
        for idx in np.array_split(np.arange(len(frame)), n_sites)
    ]


def _make_sites(seed: int = 0, n: int = 450, n_sites: int = 3):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    data = np.column_stack(
        [
            2 * z[:, 0] + rng.normal(size=n),
            z[:, 0] + z[:, 1] + 0.3 * rng.normal(size=n),
            3 * z[:, 1] + rng.normal(size=n),
            z[:, 0] - z[:, 1] + 0.5 * rng.normal(size=n),
            rng.normal(size=n) + z[:, 1],
        ]
    )
    # different missingness per column so the imputation order is non-trivial
    for j, frac in enumerate([0.30, 0.05, 0.20, 0.10, 0.15]):
        data[rng.random(n) < frac, j] = np.nan
    frame = pd.DataFrame(data, columns=COLUMNS)
    return frame, _split(frame, n_sites)


def _fit(sites, max_iter):
    return run_local_imputation(
        sites,
        imputation_config={
            "schema_version": 1,
            "strategy": "mice",
            "parameters": {"columns": COLUMNS, "max_iter": max_iter},
        },
    )


@pytest.mark.parametrize("max_iter", [1, 3, 10])
def test_federated_mice_matches_sklearn_ols(max_iter: int) -> None:
    pooled, sites = _make_sites()
    model = _fit(sites, max_iter)

    imputer = MiceImputer()
    federated = pd.concat([imputer.impute(site.copy(), model) for site in sites])[COLUMNS]

    local = IterativeImputer(
        estimator=LinearRegression(), max_iter=max_iter, tol=0.0, random_state=0
    ).fit_transform(pooled.to_numpy())

    np.testing.assert_allclose(federated.to_numpy(), local, atol=1e-3, rtol=1e-3)


def test_state_contains_full_sequence_and_order() -> None:
    _, sites = _make_sites()
    model = _fit(sites, 4)
    state = model["state"]

    assert len(state["estimate_sequence"]) == 4 * len(COLUMNS)
    assert sorted(state["feature_order"]) == list(range(len(COLUMNS)))
    # ascending missingness: B (5%) first, A (30%) last
    assert state["feature_order"][0] == 1 and state["feature_order"][-1] == 0
    assert len(state["global_estimates"]) == len(COLUMNS)


def test_result_is_independent_of_the_site_split() -> None:
    pooled, sites3 = _make_sites(n_sites=3)
    sites4 = _split(pooled, 4)
    imputer = MiceImputer()
    out3 = pd.concat([imputer.impute(s.copy(), _fit(sites3, 3)) for s in sites3])[COLUMNS]
    out4 = pd.concat([imputer.impute(s.copy(), _fit(sites4, 3)) for s in sites4])[COLUMNS]
    np.testing.assert_allclose(out3.to_numpy(), out4.to_numpy(), atol=1e-8)


def test_legacy_models_without_sequence_still_impute() -> None:
    _, sites = _make_sites()
    model = _fit(sites, 3)
    legacy = {**model, "state": {
        "initial_means": model["state"]["initial_means"],
        "global_estimates": model["state"]["global_estimates"],
    }}
    out = MiceImputer().impute(sites[0].copy(), legacy)
    assert not out[COLUMNS].isna().any().any()