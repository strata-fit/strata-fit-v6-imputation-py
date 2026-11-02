"""Compare local sklearn MICE vs. federated MICE from two repos.

  meta   : strata-fit-v6-meta-algo-py (branch discrepancy-analysis-202610)  -> single round, no iteration
  stand  : strata-fit-v6-imputation-py (main) via run_local_imputation       -> multi-round, chained

Needs on PYTHONPATH (or pip -e): both repos + v6_federated_core.
Masked-truth design: hide a fraction of observed values (MCAR), impute, score vs truth.

  python compare_fed_vs_local_mice.py --csv my.csv --cols DAS28 CRP ESR --n-sites 3 --max-iter 5
"""
import argparse, warnings
import numpy as np, pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LinearRegression
import json, pathlib
warnings.filterwarnings("ignore")

# replace: from strata_fit_v6_meta_algo_py.imputation import MiceImputer as MetaMice
try:
    from strata_fit_v6_meta_algo_py.imputation import MiceImputer as MetaMice
except ImportError:
    MetaMice = None
from strata_fit_v6_imputation_py import run_local_imputation
from strata_fit_v6_imputation_py.imputation_strategies.mice import MiceImputer as StandMice

def export(df, cols, n_sites, by, seed, max_iter, out):
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    sites = split_sites(df, n_sites, seed, by)
    cfg = run_local_imputation(sites, imputation_config={
        "schema_version": 1, "strategy": "mice",
        "parameters": {"columns": cols, "max_iter": max_iter}})
    (out / "fed_model.json").write_text(json.dumps(cfg, indent=2, default=str))  # portable global model
    m = StandMice()
    fed = (pd.concat([pd.DataFrame(m.impute(s.copy(), cfg)) for s in sites])
             .set_index("_rid").loc[df["_rid"]].reset_index(drop=True))
    loc = df.drop(columns="_rid").copy()
    loc[cols] = local_mice(df, cols, BayesianRidge(), max_iter)
    fed.drop(columns="_rid", errors="ignore").to_csv(out / "imputed_federated.csv", index=False)
    loc.to_csv(out / "imputed_local_sklearn.csv", index=False)

def split_sites(df, n, seed, by=None):
    rng = np.random.default_rng(seed)
    if by:
        keys = df[by].unique(); rng.shuffle(keys)
        return [df[df[by].isin(k)].reset_index(drop=True) for k in np.array_split(keys, n)]
    return [df.iloc[i].reset_index(drop=True) for i in np.array_split(rng.permutation(len(df)), n)]


def fed_meta(sites, cols, max_iter):
    m = MetaMice()
    glob = m.aggregate([m.compute(s, cols) for s in sites], cols)
    return [pd.DataFrame(m.impute(s.copy(), glob)) for s in sites]


def fed_stand(sites, cols, max_iter):
    cfg = run_local_imputation(
        sites, imputation_config={"schema_version": 1, "strategy": "mice",
                                  "parameters": {"columns": cols, "max_iter": max_iter}})
    m = StandMice()
    return [pd.DataFrame(m.impute(s.copy(), cfg)) for s in sites]


def local_mice(df, cols, est, max_iter, seed=0):
    imp = IterativeImputer(estimator=est, max_iter=max_iter, random_state=seed)
    return imp.fit_transform(df[cols].to_numpy(float))


def reassemble(parts, rid, cols):
    return pd.concat(parts).set_index("_rid").loc[rid][cols].to_numpy(float)


def nrmse(a, b, mask):  # per-column RMSE normalised by column SD of truth, averaged
    out = []
    for j in range(a.shape[1]):
        m = mask[:, j]
        if m.any(): out.append(np.sqrt(np.mean((a[m, j] - b[m, j]) ** 2)) / np.nanstd(b[:, j]))
    return float(np.mean(out))


def run(df, cols, n_sites, frac, reps, seed, by, max_iter):
    rows = []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        truth = df[cols].to_numpy(float)
        hide = ~np.isnan(truth) & (rng.random(truth.shape) < frac)
        X = truth.copy(); X[hide] = np.nan
        d = df.copy(); d[cols] = X
        sites = split_sites(d, n_sites, seed + r, by)
        rid = d["_rid"]
        meta = reassemble(fed_meta([s.copy() for s in sites], cols, max_iter), rid, cols) if MetaMice else np.full_like(truth, np.nan)
        stand = reassemble(fed_stand([s.copy() for s in sites], cols, max_iter), rid, cols)
        stand1 = reassemble(fed_stand(split_sites(d, n_sites + 1, seed + 99 + r, by), cols, max_iter), rid, cols)  # different split (min 3 orgs enforced)
        loc_br = local_mice(d, cols, BayesianRidge(), max_iter)
        loc_ols = local_mice(d, cols, LinearRegression(), max_iter)
        mean = np.where(np.isnan(X), np.nanmean(X, 0), X)
        rows.append({
            "nRMSE_mean_baseline": nrmse(mean, truth, hide),
            "nRMSE_local_BayesRidge": nrmse(loc_br, truth, hide),
            "nRMSE_local_OLS": nrmse(loc_ols, truth, hide),
            "nRMSE_fed_meta": nrmse(meta, truth, hide),
            "nRMSE_fed_standalone": nrmse(stand, truth, hide),
            "diff_stand_vs_localOLS(nRMSE)": nrmse(stand, loc_ols, hide),
            "diff_meta_vs_localOLS(nRMSE)": nrmse(meta, loc_ols, hide),
            "maxabs_stand_split_N_vs_N+1": float(np.max(np.abs(stand - stand1))),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--cols", nargs="+", required=True)
    ap.add_argument("--n-sites", type=int, default=3); ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--reps", type=int, default=5); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=5); ap.add_argument("--group-col", default=None)
    ap.add_argument("--save-dir", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.csv); df["_rid"] = np.arange(len(df))
    res = run(df, a.cols, a.n_sites, a.frac, a.reps, a.seed, a.group_col, a.max_iter)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(res.mean().round(4).to_string())

    if a.save_dir:
        export(df, a.cols, a.n_sites, a.group_col, a.seed, a.max_iter, a.save_dir)
