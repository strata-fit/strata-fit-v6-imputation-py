"""Compare local sklearn MICE vs. federated MICE (standalone repo, optionally the meta-algo repo).

Masked-truth design: a fraction of OBSERVED values in --cols is hidden (MCAR), imputed by every
method, and scored against the truth. Data never leaves the machine (sites are simulated in-process).

  stand : strata-fit-v6-imputation-py (this repo)      - global means + chained multi-round MICE
  meta  : strata-fit-v6-meta-algo-py (optional)         - single-round version (NaN if not importable)

--aux: auxiliary columns. Used as predictors (and imputed along the way) but never masked or scored,
       like the predictor-only variables in R mice.

Example:
  uv run python test/compare_fed_vs_local_mice.py --csv test/data_times/data.csv \
      --cols DAS28 CRP ESR --aux SJC28 TJC28 Age_diagnosis --n-sites 3 --max-iter 5 \
      --group-col pat_ID --reps 5 [--save-dir out/]
"""
import argparse, json, pathlib, warnings
import numpy as np, pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LinearRegression

warnings.filterwarnings("ignore")

try:
    from strata_fit_v6_meta_algo_py.imputation import MiceImputer as MetaMice
except Exception:  # meta-algo repo not installed -> meta columns stay NaN
    MetaMice = None
from strata_fit_v6_imputation_py import run_local_imputation
from strata_fit_v6_imputation_py.imputation_strategies.mice import MiceImputer as StandMice


# ----------------------------------------------------------------------------- helpers
def split_sites(df, n, seed, by=None):
    rng = np.random.default_rng(seed)
    if by:
        keys = df[by].unique(); rng.shuffle(keys)
        return [df[df[by].isin(k)].reset_index(drop=True) for k in np.array_split(keys, n)]
    return [df.iloc[i].reset_index(drop=True) for i in np.array_split(rng.permutation(len(df)), n)]


def fit_stand(sites, cols, max_iter):
    return run_local_imputation(sites, imputation_config={
        "schema_version": 1, "strategy": "mice",
        "parameters": {"columns": cols, "max_iter": max_iter}})


def fed_stand(sites, cols, max_iter):
    cfg, m = fit_stand(sites, cols, max_iter), StandMice()
    return [pd.DataFrame(m.impute(s.copy(), cfg)) for s in sites]


def fed_meta(sites, cols):
    m = MetaMice()
    glob = m.aggregate([m.compute(s, cols) for s in sites], cols)
    return [pd.DataFrame(m.impute(s.copy(), glob)) for s in sites]


def local_mice(df, cols, est, max_iter, seed=0):
    return IterativeImputer(estimator=est, max_iter=max_iter, random_state=seed).fit_transform(
        df[cols].to_numpy(float))


def reassemble(parts, rid, cols):
    return pd.concat(parts).set_index("_rid").loc[rid][cols].to_numpy(float)


def nrmse_cols(pred, truth, mask):
    """Per-column RMSE / SD(truth); NaN for columns with nothing hidden (e.g. aux)."""
    out = np.full(pred.shape[1], np.nan)
    for j in range(pred.shape[1]):
        m = mask[:, j]
        if m.any():
            out[j] = np.sqrt(np.mean((pred[m, j] - truth[m, j]) ** 2)) / np.nanstd(truth[:, j])
    return out


# ----------------------------------------------------------------------------- experiment
def run(df, cols, aux, n_sites, frac, reps, seed, by, max_iter):
    allc = cols + aux
    score_idx = np.arange(len(cols))  # aux columns come last and are never masked
    per_rep, per_col = [], []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        truth = df[allc].to_numpy(float)
        hide = np.zeros_like(truth, dtype=bool)
        hide[:, score_idx] = ~np.isnan(truth[:, score_idx]) & (rng.random((len(df), len(cols))) < frac)
        X = truth.copy(); X[hide] = np.nan
        d = df.copy(); d[allc] = X
        sites, rid = split_sites(d, n_sites, seed + r, by), d["_rid"]

        preds = {
            "mean": np.where(np.isnan(X), np.nanmean(X, 0), X),
            "local_BayesRidge": local_mice(d, allc, BayesianRidge(), max_iter),
            "local_OLS": local_mice(d, allc, LinearRegression(), max_iter),
            "fed_standalone": reassemble(fed_stand([s.copy() for s in sites], allc, max_iter), rid, allc),
            "fed_meta": (reassemble(fed_meta([s.copy() for s in sites], allc), rid, allc)
                         if MetaMice else np.full_like(truth, np.nan)),
        }
        alt = reassemble(fed_stand(split_sites(d, n_sites + 1, seed + 99 + r, by), allc, max_iter), rid, allc)
        cols_err = {k: nrmse_cols(v, truth, hide) for k, v in preds.items()}
        row = {k: np.nanmean(v) for k, v in cols_err.items()}
        row["fed_standalone_vs_localOLS"] = np.nanmean(nrmse_cols(preds["fed_standalone"], preds["local_OLS"], hide))
        row["fed_meta_vs_localOLS"] = np.nanmean(nrmse_cols(preds["fed_meta"], preds["local_OLS"], hide))
        row["split_invariance_maxabs"] = float(np.max(np.abs(preds["fed_standalone"] - alt)))
        per_rep.append(row)
        per_col.append(pd.DataFrame(cols_err, index=allc).loc[cols])
    return pd.DataFrame(per_rep), sum(per_col) / len(per_col)


# ----------------------------------------------------------------------------- reporting
def recovery(base, method, ref):
    """% of the improvement local MICE achieves over the mean baseline that `method` also achieves."""
    gain = base - ref
    return np.nan if not np.isfinite(gain) or gain < 1e-6 else 100 * (base - method) / gain


def verdict(rec):
    if np.isnan(rec): return "n/a"
    if rec >= 95: return "essentially equivalent to local MICE"
    if rec >= 80: return "close to local MICE"
    if rec >= 50: return "usable, clearly weaker than local MICE"
    if rec >= 0:  return "barely better than mean imputation"
    return "WORSE than mean imputation"


def report(res, percol, a):
    m, sd = res.mean(), res.std(ddof=0)
    base, ref = m["mean"], m["local_OLS"]
    print("\n" + "=" * 74)
    print(f"RESULT  ({a.reps} reps, {a.n_sites} simulated sites, hidden={a.frac:.0%}, max_iter={a.max_iter}, "
          f"scored cols={len(a.cols)}, aux cols={len(a.aux)})")
    print("=" * 74)
    print("\n1) Error on hidden values (nRMSE, lower = better; mean-imputation = ~1.0)")
    print(f"   {'method':<20}{'nRMSE':>8}{'± sd':>8}   {'improvement recovered*':>24}")
    for k, label in [("mean", "mean baseline"), ("local_BayesRidge", "local sklearn (BR)"),
                     ("local_OLS", "local sklearn (OLS)"), ("fed_standalone", "federated standalone"),
                     ("fed_meta", "federated meta-algo")]:
        if np.isnan(m[k]):
            print(f"   {label:<20}{'n/a':>8}"); continue
        rec = "" if k == "mean" else f"{recovery(base, m[k], ref):>22.0f} %"
        print(f"   {label:<20}{m[k]:>8.4f}{sd[k]:>8.4f}   {rec}")
    print("   *100% = as good as local OLS MICE, 0% = no better than the column mean")

    print("\n2) Quick verdict")
    for k, label in [("fed_standalone", "standalone"), ("fed_meta", "meta-algo ")]:
        if not np.isnan(m[k]):
            rec = recovery(base, m[k], ref)
            print(f"   {label}: {rec:.0f}% recovered, gap to local = {m[k]-ref:+.4f} nRMSE -> {verdict(rec)}")
    if not np.isnan(m["fed_meta"]):
        gap = m["fed_meta"] - m["fed_standalone"]
        print(f"   standalone vs meta-algo: {gap:+.4f} nRMSE in favour of "
              f"{'standalone' if gap > 0 else 'meta-algo'}")

    print("\n3) Distance of imputed values to local MICE (nRMSE, 0 = identical)")
    print(f"   standalone: {m['fed_standalone_vs_localOLS']:.4f}"
          + (f"   meta-algo: {m['fed_meta_vs_localOLS']:.4f}" if not np.isnan(m['fed_meta_vs_localOLS']) else ""))

    print("\n4) Per-column nRMSE (mean over reps; aux columns are not scored)")
    print(percol.rename(columns={"mean": "mean", "local_BayesRidge": "loc_BR", "local_OLS": "loc_OLS",
                                 "fed_standalone": "fed_stand", "fed_meta": "fed_meta"})
          .round(3).dropna(axis=1, how="all").to_string())

    inv = res["split_invariance_maxabs"].max()
    print("\n5) Sanity checks")
    print(f"   split invariance (N vs N+1 sites), max abs diff: {inv:.2e} -> "
          f"{'OK' if inv < 1e-6 else 'PROBLEM: result depends on site split'}")
    if m["fed_standalone"] >= base:
        print("   WARNING: federated is not better than mean imputation - check columns/dtypes.")
    print("   Note: hidden cells are MCAR and add to real missingness; compare methods, not absolutes.")
    print("=" * 74)


def export(df, allc, n_sites, by, seed, max_iter, out):
    """Impute the REAL (unmasked) data; save federated model + federated/local imputed datasets."""
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    sites = split_sites(df, n_sites, seed, by)
    cfg, m = fit_stand(sites, allc, max_iter), StandMice()
    (out / "fed_model.json").write_text(json.dumps(cfg, indent=2, default=str))
    fed = (pd.concat([pd.DataFrame(m.impute(s.copy(), cfg)) for s in sites])
           .set_index("_rid").loc[df["_rid"]].reset_index(drop=True))
    loc = df.copy(); loc[allc] = local_mice(df, allc, BayesianRidge(), max_iter)
    fed.drop(columns="_rid", errors="ignore").to_csv(out / "imputed_federated.csv", index=False)
    loc.drop(columns="_rid").to_csv(out / "imputed_local_sklearn.csv", index=False)
    print(f"\nSaved to {out}/: fed_model.json, imputed_federated.csv, imputed_local_sklearn.csv "
          f"(contain patient data - do not share)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cols", nargs="+", required=True, help="columns to impute AND score")
    ap.add_argument("--aux", nargs="*", default=[], help="auxiliary predictor-only columns (not scored)")
    ap.add_argument("--n-sites", type=int, default=3); ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--reps", type=int, default=5); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=5); ap.add_argument("--group-col", default=None)
    ap.add_argument("--save-dir", default=None)
    a = ap.parse_args()
    a.aux = [c for c in a.aux if c not in a.cols]
    df = pd.read_csv(a.csv); df["_rid"] = np.arange(len(df))
    res, percol = run(df, a.cols, a.aux, a.n_sites, a.frac, a.reps, a.seed, a.group_col, a.max_iter)
    report(res, percol, a)
    if a.save_dir:
        export(df, a.cols + a.aux, a.n_sites, a.group_col, a.seed, a.max_iter, a.save_dir)