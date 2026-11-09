"""Model A: linear baseline. SARIMAX with one region at a time + Granger tests (region mean -> Seoul Jung-gu).

Data: the C hourly cache (work/c_universe.npz; target column 0 = Korea_Air KR:111121, same source as the grid).
Endog: log1p PM2.5 at Seoul. Seasonality: daily + annual Fourier (K=2 each) as exog, ARMA(p,0,q) errors
(order by 2021-22 AIC on the base model, then fixed for every region model).
  BASE      Fourier only
  REG_<r>   Fourier + region-mean log1p PM2.5 lagged L hours, L in {h, h+3, h+6, h+12} (L >= h so the exog is
            known at the origin). L per region and horizon is picked on 2023 (validation score is therefore
            optimistic for REG; 2024 is the clean comparison).
  ALL       Fourier + all regions lagged h (coefficient table with 95% CI).
Forecast for origin t: y_hat(t+h) = Z T^h a_{t|t} + x_{t+h}' beta, parameters fixed from 2021-22 (no refit).
Region mean = mean over the region's stations of log1p PM2.5, standardised on 2021-22; gaps ffilled <= 6 h, then 0.
  ponytail: 0-fill after 6 h is crude; a missing indicator would be cleaner if long gaps matter.
Granger: 2021-22 hourly, OLS of Seoul on its own p lags + region p lags (p by AIC from {1,2,3,6,12,24}),
F-test region lags = 0; all, cold/warm and west/other-route subsets; Holm over every test.

Usage: python asarimax.py [--jobs 32] [--data ROOT];  python asarimax.py --score [--split test --final];  python asarimax.py --figures
Output: results/A/runs/A_SARIMAX/<cond>/seed0/predictions.parquet, results/A/{fits,granger,lag_choice,coefficients}.csv
"""
import argparse
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "C_graph"))
import cdata  # noqa: E402

OUT = REPO / "results" / "A"
H = (3, 6, 12, 24)
ORDERS = [(1, 0, 0), (2, 0, 0), (1, 0, 1), (2, 0, 1), (3, 0, 1), (2, 0, 2)]
DATA_VERSION = "A-v5-regions-1107+c-universe"
EXT = 48


def prepare(root):
    u = np.load(cdata.WORK / "c_universe.npz", allow_pickle=True)
    kst = pd.to_datetime(u["hours_utc"], unit="h", utc=True).tz_convert("Asia/Seoul")
    lp = np.log1p(u["pm25"].astype(float))
    train = (kst.year >= 2021) & (kst.year <= 2022)
    col = {k: i for i, k in enumerate(u["keys"])}
    _, regions = cdata.conditions(root)
    reg_raw, reg = {}, {}
    for r, keys in regions.items():
        idx = [col[k] for k in keys if k in col]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = np.nanmean(lp[:, idx], axis=1)
        z = (m - np.nanmean(m[train])) / np.nanstd(m[train])
        reg_raw[r] = z
        reg[r] = pd.Series(z).ffill(limit=6).fillna(0).to_numpy()
    n = len(kst)
    t = np.arange(n + EXT, dtype=float)
    four = np.column_stack([f(2 * np.pi * k * t / P) for P in (24, 8766) for k in (1, 2) for f in (np.sin, np.cos)])
    return dict(kst=kst, y=lp[:, 0], train=train, reg=reg, reg_raw=reg_raw, four=four, n=n)


def lagged(z, L, n_ext):
    out = np.zeros(n_ext)
    out[L:L + len(z)] = z[:n_ext - L]
    return out


def exog(d, cols):
    """cols: list of (region, lag). Returns (n+EXT) x k matrix: const, Fourier, region lags."""
    n_ext = d["n"] + EXT
    X = [np.ones((n_ext, 1)), d["four"]] + [lagged(d["reg"][r], L, n_ext)[:, None] for r, L in cols]
    return np.hstack(X)


def fit(d, cols, order):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    X = exog(d, cols)
    n, tr = d["n"], d["train"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(d["y"][tr], X[:n][tr], order=order, trend="n").fit(disp=False, maxiter=300)
        full = res.apply(d["y"], X[:n])
    return res, full, X


def forecasts(full, X, origins, h):
    fr = full.filter_results
    Z, T = fr.design[:, :, 0], fr.transition[:, :, 0]
    k = X.shape[1]
    beta = np.asarray(full.params)[:k]
    a = fr.filtered_state[:, origins]
    return (Z @ np.linalg.matrix_power(T, h) @ a).ravel() + X[origins + h] @ beta


def grid_origins(root, d):
    g = pd.read_parquet(root / "Dataset" / "Evaluation_Contract_20261107" / "evaluation_grid.parquet", columns=["origin_time_kst", "split"])
    out = {}
    for s in ("validation", "test"):
        t = pd.DatetimeIndex(g.origin_time_kst[g.split.eq(s)].unique())
        p = pd.Index(d["kst"]).get_indexer(t)
        assert (p >= 0).all()
        out[s] = (t, p)
    return out


def job(args):
    """one fit; cached under results/A/cache so a crash after fitting never costs the fits again."""
    import pickle
    root, name, cols, order = args
    cache = OUT / "cache" / (name + "".join(f"_{r}{L}" for r, L in cols[:1]) + f"_{len(cols)}.pkl")
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    out = _job(root, name, cols, order)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    return out


def _job(root, name, cols, order):
    d = prepare(root)
    res, full, X = fit(d, cols, order)
    og = grid_origins(root, d)
    preds = {}
    for h in H:
        if any(L < h for _, L in cols):
            continue
        for s, (t, p) in og.items():
            preds[(h, s)] = (t, np.expm1(forecasts(full, X, p, h)).clip(min=0))
    tr = d["train"]
    ci = res.conf_int()
    coef = pd.DataFrame({"param": res.model.param_names, "coef": res.params, "lo": ci[:, 0], "hi": ci[:, 1]})
    return name, cols, preds, {"aic": res.aic, "llf": res.llf, "n_train": int(tr.sum())}, coef


def val_rmse(root, preds, h):
    g = pd.read_parquet(root / "Dataset" / "Evaluation_Contract_20261107" / "evaluation_grid.parquet")
    g = g[g.split.eq("validation") & g.eligible & g.horizon.eq(h)].set_index("origin_time_kst").y_true
    t, v = preds[(h, "validation")]
    s = pd.Series(v, index=t).reindex(g.index)
    return float(np.sqrt(np.mean((s - g) ** 2)))


def write(cond, per_h):
    rows = []
    for h, preds in per_h.items():
        for s in ("validation", "test"):
            t, v = preds[(h, s)]
            rows.append(pd.DataFrame({"origin_time_kst": t, "horizon": h, "model": "A_SARIMAX", "y_hat": v, "split": s, "seed": 0,
                                      "prediction_kind": "seed", "input_condition": cond, "data_version": DATA_VERSION}))
    p = OUT / "runs" / "A_SARIMAX" / cond / "seed0"
    p.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(p / "predictions.parquet", index=False)


def granger(root, d):
    from scipy import stats
    g = pd.read_parquet(root / "Dataset" / "Evaluation_Contract_20261107" / "evaluation_grid.parquet", columns=["origin_time_kst", "split", "cold_season", "route_group"])
    g = g[g.split.eq("train")].drop_duplicates("origin_time_kst").set_index("origin_time_kst")
    lab = g.reindex(d["kst"].floor("3h"))
    y = d["y"]
    tr = d["train"]

    def design(x, p):
        Y = np.column_stack([np.roll(y, i) for i in range(1, p + 1)])
        Xr = np.column_stack([np.roll(x, i) for i in range(1, p + 1)])
        ok = tr.copy()
        ok[:24] = False
        ok &= np.isfinite(y) & np.isfinite(Y).all(1) & np.isfinite(Xr).all(1)
        return Y, Xr, ok

    def ols(A, b):
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        e = b - A @ beta
        return beta, float(e @ e)

    rows = []
    for r, x in d["reg_raw"].items():
        aic = {}
        for p in (1, 2, 3, 6, 12, 24):
            Y, Xr, ok = design(x, p)
            A = np.column_stack([np.ones(ok.sum()), Y[ok], Xr[ok]])
            _, sse = ols(A, y[ok])
            aic[p] = ok.sum() * np.log(sse / ok.sum()) + 2 * A.shape[1]
        p = min(aic, key=aic.get)
        Y, Xr, ok0 = design(x, p)
        for sub, mask in [("all", np.ones_like(ok0)),
                          ("cold", lab.cold_season.eq("cold").to_numpy()), ("warm", lab.cold_season.ne("cold").to_numpy() & lab.cold_season.notna().to_numpy()),
                          ("route_west", lab.route_group.eq("west").to_numpy()), ("route_other", lab.route_group.ne("west").to_numpy() & lab.route_group.notna().to_numpy())]:
            ok = ok0 & mask
            nobs = int(ok.sum())
            Afull = np.column_stack([np.ones(nobs), Y[ok], Xr[ok]])
            Ares = Afull[:, :1 + p]
            bf, ssef = ols(Afull, y[ok])
            _, sser = ols(Ares, y[ok])
            df2 = nobs - Afull.shape[1]
            F = ((sser - ssef) / p) / (ssef / df2)
            c = bf[1 + p:]
            rows.append({"region": r, "subset": sub, "p_lags": p, "n": nobs, "F": F, "p_value": float(stats.f.sf(F, p, df2)),
                         "sum_region_coef": float(c.sum()), "peak_lag_h": int(np.argmax(np.abs(c)) + 1),
                         "partial_r2": (sser - ssef) / sser})
    t = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    t["p_holm"] = multipletests(t.p_value, method="holm")[1]
    return t


def main(root, jobs):
    from concurrent.futures import ProcessPoolExecutor
    OUT.mkdir(parents=True, exist_ok=True)
    d = prepare(root)
    # 1) order by AIC on BASE
    with ProcessPoolExecutor(len(ORDERS)) as ex:
        a = dict(zip(ORDERS, ex.map(_aic, [(root, o) for o in ORDERS])))
    order = min(a, key=a.get)
    print("orders AIC", {str(k): round(v, 1) for k, v in a.items()}, "->", order, flush=True)
    # 2) all fits
    lags = sorted({L for h in H for L in (h, h + 3, h + 6, h + 12)})
    tasks = [(root, "BASE", [], order)] + [(root, f"REG_{r}", [(r, L)], order) for r in d["reg"] for L in lags]
    tasks += [(root, f"ALL_h{h}", [(r, h) for r in d["reg"]], order) for h in H]
    res = {}
    with ProcessPoolExecutor(jobs) as ex:
        for name, cols, preds, info, coef in ex.map(job, tasks):
            res[(name, tuple(cols))] = (preds, info, coef)
            print("fit", name, cols[:1], round(info["aic"], 1), flush=True)
    fits = [{"name": n, "cols": json.dumps(c), **v[1]} for (n, c), v in res.items()]
    pd.DataFrame(fits).to_csv(OUT / "fits.csv", index=False)
    # 3) predictions: BASE, ALL, REG_r with L chosen on 2023
    base = next(v for (n, _), v in res.items() if n == "BASE")[0]
    write("BASE", {h: base for h in H})
    write("ALL", {h: next(v for (n, _), v in res.items() if n == f"ALL_h{h}")[0] for h in H})
    coefs = []
    for h in H:
        c = next(v for (n, _), v in res.items() if n == f"ALL_h{h}")[2]
        coefs.append(c.assign(horizon=h))
    pd.concat(coefs).to_csv(OUT / "coefficients_ALL.csv", index=False)
    choice = []
    for r in d["reg"]:
        per_h = {}
        for h in H:
            cand = {c[0][1]: v[0] for (n, c), v in res.items() if n == f"REG_{r}" and c[0][1] in (h, h + 3, h + 6, h + 12)}
            sc = {L: val_rmse(root, p, h) for L, p in cand.items()}
            L = min(sc, key=sc.get)
            per_h[h] = cand[L]
            coef = next(v[2] for (n, c), v in res.items() if n == f"REG_{r}" and c[0][1] == L)
            b = coef[coef.param.str.startswith("x")].iloc[-1]   # last exog column = the region lag
            choice.append({"region": r, "horizon": h, "lag": L, "val_rmse": sc[L], "val_rmse_base": val_rmse(root, base, h),
                           "coef": b.coef, "lo": b.lo, "hi": b.hi, **{f"rmse_L{k}": v for k, v in sc.items()}})
        write(f"REG_{r}", per_h)
    pd.DataFrame(choice).to_csv(OUT / "lag_choice.csv", index=False)
    # 4) Granger
    granger(root, d).to_csv(OUT / "granger.csv", index=False)
    print("done", flush=True)


def _aic(args):
    root, o = args
    d = prepare(root)
    X = exog(d, [])[:d["n"]]
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(d["y"][d["train"]], X[d["train"]], order=o).fit(disp=False, maxiter=300).aic


def figures():
    sys.path.insert(0, str(REPO / "B_lgbm"))
    from banalysis import heat
    lc = pd.read_csv(OUT / "lag_choice.csv")
    imp = lc.assign(v=100 * (lc.val_rmse_base - lc.val_rmse) / lc.val_rmse_base).pivot(index="region", columns="horizon", values="v")
    heat(imp.sort_values(6, ascending=False).rename(columns=lambda h: f"{h} h"),
         "A: RMSE gain vs BASE, %\n(2023, lag picked on 2023)", OUT / "a_region_improvement.png", fmt="{:.1f}", cmap="Reds")
    lag = lc.pivot(index="region", columns="horizon", values="lag").loc[imp.sort_values(6, ascending=False).index]
    lag.to_csv(OUT / "a_lag_table.csv")
    g = pd.read_csv(OUT / "granger.csv")
    p = g.assign(v=-np.log10(g.p_holm.clip(lower=1e-300))).pivot(index="region", columns="subset", values="v")
    heat(p.loc[p["all"].sort_values(ascending=False).index], "Granger region -> Seoul\n-log10 Holm p (2021-22)",
         OUT / "a_granger.png", fmt="{:.0f}", cmap="viridis")
    print("figures ->", OUT)


def score(root, split, final):
    fs = sorted((OUT / "runs").glob("A_SARIMAX/*/seed0/predictions.parquet"))
    conds = [f.parts[-3] for f in fs]
    pairs = [f"A_SARIMAX|{c}|seed|seed=0:A_SARIMAX|BASE|seed|seed=0" for c in conds if c != "BASE"]   # one deterministic fit = seed 0
    d = OUT / "scores" / split
    cmd = [sys.executable, str(root / "Model" / "Common" / "score.py"), "--predictions", *map(str, fs), "--split", split,
           "--output-dir", str(d), "--pairs", *pairs] + (["--final"] if final else [])
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(r.stdout[-2000:], r.stderr[-3000:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    root = cdata.data_root(a.data)
    if a.figures:
        figures()
    elif a.score:
        score(root, a.split, a.final)
    else:
        main(root, a.jobs)
