"""Model B: LightGBM on the team's 26 regional input conditions + 3 low-passage control conditions, with TreeSHAP.

Inputs come only from the team loader (Model/Common/regional_data.py, fixed allowlists, no imputation).
CTRL_CN_EAST / CTRL_JP_EAST = DN + the 11-07 sidecar columns (Regions_1107_Station_History);
CTRL_JP_WEST25 = DN + the v3 columns of the 25 kept western-Japan stations.
Per condition and horizon: log1p target; grid num_leaves {15,31} x feature_fraction {0.3,0.5}
(lr 0.03, min_data_in_leaf 50, bagging 0.8, lambda_l2 1), early stopping on 2023; best setting by 2023 RMSE
(ug/m3), then seeds 11/22/33 with that setting. Predictions on every eligible 2023/2024 row (team format).
TreeSHAP (exact, LightGBM pred_contrib) for F, DN and CTRL models on 2023+2024, aggregated per origin to
region (station owner -> region) and to region x statistic (now/lag/mean/std/coverage/change/age/missing).

Usage: python blgbm.py [--conditions F DN ...] [--jobs 16] [--threads 8] [--data ROOT]
Output: results/B/runs/LightGBM/<condition>/seed<k>/predictions.parquet, meta.json
        results/B/shap/<condition>/h<h>_seed<k>_{region,region_stat}.parquet
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "C_graph"))
import cdata  # noqa: E402

OUT = REPO / "results" / "B"
HORIZONS, SEEDS = (3, 6, 12, 24), (11, 22, 33)
BASE = dict(objective="regression", learning_rate=0.03, min_data_in_leaf=50, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=1.0, verbose=-1, max_bin=63)
GRID = [dict(num_leaves=l, feature_fraction=f) for l in (15, 31) for f in (0.3, 0.5)]
SHAP_CONDITIONS = ("F", "DN", "CTRL_CN_EAST", "CTRL_JP_EAST", "CTRL_JP_WEST25", "F_TRAJ", "DN_TRAJ")
DATA_VERSION = "B-v5-regions-1107+regional-pm25-inputs-v1.0.0"


def loader(root):
    sys.path.insert(0, str(root / "Model" / "Common"))
    from regional_data import RegionalDataLoader
    return RegionalDataLoader(project_root=root)


def extra_columns(root, condition):
    """(parquet path, columns) added on top of DN for the control conditions."""
    ds = root / "Dataset"
    sets = json.loads((ds / "Regions_1107_Station_History" / "feature_sets.json").read_text(encoding="utf-8"))["sets"]
    if condition == "CTRL_CN_EAST":
        return ds / "Regions_1107_Station_History" / "regions_1107_station_history.parquet", sets["china_east_station_history"]
    if condition == "CTRL_JP_EAST":
        return ds / "Regions_1107_Station_History" / "regions_1107_station_history.parquet", sets["japan_east_station_history"]
    v3 = ds / "Origin_Features_2021_2024_v3" / "origin_features_2021_2024_kst.parquet"
    if condition == "CTRL_JP_WEST25":
        return v3, sets["japan_west_kept25_station_history_v3_columns"]
    if condition.endswith("_TRAJ"):     # H1c: HYSPLIT trajectory weather (met_*) + route geometry (route_*), 114 numeric features
        import pyarrow.parquet as pq
        return v3, [c for c in pq.read_schema(v3).names if c.startswith(("met_", "route_")) and c != "route_history12h_sector"]
    return None, []


def base_condition(condition):
    return "DN" if condition.startswith("CTRL_") else condition.removesuffix("_TRAJ")


def load(L, root, condition, h, split):
    base = base_condition(condition)
    X, y, rows = L.load_xy(base, horizon=h, split=split, allow_test=split == "test")
    path, cols = extra_columns(root, condition)
    if cols:
        cols = [c for c in cols if c not in X.columns]
        e = pd.read_parquet(path, columns=["origin_time_kst"] + cols)
        e["origin_time_kst"] = pd.DatetimeIndex(e.origin_time_kst).tz_convert("Asia/Seoul")
        e = e.set_index("origin_time_kst").reindex(pd.DatetimeIndex(rows.origin_time_kst))
        X = pd.concat([X.reset_index(drop=True), e.reset_index(drop=True).astype("float64")], axis=1)
    return X, y.to_numpy(float), rows


def fit(params, Xtr, ytr, Xva, yva, seed, threads):
    import lightgbm as lgb
    p = {**BASE, **params, "seed": seed, "num_threads": threads, "feature_fraction_seed": seed, "bagging_seed": seed}
    dtr = lgb.Dataset(Xtr, np.log1p(ytr), free_raw_data=False)
    dva = lgb.Dataset(Xva, np.log1p(yva), reference=dtr)
    return lgb.train(p, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])


def region_map(L, root, columns):
    _, regions = cdata.conditions(root)
    owner = L.registry.set_index("name")["owner_station_key"].to_dict()
    stat = L.registry.set_index("name")["stat"].to_dict()
    group = L.registry.set_index("name")["feature_group"].to_dict()
    k2r = {k: r for r, ks in regions.items() for k in ks if not r.startswith("JP_") or r == "JP"}
    k2r.update({k: r for r in ("CN_EAST", "JP_EAST") for k in regions[r]})
    for k in regions["JP_WEST25"]:
        k2r.setdefault(k, "JP_WEST25")
    reg, st = [], []
    for c in columns:
        if c in owner and owner[c]:
            reg.append(k2r.get(owner[c], "UNMAPPED")); st.append(stat.get(c) or "other")
        elif c.startswith(("met_", "route_")):   # trajectory features (_TRAJ conditions)
            reg.append("TRAJ_weather" if c.startswith("met_") else "TRAJ_route"); st.append(c.split("_")[1])
        elif c.startswith("obs_"):          # sidecar columns: obs_<cc>_<id>_pm25_<stat>
            key = c[4:6].upper() + ":" + c[7:].rsplit("_pm25_", 1)[0]
            reg.append(k2r.get(key, "UNMAPPED")); st.append(c.rsplit("_pm25_", 1)[1])
        else:
            reg.append("COMMON_" + str(group.get(c, "other"))); st.append("common")
    return np.array(reg), np.array(st)


def shap_tables(model, X, rows, reg, st, d, h, seed):
    contrib = model.predict(X, pred_contrib=True)[:, :-1]   # log1p scale; last column = bias
    t = pd.DatetimeIndex(rows.origin_time_kst)
    by_r = pd.DataFrame(contrib).T.groupby(reg).sum().T
    by_r.insert(0, "origin_time_kst", t)
    by_r.insert(1, "split", rows.split.to_numpy())
    by_r.to_parquet(d / f"h{h}_seed{seed}_region.parquet", index=False)
    lab = np.char.add(np.char.add(reg.astype(str), "|"), st.astype(str))
    rs = pd.DataFrame(np.abs(contrib)).T.groupby(lab).sum().T.mean().rename("mean_abs_shap").reset_index()
    rs[["region", "stat"]] = rs["index"].str.split("|", expand=True)
    rs.drop(columns="index").to_parquet(d / f"h{h}_seed{seed}_region_stat.parquet", index=False)


def run_condition(root, condition, threads):
    L = loader(root)
    d = OUT / "runs" / "LightGBM" / condition
    if all((d / f"seed{s}" / "predictions.parquet").exists() for s in SEEDS):
        return condition, "skip"
    t0, preds, meta = time.time(), {s: [] for s in SEEDS}, {"condition": condition, "horizons": {}}
    for h in HORIZONS:
        Xtr, ytr, _ = load(L, root, condition, h, "train")
        Xva, yva, rva = load(L, root, condition, h, "validation")
        Xte, _, rte = load(L, root, condition, h, "test")
        scores = []
        for g in GRID:
            m = fit(g, Xtr, ytr, Xva, yva, 11, threads)
            scores.append(float(np.sqrt(np.mean((np.expm1(m.predict(Xva, num_iteration=m.best_iteration)) - yva) ** 2))))
        best = GRID[int(np.argmin(scores))]
        reg, st = region_map(L, root, list(Xtr.columns))
        meta["horizons"][h] = {"grid_val_rmse": scores, "chosen": best, "n_features": Xtr.shape[1], "best_iterations": {}}
        for s in SEEDS:
            m = fit(best, Xtr, ytr, Xva, yva, s, threads)
            meta["horizons"][h]["best_iterations"][s] = m.best_iteration
            for X, rows in ((Xva, rva), (Xte, rte)):
                preds[s].append(pd.DataFrame({"origin_time_kst": pd.DatetimeIndex(rows.origin_time_kst), "horizon": h, "model": "LightGBM",
                                              "y_hat": np.expm1(m.predict(X, num_iteration=m.best_iteration)), "split": rows.split.to_numpy(),
                                              "seed": s, "prediction_kind": "seed", "input_condition": condition, "data_version": DATA_VERSION}))
            if condition in SHAP_CONDITIONS:
                sd = OUT / "shap" / condition
                sd.mkdir(parents=True, exist_ok=True)
                shap_tables(m, pd.concat([Xva, Xte]), pd.concat([rva, rte]), reg, st, sd, h, s)
    for s in SEEDS:
        sd = d / f"seed{s}"
        sd.mkdir(parents=True, exist_ok=True)
        pd.concat(preds[s], ignore_index=True).to_parquet(sd / "predictions.parquet", index=False)
    meta.update(seconds=round(time.time() - t0, 1), base_params=BASE, grid=GRID, data_version=DATA_VERSION)
    (d / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return condition, f"{meta['seconds']}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="*")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--data")
    a = ap.parse_args()
    root = cdata.data_root(a.data)
    conds = a.conditions or list(cdata.conditions(root)[0])
    order = ["F", "DN", "D67"] + [c for c in conds if c.startswith("CTRL_")]
    conds = [c for c in order if c in conds] + [c for c in conds if c not in order]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(run_condition, root, c, a.threads) for c in conds]
        for f in as_completed(futs):
            print(*f.result(), flush=True)
