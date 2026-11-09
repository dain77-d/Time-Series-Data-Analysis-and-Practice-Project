"""Data layer for model C (GRU / GCN-GRU) on the team's 2026-11-07 contract + the 11-07 region additions.

Node universe = Seoul Jung-gu target (node 0) + every station any condition can use:
  team regional registry (KR 514, CN 61, JP balanced 24, MN 1) + China_East 16 + Japan_East 18 + Japan west kept 25.
Conditions = the team's 26 (Model/Common/Regional_Experiments_20261107/experiments.json) + 3 control conditions
  CTRL_CN_EAST / CTRL_JP_EAST / CTRL_JP_WEST25 = DN + that region (built from Dataset_11_07_regions_v5).
Values are raw hourly PM2.5 (flag != 0 -> NaN), no imputation. Scaling statistics come from 2021-22 only (models).

Data root: env TSA_DATA or --data; it is the folder that holds Dataset/ and Model/ (the unzipped v5 zip).
Build cache: python -I cdata.py --data <root>  ->  <repo>/work/c_universe.npz
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "work"
RUNS = Path(os.environ.get("C_RUNS", REPO / "results" / "C" / "runs"))   # override for tests
TARGET = "KR:111121"
HORIZONS = (3, 6, 12, 24)
DIST_SIGMA_KM, DIST_CUTOFF_KM = 182.34497839213313, 500.0   # team protocol: fixed, not re-estimated
DATA_VERSION = "C-v5-regions-1107+regional-pm25-inputs-v1.0.0"
HOURS = pd.date_range("2020-12-29 00:00", "2025-01-01 00:00", freq="h", tz="Asia/Seoul")
SOURCES = {  # dataset folder -> (country prefix, main parquet, buffer parquet)
    "Korea_National_2021_2024": ("KR", "korea_national_pm25_hourly_2021_2024_kst.parquet", "korea_national_pm25_context_buffer.parquet"),
    "Korea_Air": ("KR", "korea_34stations_hourly_2021_2024_kst.parquet", "korea_context_buffer.parquet"),
    "China": ("CN", "china_48stations_hourly_2021_2024_kst.parquet", "china_48stations_context_buffer.parquet"),
    "China_Border_Proxy": ("CN", "china_border_cities_hourly_2021_2024_kst.parquet", "china_border_cities_context_buffer.parquet"),
    "China_East": ("CN", "china_east_cities_hourly_2021_2024_kst.parquet", "china_east_cities_context_buffer.parquet"),
    "Japan": ("JP", "japan_west_hourly_2021_2024_kst.parquet", "japan_west_context_buffer.parquet"),
    "Japan_East": ("JP", "japan_east_hourly_2021_2024_kst.parquet", "japan_east_context_buffer.parquet"),
    "Mongolia": ("MN", "mongolia_ub_embassy_hourly_2021_2024_kst.parquet", "mongolia_ub_embassy_context_buffer.parquet"),
}


def data_root(arg=None):
    root = Path(arg or os.environ.get("TSA_DATA", REPO.parent / "data_v5"))
    assert (root / "Dataset").exists() and (root / "Model").exists(), f"not a v5 data root: {root}"
    return root


def regional(root):
    base = root / "Model" / "Common" / "Regional_Experiments_20261107"
    exp = json.loads((base / "experiments.json").read_text(encoding="utf-8"))["conditions"]
    regions = {k: list(v["station_keys"]) for k, v in json.loads((base / "regions.json").read_text(encoding="utf-8"))["regions"].items()}
    reg = pd.read_csv(base / "station_registry.csv", dtype={"station_id": str}, encoding="utf-8-sig")
    return exp, regions, reg


def new_regions(root):
    v5 = pd.read_csv(root / "Dataset" / "Trajectory_Matching_Regions_v5" / "station_registry.csv",
                     dtype={"station_id": str}, encoding="utf-8-sig")
    v5["added_1107"] = v5.added_1107.astype(str).str.lower().eq("true")
    out = {"CN_EAST": v5.station_key[v5.added_1107 & v5.country.eq("CN")].tolist(),
           "JP_EAST": v5.station_key[v5.added_1107 & v5.country.eq("JP")].tolist(),
           "JP_WEST25": v5.station_key[~v5.added_1107 & v5.country.eq("JP")].tolist()}
    assert [len(out[k]) for k in ("CN_EAST", "JP_EAST", "JP_WEST25")] == [16, 18, 25]
    return out, v5


def conditions(root):
    """condition id -> sorted external station keys (target excluded); regions incl. the 3 new ones."""
    exp, regions, _ = regional(root)
    extra, _ = new_regions(root)
    conds = {k: sorted(v["station_keys"]) for k, v in exp.items()}
    for r, keys in extra.items():
        conds[f"CTRL_{r}"] = sorted(set(conds["DN"]) | set(keys))
    return conds, {**regions, **extra}


def build_universe(root):
    exp, regions, reg = regional(root)
    extra, v5 = new_regions(root)
    conds, _ = conditions(root)
    keys = sorted(set().union(*conds.values()))
    assert TARGET not in keys
    coord = pd.concat([reg[["station_key", "longitude", "latitude"]], v5[["station_key", "longitude", "latitude"]]]).drop_duplicates("station_key")
    coord = coord.set_index("station_key")
    tgt = pd.read_csv(root / "Dataset" / "Korea_Air" / "station_metadata.csv", dtype={"station_id": str}, encoding="utf-8-sig")
    tgt = tgt[tgt.station_id.eq("111121")].iloc[0]
    all_keys = [TARGET] + keys
    lon = np.r_[float(tgt.longitude), coord.loc[keys, "longitude"].to_numpy(float)]
    lat = np.r_[float(tgt.latitude), coord.loc[keys, "latitude"].to_numpy(float)]
    pm = np.full((len(HOURS), len(all_keys)), np.nan, np.float32)
    col = {k: i for i, k in enumerate(all_keys)}
    filled = np.zeros(len(all_keys), bool)
    for ds, (cc, main, buf) in SOURCES.items():
        for f in (main, buf):
            t = pd.read_parquet(root / "Dataset" / ds / f, columns=["time_kst", "station_id", "pm25", "pm25_flag"])
            t["key"] = cc + ":" + t.station_id.astype(str)
            if ds == "Korea_Air":            # target only, from the same source as the evaluation grid
                t = t[t.key.eq(TARGET)]
            elif cc == "KR":
                t = t[t.key.ne(TARGET)]
            t = t[t.key.isin(col)]
            if not len(t):
                continue
            ti = HOURS.get_indexer(pd.DatetimeIndex(t.time_kst))
            ci = t.key.map(col).to_numpy()
            ok = (ti >= 0) & t.pm25_flag.eq(0).to_numpy() & np.isfinite(t.pm25.to_numpy(float))
            pm[ti[ok], ci[ok]] = t.pm25.to_numpy(np.float32)[ok]
            filled[np.unique(ci)] = True
    assert filled.all(), [k for k, f in zip(all_keys, filled) if not f][:5]
    glob, gnames = asos(root)
    np.savez_compressed(WORK / "c_universe.npz", keys=np.array(all_keys), lon=lon, lat=lat, pm25=pm,
                        hours_utc=((HOURS - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)).to_numpy(), globals=glob, global_names=np.array(gnames))
    return {"nodes": len(all_keys), "conditions": len(conds), "missing_share_2021_2022": float(np.isnan(pm[(HOURS.year >= 2021) & (HOURS.year <= 2022)]).mean()),
            "target_missing_2021_2022": float(np.isnan(pm[(HOURS.year >= 2021) & (HOURS.year <= 2022), 0]).mean())}


def asos(root):
    """ASOS 108/102 hourly temperature, humidity, pressure, precipitation, wind u/v (calm -> 0, bad dir -> NaN)."""
    d = root / "Dataset" / "ASOS"
    a = pd.concat([pd.read_parquet(d / f) for f in ("asos_102_108_hourly_2021_2024_kst.parquet", "asos_context_after.parquet")])
    out = {}
    for st in ("108", "102"):
        g = a[a.station_id.astype(str).eq(st)].drop_duplicates("time_kst").set_index("time_kst").sort_index()
        spd, dr = g.wind_speed, g.wind_direction
        good = spd.gt(0) & dr.gt(0) & dr.le(360)
        u = (-spd * np.sin(np.deg2rad(dr))).where(good).mask(spd.eq(0), 0.0)
        v = (-spd * np.cos(np.deg2rad(dr))).where(good).mask(spd.eq(0), 0.0)
        for name, s in (("temperature", g.temperature), ("rh", g.relative_humidity), ("pressure", g.surface_pressure),
                        ("precip", g.precipitation), ("wind_u", u), ("wind_v", v)):
            out[f"asos{st}_{name}"] = s
    f = pd.DataFrame(out)
    f.index = pd.DatetimeIndex(f.index).tz_convert("Asia/Seoul")
    f = f.reindex(HOURS)
    return f.to_numpy(np.float32), list(f.columns)


def distance_matrix(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    a = np.sin((la[:, None] - la[None]) / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None]) * np.sin((lo[:, None] - lo[None]) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def distance_graph(lon, lat):
    d = distance_matrix(lon, lat)
    return np.where((d <= DIST_CUTOFF_KM) & ~np.eye(len(d), dtype=bool), np.exp(-(d / DIST_SIGMA_KM) ** 2), 0.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data")
    WORK.mkdir(exist_ok=True)
    print(json.dumps(build_universe(data_root(ap.parse_args().data)), indent=1))
