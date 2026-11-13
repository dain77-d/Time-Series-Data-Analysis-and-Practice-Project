"""E1 data quality on the raw station files (training 2021-22): flags, gaps, outliers, stuck values, negatives, ASOS QC.

Outputs: tables/e1_flags.csv, e1_gaps_by_region.csv, e1_extremes.csv, e1_outlier_summary.csv, e1_asos_qc.csv
         figures/e1_flag_share.png, e1_longest_gap_by_region.png, e1_extreme_dates.png
"""
import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402

c.setup(__file__)

R = c.root() / "Dataset"
FILES = {"KR national": ("Korea_National_2021_2024/korea_national_pm25_hourly_2021_2024_kst.parquet", "KR"),
         "KR 34 (6 pollutants)": ("Korea_Air/korea_34stations_hourly_2021_2024_kst.parquet", "KR"),
         "China 48": ("China/china_48stations_hourly_2021_2024_kst.parquet", "CN"),
         "China border": ("China_Border_Proxy/china_border_cities_hourly_2021_2024_kst.parquet", "CN"),
         "China east": ("China_East/china_east_cities_hourly_2021_2024_kst.parquet", "CN"),
         "Japan west": ("Japan/japan_west_hourly_2021_2024_kst.parquet", "JP"),
         "Japan east": ("Japan_East/japan_east_hourly_2021_2024_kst.parquet", "JP"),
         "Mongolia": ("Mongolia/mongolia_ub_embassy_hourly_2021_2024_kst.parquet", "MN")}
FLAG = {0: "valid", 1: "blank in source", 2: "row absent", 3: "column absent", 5: "negative", 6: "non-numeric",
        7: "non-finite", 8: "duplicate conflict", 11: "NIES 9999 missing", 12: "NIES not measured", 13: "NIES error"}


def load(f):
    names = pd.read_parquet(R / f, columns=None).columns if False else None   # noqa: F841 (schema read below)
    import pyarrow.parquet as pq
    cols = [x for x in ("time_kst", "station_id", "pm25", "pm25_flag", "pm10", "pm25_negative_raw") if x in pq.read_schema(R / f).names]
    d = pd.read_parquet(R / f, columns=cols)
    d["time_kst"] = pd.DatetimeIndex(d.time_kst).tz_convert("Asia/Seoul")
    return d[(d.time_kst >= "2021-01-01") & (d.time_kst < "2023-01-01")]


def runs(s):
    """hours that sit in a run of >= 24 identical valid values (per station, time-sorted)."""
    r = (s != s.shift()).cumsum()
    n = s.groupby(r).transform("size")
    return int(((n >= 24) & s.notna()).sum())


def main():
    flag_rows, out_rows, ext = [], [], []
    for name, (f, cc) in FILES.items():
        d = load(f)
        fl = d.pm25_flag.map(FLAG).fillna("other").value_counts(normalize=True)
        flag_rows.append(fl.rename(name))
        d = d.sort_values(["station_id", "time_kst"])
        v = d.pm25
        stuck = sum(runs(g.pm25) for _, g in d.groupby("station_id"))
        spikes = 0
        for _, g in d.groupby("station_id"):
            a = g.pm25.to_numpy()
            if len(a) > 2:
                spikes += int(np.nansum((a[1:-1] - np.fmax(a[:-2], a[2:])) > 150))
        row = {"dataset": name, "cells": len(d), "missing_pct": round(100 * v.isna().mean(), 2), "max": v.max(),
               "p99_9": v.quantile(0.999), "n_gt_300": int((v > 300).sum()), "n_gt_500": int((v > 500).sum()),
               "zero_values": int((v == 0).sum()), "stuck_24h_hours": stuck, "isolated_spikes_gt150": spikes}
        if "pm10" in d:
            row["pm25_gt_pm10"] = int((d.pm25 > d.pm10 + 5).sum())
        if "pm25_negative_raw" in d:
            row["negative_raw_hours"] = int(d.pm25_negative_raw.notna().sum())
        out_rows.append(row)
        top = d[v > 300].assign(dataset=name)
        ext.append(top[["dataset", "station_id", "time_kst", "pm25"] + (["pm10"] if "pm10" in d else [])])
        print("done", name)
    flags = pd.DataFrame(flag_rows).fillna(0)
    c.table(flags.round(4), "e1_flags")
    c.table(pd.DataFrame(out_rows), "e1_outlier_summary", index=False)
    ext = pd.concat(ext).sort_values("pm25", ascending=False)
    c.table(ext, "e1_extremes", index=False)

    fig, ax = c.plt.subplots(figsize=(10, 4.5))
    bad = flags.drop(columns="valid", errors="ignore") * 100
    bad.plot.barh(stacked=True, ax=ax, colormap="tab20")
    ax.set_xlabel("% of station-hours (not usable)")
    ax.set_title("Why PM2.5 values are missing, by dataset (2021-22)")
    ax.legend(fontsize=7, loc="lower right")
    c.save(fig, "e1_flag_share")

    dd = ext.assign(date=ext.time_kst.dt.date).groupby("date").size().sort_values(ascending=False).head(15)
    fig, ax = c.plt.subplots(figsize=(9, 4))
    dd.sort_values().plot.barh(ax=ax, color="#a23b8f")
    ax.set_xlabel("station-hours with PM2.5 > 300")
    ax.set_title("Days with extreme PM2.5 (2021-22): Lunar New Year and dust events stand out")
    c.save(fig, "e1_extreme_dates")


def gaps():
    pm, st = c.universe()
    tr = c.train(pm)
    rows = []
    for k in tr.columns:
        na = tr[k].isna().to_numpy()
        if not na.any():
            rows.append((k, 0)); continue
        edges = np.diff(np.r_[0, na.astype(int), 0])
        rows.append((k, int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())))
    g = pd.DataFrame(rows, columns=["key", "longest_gap_h"]).set_index("key").join(st.region)
    s = g.groupby("region").longest_gap_h.describe()[["count", "50%", "max"]].reindex(c.REGION_ORDER + ["TARGET"])
    c.table(s, "e1_gaps_by_region")
    fig, ax = c.plt.subplots(figsize=(10, 4.5))
    order = [r for r in c.REGION_ORDER if r in g.region.values]
    ax.boxplot([g.longest_gap_h[g.region.eq(r)].clip(lower=1) for r in order], tick_labels=order, showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel("longest gap per station (hours, log)")
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.set_title("Longest run of missing hours per station, by region (2021-22)")
    c.save(fig, "e1_longest_gap_by_region")


def asos_qc():
    a = pd.read_parquet(R / "ASOS/asos_102_108_hourly_2021_2024_kst.parquet")
    a["time_kst"] = pd.DatetimeIndex(a.time_kst).tz_convert("Asia/Seoul")
    a = a[(a.time_kst >= "2021-01-01") & (a.time_kst < "2023-01-01")]
    rows = {}
    for st, g in a.groupby("station_id"):
        for v in ("temperature", "relative_humidity", "precipitation", "surface_pressure", "wind_speed", "wind_direction"):
            rows[(st, v)] = {"missing_pct": round(100 * g[v].isna().mean(), 2), "flag_nonzero_pct": round(100 * g[f"{v}_flag"].ne(0).mean(), 2)}
    c.table(pd.DataFrame(rows).T, "e1_asos_qc")


if __name__ == "__main__":
    main()
    gaps()
    asos_qc()
