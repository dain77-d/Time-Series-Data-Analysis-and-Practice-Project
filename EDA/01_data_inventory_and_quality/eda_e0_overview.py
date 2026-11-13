"""E0 data overview: what is in the dataset, where the stations are, when they report (2021-22).

Outputs: tables/e0_inventory.csv, tables/e0_region_stations.csv,
         figures/e0_station_map.png, figures/e0_coverage_region_month.png
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402

c.setup(__file__)
from plot_map import land

R = c.root() / "Dataset"
FILES = {"Korea national PM2.5": "Korea_National_2021_2024/korea_national_pm25_hourly_2021_2024_kst.parquet",
         "Korea 34 stations (6 pollutants)": "Korea_Air/korea_34stations_hourly_2021_2024_kst.parquet",
         "China 48 stations": "China/china_48stations_hourly_2021_2024_kst.parquet",
         "China border cities": "China_Border_Proxy/china_border_cities_hourly_2021_2024_kst.parquet",
         "China east (control)": "China_East/china_east_cities_hourly_2021_2024_kst.parquet",
         "Japan west": "Japan/japan_west_hourly_2021_2024_kst.parquet",
         "Japan east (control)": "Japan_East/japan_east_hourly_2021_2024_kst.parquet",
         "Mongolia (Ulaanbaatar)": "Mongolia/mongolia_ub_embassy_hourly_2021_2024_kst.parquet",
         "ASOS Seoul 108 / Baengnyeong 102": "ASOS/asos_102_108_hourly_2021_2024_kst.parquet",
         "HYSPLIT back-trajectory points": "Trajectories/trajectory_points_2021_2024.parquet"}
COL = {"KR": "#d07a2d", "CN": "#a23b8f", "JP": "#2e7d5b", "MN": "#5b3f8f"}


def inventory():
    rows = []
    for name, f in FILES.items():
        p = R / f
        names = pq.read_schema(p).names
        tcol = "time_kst" if "time_kst" in names else "arrival_time_kst"
        idc = "station_id" if "station_id" in names else "trajectory_id"
        t = pd.read_parquet(p, columns=[tcol, idc])
        rows.append({"dataset": name, "rows": len(t), "units": t[idc].nunique(),
                     "first": str(pd.Timestamp(t[tcol].min()))[:16], "last": str(pd.Timestamp(t[tcol].max()))[:16],
                     "pollutants": ",".join(x for x in ("pm25", "pm10", "so2", "no2", "o3", "co") if x in names) or "-"})
    c.table(pd.DataFrame(rows), "e0_inventory", index=False)


def station_map(st):
    fig, ax = c.plt.subplots(figsize=(10, 7))
    land(ax)
    for cc, g in st[st.region.ne("TARGET")].groupby("country"):
        ax.scatter(g.lon, g.lat, s=9, c=COL[cc], label=f"{cc} ({len(g)})", zorder=3, lw=0)
    t = st.loc[c.TARGET]
    ax.scatter(t.lon, t.lat, marker="*", s=260, c="k", zorder=4, label="Seoul Jung-gu (target)")
    for r, g in st[st.region.ne("TARGET")].groupby("region"):
        if r.startswith("KR_") or r == "JP_WEST25":     # Korean labels would cover the dots; JP_WEST25 sits on JP
            continue
        ax.text(g.lon.mean(), g.lat.mean() + 0.6, r, fontsize=7, ha="center", zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"All {len(st) - 1} input stations by country and region (+ target)")
    c.save(fig, "e0_station_map")


def coverage(pm, st):
    tr = c.train(pm)
    avail = tr.notna().groupby(tr.index.to_period("M")).mean().T          # station x month
    avail["region"] = st.region.reindex(avail.index)
    by_r = avail.groupby("region").mean().reindex([r for r in c.REGION_ORDER + ["TARGET"]])
    n = st.region.value_counts().reindex(by_r.index)
    c.table(pd.DataFrame({"stations": n, "train_availability": tr.notna().T.groupby(st.region.reindex(tr.columns)).mean().mean(1).reindex(by_r.index)}),
            "e0_region_stations")
    fig, ax = c.plt.subplots(figsize=(12, 6))
    im = ax.imshow(by_r.to_numpy(float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(by_r)), [f"{r} (n={n[r]})" for r in by_r.index], fontsize=8)
    ax.set_xticks(range(0, by_r.shape[1], 2), [str(p) for p in by_r.columns[::2]], rotation=60, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="share of hours with valid PM2.5")
    ax.set_title("Data availability by region and month (training 2021-22)")
    c.save(fig, "e0_coverage_region_month")


if __name__ == "__main__":
    pm, st = c.universe()
    inventory()
    station_map(st)
    coverage(pm, st)
