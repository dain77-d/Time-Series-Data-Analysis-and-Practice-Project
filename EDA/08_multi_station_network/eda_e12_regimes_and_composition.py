"""E12 the whole network under each airflow regime, and particle chemistry at many stations (training 2021-22).

- regime composites: every station's mean log PM2.5 anomaly when the air reaching Korea follows each route cluster (E5)
- route sensitivity: how much each station's PM2.5 depends on the regime (max - min regime mean / overall mean)
- Korea 34 stations: NO2 per PM2.5 (local combustion) and PM2.5/PM10 per station, mapped
- China 48 stations: SO2 and CO per PM2.5 per station, winter vs summer
Outputs: figures/e12_*.png, tables/e12_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402
from plot_map import land  # noqa: E402

c.setup(__file__)
GAS = ["pm25", "pm10", "so2", "no2", "co"]


def regimes(pm, st):
    tr = c.train(pm)
    lt = np.log1p(tr)
    an = lt - lt.mean()                                       # station anomaly vs its own 2-year mean (log)
    cl = c.clusters().reindex(tr.index, method="ffill", limit=2)
    order = sorted(cl.dropna().unique())
    comp = an.groupby(cl).mean().reindex(order)                # regime x station
    c.table(comp.T.join(st[["region", "lon", "lat"]]).round(3), "e12_regime_composites")
    fig, axs = c.plt.subplots(2, 3, figsize=(20, 11))
    for ax, k in zip(axs.ravel(), order):
        land(ax)
        v = comp.loc[k]
        sc = ax.scatter(st.lon.reindex(v.index), st.lat.reindex(v.index), c=v, cmap="RdBu_r", vmin=-0.6, vmax=0.6, s=11, zorder=3, lw=0)
        ax.scatter(st.loc[c.TARGET, "lon"], st.loc[c.TARGET, "lat"], marker="*", s=180, facecolor="none", edgecolor="k", zorder=4)
        ax.set_title(f"{k} ({100 * cl.eq(k).mean():.0f}% of hours)")
        fig.colorbar(sc, ax=ax, shrink=0.7, label="log PM2.5 vs station mean")
    fig.suptitle("Whole-network PM2.5 under each airflow regime (route cluster of air arriving at Seoul, 2021-22)", fontsize=13)
    c.save(fig, "e12_regime_composites")

    raw = tr.groupby(cl).mean().reindex(order)
    sens = (raw.max() - raw.min()) / tr.mean()
    s = st.join(pd.DataFrame({"route_sensitivity": sens, "highest_regime": raw.idxmax(), "lowest_regime": raw.idxmin()})).drop(c.TARGET, errors="ignore")
    c.table(s.round(3), "e12_route_sensitivity")
    r = s.groupby("region").agg(route_sensitivity=("route_sensitivity", "median"),
                                highest_regime=("highest_regime", lambda x: x.value_counts().index[0]),
                                lowest_regime=("lowest_regime", lambda x: x.value_counts().index[0]))
    c.table(r.reindex([x for x in c.REGION_ORDER if x in r.index]).round(3), "e12_route_sensitivity_by_region")
    fig, ax = c.plt.subplots(figsize=(10, 7))
    land(ax)
    sc = ax.scatter(s.lon, s.lat, c=s.route_sensitivity, cmap="viridis", s=12, vmin=0, vmax=s.route_sensitivity.quantile(0.98), zorder=3)
    fig.colorbar(sc, ax=ax, shrink=0.75, label="(highest - lowest regime mean) / overall mean")
    ax.set_title("How strongly each station's PM2.5 depends on the airflow regime")
    c.save(fig, "e12_route_sensitivity")


def load(f):
    d = pd.read_parquet(c.root() / "Dataset" / f, columns=["time_kst", "station_id"] + GAS + [g + "_flag" for g in GAS])
    for g in GAS:
        d.loc[d[g + "_flag"].ne(0), g] = np.nan
    d["time_kst"] = pd.DatetimeIndex(d.time_kst).tz_convert("Asia/Seoul")
    d = d[(d.time_kst >= "2021-01-01") & (d.time_kst < "2023-01-01")]
    d["station_id"] = d.station_id.astype(str)
    d["winter"] = d.time_kst.dt.month.isin([12, 1, 2])
    return d


def composition(st):
    R = c.root() / "Dataset"
    kr = load("Korea_Air/korea_34stations_hourly_2021_2024_kst.parquet")
    meta = pd.read_csv(R / "Korea_Air/station_metadata.csv", dtype={"station_id": str}, encoding="utf-8-sig")
    ok = kr.pm25 >= 5
    g = kr[ok].assign(no2_per_pm=1000 * kr.no2 / kr.pm25, fine=kr.pm25 / kr.pm10.where(kr.pm10 >= 10)).groupby("station_id")
    k = pd.DataFrame({"pm25_mean": kr.groupby("station_id").pm25.mean(), "no2_ppb_mean": 1000 * kr.groupby("station_id").no2.mean(),
                      "no2_ppb_per_pm25": g.no2_per_pm.median(), "pm25_pm10": g.fine.median()})
    k = k.join(meta.set_index("station_id")[["station_name", "longitude", "latitude", "network"]])
    c.table(k.round(3), "e12_korea34_composition")
    fig, axs = c.plt.subplots(1, 3, figsize=(19, 6))
    for ax, col, cmap, t in ((axs[0], "no2_ppb_per_pm25", "magma_r", "NO2 (ppb) per PM2.5: local combustion share"),
                             (axs[1], "pm25_pm10", "viridis", "PM2.5 / PM10: fine-particle share")):
        land(ax)
        sc = ax.scatter(k.longitude, k.latitude, c=k[col], cmap=cmap, s=70, zorder=3, edgecolor="k", lw=0.4)
        fig.colorbar(sc, ax=ax, shrink=0.75)
        ax.set_xlim(124, 130.5)
        ax.set_ylim(33, 39)
        ax.set_aspect(1.25)
        ax.set_title(t, fontsize=10)
    kind = k.network.astype(str).map(lambda n: "national background" if "배경" in n else "urban" if "도시" in n else "suburban / roadside / other")
    for name, col in (("national background", "#2e86c1"), ("urban", "#c0392b"), ("suburban / roadside / other", "#7f8c8d")):
        m = kind.eq(name)
        axs[2].scatter(k.no2_ppb_per_pm25[m], k.pm25_mean[m], s=45, c=col, edgecolor="k", lw=0.4, label=f"{name} ({m.sum()})")
    names = {"111121": "Seoul Jung-gu", "831492": "Baengnyeong", "831494": "Yeonpyeong", "339312": "Gosan (Jeju)",
             "336522": "Hongdo", "437541": "Taeha (Ulleungdo)"}
    for sid, r in k.iterrows():
        if sid in names:
            axs[2].annotate(names[sid], (r.no2_ppb_per_pm25, r.pm25_mean), fontsize=8, xytext=(4, 2), textcoords="offset points")
    axs[2].legend(fontsize=8)
    axs[2].set_xlabel("NO2 (ppb) per PM2.5")
    axs[2].set_ylabel("mean PM2.5")
    axs[2].set_title("Background islands: PM2.5 as high as cities with little NO2")
    fig.suptitle("Korea 34 multi-pollutant stations (2021-22)")
    c.save(fig, "e12_korea34_composition")

    cn = load("China/china_48stations_hourly_2021_2024_kst.parquet")
    ok = cn.pm25 >= 5
    cc = cn[ok].assign(so2_pm=cn.so2 / cn.pm25, co_pm=1000 * cn.co / cn.pm25)
    t = cc.groupby(["station_id", "winter"])[["so2_pm", "co_pm"]].median().unstack()
    t.columns = [f"{a}_{'winter' if w else 'other'}" for a, w in t.columns]
    loc = st[st.country.eq("CN")]
    loc.index = loc.index.str[3:]
    t = t.join(loc[["lon", "lat", "region"]])
    c.table(t.round(3), "e12_china48_composition")
    fig, axs = c.plt.subplots(1, 2, figsize=(15, 6))
    for ax, col, title in ((axs[0], "so2_pm_winter", "SO2 / PM2.5 in winter"), (axs[1], "co_pm_winter", "CO / PM2.5 in winter (ug/ug)")):
        land(ax)
        sc = ax.scatter(t.lon, t.lat, c=t[col], cmap="plasma", s=60, zorder=3, edgecolor="k", lw=0.4)
        fig.colorbar(sc, ax=ax, shrink=0.75)
        ax.set_xlim(108, 132)
        ax.set_ylim(34, 50)
        ax.set_title(title)
    fig.suptitle("China 48 stations: gas-to-particle ratios differ by station (2021-22 winter)")
    c.save(fig, "e12_china48_composition")


def main():
    pm, st = c.universe()
    regimes(pm, st)
    composition(st)


if __name__ == "__main__":
    main()
