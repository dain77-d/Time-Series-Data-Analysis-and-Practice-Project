"""E7 what the particles come with: PM2.5/PM10 and co-pollutants (training 2021-22).

Korea (34 stations incl. Seoul Jung-gu and Baengnyeong): gases in ppm. China (48 stations): ug/m3, CO mg/m3.
Units differ between countries, so ratios are compared only within a country.
- Seoul PM2.5/PM10 by season and route cluster
- Seoul co-pollutants by route cluster (z-scores) and their correlation with PM2.5
- Baengnyeong (no big local source) vs Seoul: NO2 per PM2.5 as a local-emission marker
- China: SO2 and CO per PM2.5 by region and month (heating season)
- top-20 episodes from E6: NO2/PM2.5 in transported vs local episodes
Outputs: figures/e7_*.png, tables/e7_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402

c.setup(__file__)
GAS = ["pm25", "pm10", "so2", "no2", "o3", "co"]


def load(f):
    d = pd.read_parquet(c.root() / "Dataset" / f, columns=["time_kst", "station_id"] + GAS + [g + "_flag" for g in GAS])
    for g in GAS:
        d.loc[d[g + "_flag"].ne(0), g] = np.nan
    d["time_kst"] = pd.DatetimeIndex(d.time_kst).tz_convert("Asia/Seoul")
    d = d[(d.time_kst >= "2021-01-01") & (d.time_kst < "2023-01-01")]
    d["station_id"] = d.station_id.astype(str)
    return d[["time_kst", "station_id"] + GAS]


def main():
    kr = load("Korea_Air/korea_34stations_hourly_2021_2024_kst.parquet")
    cn = load("China/china_48stations_hourly_2021_2024_kst.parquet")
    seoul = kr[kr.station_id.eq("111121")].set_index("time_kst")[GAS]
    baeng = kr[kr.station_id.eq("831492")].set_index("time_kst")[GAS]
    cl = c.clusters().reindex(seoul.index, method="ffill", limit=2)
    sea = pd.Series(c.season(seoul.index), index=seoul.index)
    order = sorted(cl.dropna().unique())

    # 1 PM2.5/PM10
    ratio = (seoul.pm25 / seoul.pm10).where(seoul.pm10 >= 10).clip(upper=1.2)
    fig, axs = c.plt.subplots(1, 2, figsize=(13, 4))
    axs[0].boxplot([ratio[sea.eq(s)].dropna() for s in ["DJF", "MAM", "JJA", "SON"]], tick_labels=["DJF", "MAM", "JJA", "SON"], showfliers=False)
    axs[0].set_title("Seoul PM2.5/PM10 by season")
    axs[1].boxplot([ratio[cl.eq(k)].dropna() for k in order], tick_labels=order, showfliers=False)
    axs[1].tick_params(axis="x", rotation=30, labelsize=8)
    axs[1].set_title("Seoul PM2.5/PM10 by route cluster")
    for a in axs:
        a.set_ylabel("PM2.5 / PM10 (low = coarse dust, high = fine combustion/secondary)")
    c.save(fig, "e7_pm_ratio")
    c.table(pd.DataFrame({"season": ratio.groupby(sea).median()}).round(3), "e7_pm_ratio_by_season")

    # 2 co-pollutants by route cluster (z-scores of each gas in Seoul)
    z = (seoul - seoul.mean()) / seoul.std()
    zc = z.groupby(cl).mean().reindex(order)
    c.table(zc.round(3), "e7_seoul_gas_by_cluster")
    fig, axs = c.plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})
    im = axs[0].imshow(zc.to_numpy(float), cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
    axs[0].set_xticks(range(len(GAS)), [g.upper() for g in GAS])
    axs[0].set_yticks(range(len(order)), order)
    for (i, j), v in np.ndenumerate(zc.to_numpy(float)):
        axs[0].text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8)
    axs[0].grid(False)
    fig.colorbar(im, ax=axs[0], shrink=0.8, label="mean z-score")
    axs[0].set_title("Seoul pollutants by route cluster (2021-22)")
    corr = pd.DataFrame({s: seoul[sea.eq(s)].corr(method="spearman")["pm25"].drop("pm25") for s in ["DJF", "MAM", "JJA", "SON"]})
    c.table(corr.round(3), "e7_seoul_gas_corr_with_pm25")
    corr.plot.bar(ax=axs[1])
    axs[1].set_title("Spearman with Seoul PM2.5, by season")
    axs[1].tick_params(axis="x", rotation=0)
    c.save(fig, "e7_seoul_gases")

    # 3 Baengnyeong vs Seoul: NO2 per PM2.5 (local combustion marker), CO per PM2.5
    rows = {}
    for name, d in (("Seoul Jung-gu", seoul), ("Baengnyeong", baeng)):
        ok = d.pm25 >= 5
        rows[name] = {"NO2_ppb_per_PM25": (1000 * d.no2[ok] / d.pm25[ok]).median(), "CO_ppb_per_PM25": (1000 * d.co[ok] / d.pm25[ok]).median(),
                      "SO2_ppb_per_PM25": (1000 * d.so2[ok] / d.pm25[ok]).median(), "PM25_mean": d.pm25.mean(), "NO2_ppb_mean": 1000 * d.no2.mean()}
    bn = pd.DataFrame(rows).T
    c.table(bn.round(3), "e7_baengnyeong_vs_seoul")

    # 4 China: SO2 and CO per PM2.5 by region and month
    pm, st = c.universe()
    reg = st.region.copy()
    reg.index = reg.index.str.replace("CN:", "", regex=False)
    cn["region"] = cn.station_id.map(reg)
    cn = cn.dropna(subset=["region"])
    cn["month"] = cn.time_kst.dt.month
    ok = cn.pm25 >= 5
    cr = cn[ok].assign(so2_pm=lambda x: x.so2 / x.pm25, co_pm=lambda x: 1000 * x.co / x.pm25)
    m_so2 = cr.groupby(["region", "month"]).so2_pm.median().unstack()
    m_co = cr.groupby(["region", "month"]).co_pm.median().unstack()
    c.table(m_so2.round(3), "e7_china_so2_per_pm25_by_month")
    c.table(m_co.round(2), "e7_china_co_per_pm25_by_month")
    fig, axs = c.plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, m, t in ((axs[0], m_so2, "SO2 / PM2.5 (ug/ug)"), (axs[1], m_co, "CO / PM2.5 (ug/ug)")):
        for r, row in m.iterrows():
            ax.plot(row.index, row.values, marker=".", label=r.replace("CN_", ""))
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("month")
        ax.set_title(f"China: {t} by region")
    axs[0].legend(fontsize=7)
    fig.suptitle("Upstream gas-to-particle ratios by month (2021-22): lowest in winter-spring, highest in summer")
    c.save(fig, "e7_china_ratios")

    # 5 episodes from E6: NO2/PM2.5 in Seoul around each peak
    ep = pd.read_csv(c.EDA / "05_transport_pathways_and_events" / "tables" / "e6_top20_episodes.csv", parse_dates=["peak_time"])
    ep["peak_time"] = pd.DatetimeIndex(pd.to_datetime(ep.peak_time, utc=True)).tz_convert("Asia/Seoul")
    out = []
    for r in ep.itertuples():
        w = seoul.loc[r.peak_time - pd.Timedelta(hours=6):r.peak_time]
        b = baeng.loc[r.peak_time - pd.Timedelta(hours=18):r.peak_time - pd.Timedelta(hours=3)]
        out.append({"peak_time": r.peak_time, "type": r.type, "seoul_no2_ppb_per_pm25": (1000 * w.no2 / w.pm25).median(),
                    "seoul_pm25_pm10": (w.pm25 / w.pm10).median(), "baengnyeong_pm25_before": b.pm25.max()})
    o = pd.DataFrame(out)
    c.table(o.round(3), "e7_episode_composition", index=False)
    g = o.groupby("type")[["seoul_no2_ppb_per_pm25", "seoul_pm25_pm10", "baengnyeong_pm25_before"]].median()
    c.table(g.round(3), "e7_episode_composition_by_type")
    fig, ax = c.plt.subplots(figsize=(7, 4))
    for k, col in (("transported", "#c0392b"), ("mixed", "#7f8c8d"), ("local build-up", "#2980b9")):
        s = o[o.type.eq(k)]
        ax.scatter(s.seoul_pm25_pm10, s.seoul_no2_ppb_per_pm25, c=col, s=50, label=f"{k} (n={len(s)})")
    ax.set_xlabel("Seoul PM2.5/PM10 (6 h before peak)")
    ax.set_ylabel("Seoul NO2 (ppb) per PM2.5")
    ax.legend(fontsize=8)
    ax.set_title("Top-20 Seoul peaks: transported ones carry less local NO2 per PM2.5")
    c.save(fig, "e7_episode_composition")


if __name__ == "__main__":
    main()
