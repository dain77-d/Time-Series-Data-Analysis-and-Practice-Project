"""E6 transport events (training 2021-22): natural experiments and the anatomy of high-PM episodes.

- Lunar New Year (Seollal) fireworks pulse: region means around midnight, 2021 and 2022, as a distance-time plot
- Asian dust: PM10/PM2.5 ratio at Chinese stations, Baengnyeong and Seoul during the March 2021 events
- Baengnyeong (KR:831492, the westernmost island station) as a gateway: lagged correlation with Seoul by route cluster
- top 20 Seoul PM2.5 peaks: transported (upstream rose first) vs local build-up (weak wind, upstream flat)
Outputs: figures/e6_*.png, tables/e6_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402

c.setup(__file__)
BAENG = "KR:831492"
SEOLLAL = {"2021": "2021-02-12", "2022": "2022-02-01"}
CN = ["CN_BTH", "CN_SHANDONG", "CN_LIAONING", "CN_JILIN", "CN_HEILONGJIANG", "CN_INNER_MONGOLIA", "CN_EAST"]


def km(st, lon, lat):
    return np.hypot((st.lon - lon) * np.cos(np.deg2rad(37.5)), st.lat - lat) * 111.2


def anomalies(x):
    return x - x.rolling(24 * 7, center=True, min_periods=24 * 3).mean()


def seollal(pm, st):
    rm = c.region_means(pm, st, log=False)
    rm["Baengnyeong"] = pm[BAENG]
    rm["Seoul"] = pm[c.TARGET]
    dist = {r: km(st[st.region.eq(r)], 126.98, 37.57).median() for r in CN + ["KR_CHUNGCHEONG", "KR_GANGWON"]}
    dist.update({"Baengnyeong": 190.0, "Seoul": 0.0})
    order = sorted(dist, key=dist.get, reverse=True)
    fig, axs = c.plt.subplots(2, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.5, 1]})
    rows = []
    for i, (yr, day) in enumerate(SEOLLAL.items()):
        t0 = pd.Timestamp(day, tz="Asia/Seoul")             # 00 KST; Chinese midnight is 01 KST
        w = rm.loc[t0 - pd.Timedelta(hours=48):t0 + pd.Timedelta(hours=72), order]
        base = rm.loc[t0 - pd.Timedelta(hours=48):t0 - pd.Timedelta(hours=6), order].median()
        ratio = w / base
        hrs = ((w.index - t0) / pd.Timedelta(hours=1)).astype(int)
        ax = axs[i, 0]
        im = ax.imshow(ratio.T.to_numpy(float), aspect="auto", cmap="inferno", vmin=0.5, vmax=4,
                       extent=[hrs[0], hrs[-1], len(order) - 0.5, -0.5])
        ax.set_yticks(range(len(order)), [f"{r} ({dist[r]:.0f} km)" for r in order], fontsize=8)
        ax.axvline(1, c="cyan", lw=0.8, ls="--")
        ax.set_xlabel("hours from Seollal midnight (KST)")
        ax.grid(False)
        ax.set_title(f"Seollal {yr}: PM2.5 relative to the 2 days before (cyan = Chinese midnight)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax = axs[i, 1]
        for r in ["CN_LIAONING", "CN_SHANDONG", "Baengnyeong", "Seoul"]:
            ax.plot(hrs, w[r].to_numpy(), label=r)
        ax.axvline(1, c="k", lw=0.8, ls="--")
        ax.set_ylabel("PM2.5")
        ax.legend(fontsize=8)
        ax.set_title(f"Seollal {yr}: upstream to Seoul")
        for r in order:
            s = w[r]
            pk = s.loc[t0:t0 + pd.Timedelta(hours=72)]
            rows.append({"year": yr, "series": r, "km": dist[r], "baseline": base[r], "peak": pk.max(),
                         "peak_ratio": pk.max() / base[r], "peak_hour": int((pk.idxmax() - t0) / pd.Timedelta(hours=1)) if pk.notna().any() else np.nan})
    c.save(fig, "e6_seollal_pulse")
    c.table(pd.DataFrame(rows).round(2), "e6_seollal_pulse", index=False)


def dust():
    R = c.root() / "Dataset"
    kr = pd.read_parquet(R / "Korea_Air/korea_34stations_hourly_2021_2024_kst.parquet", columns=["time_kst", "station_id", "pm25", "pm10"])
    cn = pd.read_parquet(R / "China/china_48stations_hourly_2021_2024_kst.parquet", columns=["time_kst", "station_id", "pm25", "pm10"])
    for d in (kr, cn):
        d["time_kst"] = pd.DatetimeIndex(d.time_kst).tz_convert("Asia/Seoul")
    seoul = kr[kr.station_id.astype(str).eq("111121")].set_index("time_kst")
    baeng = kr[kr.station_id.astype(str).eq("831492")].set_index("time_kst")
    cnm = cn.groupby("time_kst")[["pm25", "pm10"]].median()
    # dust days in training: Seoul daily PM10 >= 100 and PM10/PM2.5 >= 3
    sd = seoul[(seoul.index >= "2021-01-01") & (seoul.index < "2023-01-01")][["pm25", "pm10"]].resample("D").mean()
    dd = sd[(sd.pm10 >= 100) & (sd.pm10 / sd.pm25 >= 3)].assign(ratio=lambda x: x.pm10 / x.pm25)
    c.table(dd.round(1), "e6_dust_days")
    fig, axs = c.plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    s, e = pd.Timestamp("2021-03-13", tz="Asia/Seoul"), pd.Timestamp("2021-03-31", tz="Asia/Seoul")
    for name, d, col in (("China 48 median", cnm, "#a23b8f"), ("Baengnyeong", baeng, "#e67e22"), ("Seoul Jung-gu", seoul, "#2c3e50")):
        x = d.loc[s:e]
        axs[0].plot(x.index, x.pm10, c=col, label=f"{name} PM10")
        axs[0].plot(x.index, x.pm25, c=col, ls=":", label=f"{name} PM2.5")
        axs[1].plot(x.index, (x.pm10 / x.pm25).clip(upper=15), c=col, label=name)
    axs[0].set_yscale("log")
    axs[0].legend(fontsize=7, ncol=3)
    axs[0].set_title("Asian dust, March 2021: PM10 (solid) and PM2.5 (dotted)")
    axs[1].axhline(3, c="grey", ls="--", lw=0.8)
    axs[1].set_ylabel("PM10 / PM2.5")
    axs[1].legend(fontsize=8)
    axs[1].set_title("Coarse-particle ratio: dust shows up as PM10/PM2.5 >> 2, first in China, then Baengnyeong, then Seoul")
    c.save(fig, "e6_dust_march2021")


def gateway(pm):
    y = np.log1p(c.train(pm[c.TARGET]))
    b = np.log1p(c.train(pm[BAENG]))
    ya, ba = anomalies(y), anomalies(b)
    cl = c.clusters().reindex(y.index, method="ffill", limit=2)
    rows, curves = [], {}
    for k in ["all"] + sorted(cl.dropna().unique()):
        m = pd.Series(True, index=y.index) if k == "all" else cl.eq(k)
        cc = pd.Series({L: ba[m].corr(ya.shift(-L)[m]) for L in range(-6, 25)})
        curves[k] = cc
        rows.append({"cluster": k, "hours": int(m.sum()), "peak_lag_h": int(cc.idxmax()), "peak_corr": cc.max(), "corr_lag0": cc[0]})
    c.table(pd.DataFrame(rows).round(3), "e6_baengnyeong_gateway", index=False)
    fig, ax = c.plt.subplots(figsize=(10, 4.5))
    for k, cc in curves.items():
        ax.plot(cc.index, cc.values, lw=2.2 if k == "all" else 1.2, c="k" if k == "all" else None, label=k)
    ax.axvline(0, c="grey", lw=0.7)
    ax.set_xlabel("lag (h): Baengnyeong at t vs Seoul at t + lag")
    ax.set_ylabel("anomaly correlation")
    ax.legend(fontsize=8)
    ax.set_title("Baengnyeong leads Seoul on westerly routes (route cluster of the arriving trajectory, 2021-22)")
    c.save(fig, "e6_baengnyeong_gateway")


def episodes(pm, st):
    y = c.train(pm[c.TARGET]).astype(float)
    b = c.train(pm[BAENG]).astype(float)
    cn = c.region_means(c.train(pm), st, log=False)[CN].mean(1)
    a = c.train(c.asos())
    spd = np.hypot(a.asos108_wind_u, a.asos108_wind_v)
    cl = c.clusters().reindex(y.index, method="ffill", limit=2)
    peaks = []
    for t in y.sort_values(ascending=False).dropna().index:
        if all(abs((t - p) / pd.Timedelta(hours=1)) > 48 for p in peaks):
            peaks.append(t)
        if len(peaks) == 20:
            break
    rows = []
    for t in peaks:
        pre = slice(t - pd.Timedelta(hours=24), t - pd.Timedelta(hours=6))
        base = slice(t - pd.Timedelta(hours=72), t - pd.Timedelta(hours=48))
        b_rise = b[pre].max() / b[base].median()
        cn_rise = cn[slice(t - pd.Timedelta(hours=36), t - pd.Timedelta(hours=12))].max() / cn[base].median()
        w = spd[t - pd.Timedelta(hours=12):t].mean()
        # ponytail: simple rule on the Baengnyeong rise and local wind; a trajectory-weighted upstream index would be sharper
        kind = "transported" if b_rise >= 2.5 and w >= 1.5 else "local build-up" if b_rise < 1.5 and w < 2 else "mixed"
        rows.append({"peak_time": t, "seoul_pm25": y[t], "baengnyeong_rise_x": b_rise, "china_rise_x": cn_rise,
                     "seoul_wind_12h_mean": w, "route_cluster": cl.get(t), "type": kind})
    ep = pd.DataFrame(rows)
    c.table(ep.round(2), "e6_top20_episodes", index=False)
    fig, axs = c.plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    for ax, kind in zip(axs, ["transported", "mixed", "local build-up"]):
        sel = ep[ep.type.eq(kind)]
        for name, s, col in (("Seoul", y, "#2c3e50"), ("Baengnyeong", b, "#e67e22"), ("China mean", cn, "#a23b8f")):
            comp = []
            for t in sel.peak_time:
                w = s[t - pd.Timedelta(hours=72):t + pd.Timedelta(hours=24)]
                comp.append((w / s[t - pd.Timedelta(hours=72):t - pd.Timedelta(hours=48)].median()).set_axis(((w.index - t) / pd.Timedelta(hours=1)).astype(int)))
            if comp:
                m = pd.concat(comp, axis=1).mean(1)
                ax.plot(m.index, m.values, c=col, label=name)
        ax.axvline(0, c="k", lw=0.7)
        ax.set_title(f"{kind} (n={len(sel)})")
        ax.set_xlabel("hours from Seoul peak")
    axs[0].set_ylabel("PM2.5 / level 2-3 days before")
    axs[0].legend(fontsize=8)
    fig.suptitle("Top 20 Seoul PM2.5 peaks (2021-22): does the rise start upstream?")
    c.save(fig, "e6_top20_episodes")


if __name__ == "__main__":
    pm, st = c.universe()
    seollal(pm, st)
    dust()
    gateway(pm)
    episodes(pm, st)
