"""E4 spatial structure of PM2.5 across the 650 stations (training 2021-22).

- correlation of every station with Seoul vs distance (hourly log1p and weekly anomalies)
- region-mean correlation matrix
- lagged cross-correlation region -> Seoul on anomalies (log1p minus 7-day centred mean): peak lag and implied speed
- the same in cold (Nov-Mar) vs warm months
- EOF / PCA of daily station anomalies: the leading spatial modes
Outputs: figures/e4_*.png, tables/e4_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402
from plot_map import land  # noqa: E402

c.setup(__file__)
COL = {"KR": "#d07a2d", "CN": "#a23b8f", "JP": "#2e7d5b", "MN": "#5b3f8f"}
LAGS = range(-12, 49, 3)


def km_from_seoul(st):
    t = st.loc[c.TARGET]
    la1, lo1, la2, lo2 = map(np.deg2rad, (t.lat, t.lon, st.lat, st.lon))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def anomalies(x):
    return x - x.rolling(24 * 7, center=True, min_periods=24 * 3).mean()


def main():
    pm, st = c.universe()
    tr = np.log1p(c.train(pm))
    st["km"] = km_from_seoul(st)
    y = tr[c.TARGET]
    an = anomalies(tr)
    ya = an[c.TARGET]

    # 1 correlation with Seoul vs distance
    r_raw = tr.corrwith(y)
    r_an = an.corrwith(ya)
    s = st.assign(r_raw=r_raw, r_anom=r_an).drop(c.TARGET)
    c.table(s[["country", "region", "km", "r_raw", "r_anom"]].round(3), "e4_station_corr_with_seoul")
    fig, axs = c.plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    for ax, col, title in [(axs[0], "r_raw", "hourly log1p"), (axs[1], "r_anom", "anomaly (minus 7-day mean)")]:
        for cc, g in s.groupby("country"):
            ax.scatter(g.km, g[col], s=7, c=COL[cc], label=cc, alpha=0.7)
        b = pd.cut(s.km, [0, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000])
        m = s.groupby(b, observed=True)[col].median()
        ax.plot([i.mid for i in m.index], m.values, c="k", lw=1.5, label="median")
        ax.set_xscale("log")
        ax.set_xlabel("distance from Seoul Jung-gu (km, log)")
        ax.set_ylabel("correlation with Seoul")
        ax.set_title(title)
    axs[0].legend(fontsize=8)
    land(axs[2])
    sc = axs[2].scatter(s.lon, s.lat, c=s.r_anom, cmap="viridis", s=10, vmin=0, vmax=1, zorder=3)
    axs[2].scatter(st.loc[c.TARGET, "lon"], st.loc[c.TARGET, "lat"], marker="*", s=200, c="r", zorder=4)
    fig.colorbar(sc, ax=axs[2], shrink=0.8, label="anomaly correlation")
    axs[2].set_title("anomaly correlation with Seoul")
    fig.suptitle("How PM2.5 at other stations moves with Seoul (2021-22, same hour)")
    c.save(fig, "e4_corr_vs_distance")

    # 2 region-mean correlation matrix (anomalies)
    rm = c.region_means(c.train(pm), st)
    rma = anomalies(rm)
    order = [r for r in c.REGION_ORDER + ["TARGET"] if r in rma]
    cm = rma[order].corr()
    c.table(cm.round(3), "e4_region_corr_matrix")
    fig, ax = c.plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=70, fontsize=7)
    ax.set_yticks(range(len(order)), order, fontsize=7)
    for (i, j), v in np.ndenumerate(cm.to_numpy()):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5, color="w" if v < 0.6 else "k")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Same-hour correlation of region-mean PM2.5 anomalies (2021-22)")
    c.save(fig, "e4_region_corr_matrix")

    # 3 lagged cross-correlation region(t) -> Seoul(t + lag)
    cold = rma.index.month.isin([11, 12, 1, 2, 3])
    rows, curves = [], {}
    for r in order[:-1]:
        cc = pd.Series({L: rma[r].corr(ya.shift(-L)) for L in LAGS})
        curves[r] = cc
        km_r = st.loc[st.region.eq(r), "km"].median()
        best = int(cc.idxmax())
        row = {"region": r, "median_km": km_r, "peak_lag_h": best, "peak_corr": cc.max(), "corr_lag0": cc[0],
               "implied_speed_kmh": km_r / best if best > 0 else np.nan}
        for name, m in (("cold", cold), ("warm", ~cold)):
            ccm = pd.Series({L: rma[r][m].corr(ya.shift(-L)[m]) for L in LAGS})
            row[f"peak_lag_{name}"], row[f"peak_corr_{name}"] = int(ccm.idxmax()), ccm.max()
        rows.append(row)
    lag = pd.DataFrame(rows).set_index("region")
    c.table(lag.round(3), "e4_lagged_xcorr")
    fig, axs = c.plt.subplots(1, 2, figsize=(15, 5))
    cmap = c.plt.get_cmap("tab20")
    for i, (r, cc) in enumerate(curves.items()):
        if r.startswith("KR_") and r != "KR_CAPITAL":
            continue
        axs[0].plot(cc.index, cc.values, label=r, c=cmap(i % 20), lw=1.4 if r.startswith("CN") else 1)
    axs[0].axvline(0, c="k", lw=0.7)
    axs[0].set_xlabel("lag (h): region at t vs Seoul at t + lag")
    axs[0].set_ylabel("anomaly correlation")
    axs[0].legend(fontsize=7, ncol=2)
    axs[0].set_title("Lagged correlation with Seoul")
    x = lag.drop([r for r in lag.index if r.startswith("KR_")])
    for name, col, mk in (("cold", "#2c3e50", "o"), ("warm", "#e67e22", "s")):
        axs[1].scatter(x.median_km, x[f"peak_lag_{name}"], c=col, marker=mk, label=f"{name} months")
    for r, row in x.iterrows():
        axs[1].annotate(r.replace("CN_", ""), (row.median_km, row.peak_lag_cold), fontsize=7, xytext=(3, 3), textcoords="offset points")
    xs = np.linspace(0, x.median_km.max(), 10)
    for v in (20, 40):
        axs[1].plot(xs, xs / v, ls=":", c="grey")
        axs[1].text(xs[-1], xs[-1] / v, f"{v} km/h", fontsize=7)
    axs[1].set_ylim(-16, 56)
    axs[1].set_xlabel("median distance of the region from Seoul (km)")
    axs[1].set_ylabel("lag with the highest correlation (h)")
    axs[1].legend(fontsize=8)
    axs[1].set_title("Farther regions lead Seoul by more hours (foreign regions)")
    c.save(fig, "e4_lagged_xcorr")

    # 4 EOF of daily anomalies
    d = np.log1p(c.train(pm)).resample("D").mean()
    keep = d.columns[d.notna().mean() >= 0.9]
    dd = d[keep]
    z = (dd - dd.mean()) / dd.std()
    z = z.fillna(0.0)
    u, sv, vt = np.linalg.svd(z.to_numpy() - z.to_numpy().mean(0), full_matrices=False)
    ve = sv ** 2 / (sv ** 2).sum()
    c.table(pd.DataFrame({"mode": range(1, 11), "variance_share": ve[:10]}).round(4), "e4_eof_variance", index=False)
    fig, axs = c.plt.subplots(1, 3, figsize=(18, 5))
    for k, ax in enumerate(axs):
        land(ax)
        load = vt[k] * np.sign(vt[k][list(keep).index(c.TARGET)] if c.TARGET in keep else 1)
        sc = ax.scatter(st.loc[keep, "lon"], st.loc[keep, "lat"], c=load, cmap="RdBu_r", s=10,
                        vmin=-np.abs(load).max(), vmax=np.abs(load).max(), zorder=3)
        ax.set_title(f"EOF {k + 1}: {100 * ve[k]:.1f}% of daily variance")
        fig.colorbar(sc, ax=ax, shrink=0.7)
    fig.suptitle(f"Leading spatial modes of daily PM2.5 ({len(keep)} stations with >= 90% days, 2021-22)")
    c.save(fig, "e4_eof_modes")


if __name__ == "__main__":
    main()
