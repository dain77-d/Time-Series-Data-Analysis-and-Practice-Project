"""E11 how PM2.5 moves through the whole station network (training 2021-22).

- every station's lead/lag against Seoul (peak of the anomaly cross-correlation), mapped
- lead/lag between every pair of regions (who moves first, anywhere in the network)
- how the four largest Korea-wide episodes spread: time of each station's peak relative to Seoul's
- data-driven station groups (k-means on daily anomaly series) vs the administrative regions
Anomaly = log1p minus its 7-day centred mean (removes the shared seasonal swing).
Outputs: figures/e11_*.png, tables/e11_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402
from plot_map import land  # noqa: E402

c.setup(__file__)
LAGS = np.arange(-24, 49)


def anomalies(x):
    return x - x.rolling(24 * 7, center=True, min_periods=24 * 3).mean()


def zscore(x):
    return (x - x.mean()) / x.std()


def lead_lag_with(A, y):
    """peak lag and correlation of every column of A (t) against y (t + lag); A, y standardized anomalies."""
    a = A.to_numpy()
    out = np.full((len(LAGS), a.shape[1]), np.nan)
    yv = y.to_numpy()
    for i, L in enumerate(LAGS):
        ys = np.roll(yv, -L)
        if L > 0:
            ys[-L:] = np.nan
        elif L < 0:
            ys[:-L] = np.nan
        out[i] = np.nanmean(a * ys[:, None], axis=0)
    cc = pd.DataFrame(out, index=LAGS, columns=A.columns)
    return cc.idxmax(), cc.max(), cc.loc[0]


def station_lags(pm, st):
    A = zscore(anomalies(np.log1p(c.train(pm))))
    lag, peak, zero = lead_lag_with(A, A[c.TARGET])
    t = st.join(pd.DataFrame({"peak_lag_h": lag, "peak_corr": peak, "corr_lag0": zero})).drop(c.TARGET)
    c.table(t.round(3), "e11_station_lead_lag_vs_seoul")
    fig, axs = c.plt.subplots(1, 2, figsize=(19, 7))
    land(axs[0])
    x = t[t.peak_corr >= 0.1]
    sc = axs[0].scatter(x.lon, x.lat, c=x.peak_lag_h.clip(-24, 30), cmap="RdBu_r", vmin=-30, vmax=30,
                        s=10 + 60 * x.peak_corr, zorder=3, edgecolor="k", lw=0.2)
    axs[0].scatter(st.loc[c.TARGET, "lon"], st.loc[c.TARGET, "lat"], marker="*", s=250, c="lime", edgecolor="k", zorder=4)
    fig.colorbar(sc, ax=axs[0], shrink=0.75, label="hours the station leads Seoul (red) / lags it (blue)")
    axs[0].set_title("Every station vs Seoul: lead time of the best-correlated lag (size = correlation; corr >= 0.1)")
    kr = t[t.country.eq("KR")]
    axs[1].scatter(kr.lon, kr.peak_lag_h + np.random.default_rng(0).uniform(-0.3, 0.3, len(kr)), s=8, c="#d07a2d", label="Korea")
    for cc, col in (("CN", "#a23b8f"), ("JP", "#2e7d5b")):
        g = t[t.country.eq(cc)]
        axs[1].scatter(g.lon, g.peak_lag_h, s=12, c=col, label=cc)
    axs[1].axhline(0, c="k", lw=0.7)
    axs[1].axvline(126.98, c="grey", ls=":", lw=0.8)
    axs[1].set_xlabel("station longitude (west -> east)")
    axs[1].set_ylabel("hours ahead of Seoul")
    axs[1].legend(fontsize=8)
    axs[1].set_title("Lead time falls from west to east across the whole network")
    c.save(fig, "e11_station_lead_lag_map")
    g = kr[kr.peak_corr >= 0.4].sort_values("peak_lag_h", ascending=False).head(15)
    c.table(g[["region", "lon", "lat", "peak_lag_h", "peak_corr", "corr_lag0"]].round(3), "e11_korean_stations_leading_seoul")
    return A


def region_pairs(pm, st):
    R = zscore(anomalies(c.region_means(c.train(pm), st)))
    order = [r for r in c.REGION_ORDER if r in R]
    lag = pd.DataFrame(index=order, columns=order, dtype=float)
    cor = lag.copy()
    for r in order:
        lg, pk, _ = lead_lag_with(R[order], R[r])            # column at t vs r at t + lag
        lg = lg.where(lg.abs() <= 24)
        lag[r], cor[r] = lg, pk
    c.table(lag, "e11_region_pair_lag")
    c.table(cor.round(3), "e11_region_pair_corr")
    fig, ax = c.plt.subplots(figsize=(11, 9))
    show = lag.where(cor >= 0.2)
    im = ax.imshow(show.to_numpy(float), cmap="RdBu_r", vmin=-24, vmax=24)
    ax.set_xticks(range(len(order)), order, rotation=70, fontsize=7)
    ax.set_yticks(range(len(order)), order, fontsize=7)
    for (i, j), v in np.ndenumerate(show.to_numpy(float)):
        if np.isfinite(v) and i != j:
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=6)
    ax.grid(False)
    ax.set_xlabel("follower region (at t + lag)")
    ax.set_ylabel("leader region (at t)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="hours the row region leads the column region")
    ax.set_title("Who moves first: lead time between every pair of regions (blank = peak corr < 0.2)")
    c.save(fig, "e11_region_pair_lag")


def propagation(pm, st):
    tr = c.train(pm)
    kr = st.index[st.country.eq("KR")]
    nat = tr[kr].median(1).rolling(6, center=True).mean()
    daily = nat.resample("D").max()
    peaks = []
    for d in daily.sort_values(ascending=False).index:
        if all(abs((d - p).days) > 7 for p in peaks):
            peaks.append(d)
        if len(peaks) == 4:
            break
    sm = tr.rolling(3, center=True, min_periods=2).mean()
    fig, axs = c.plt.subplots(2, 2, figsize=(17, 12))
    rows = []
    for ax, d in zip(axs.ravel(), sorted(peaks)):
        t_nat = nat[d:d + pd.Timedelta(hours=23)].idxmax()
        w = sm[t_nat - pd.Timedelta(hours=60):t_nat + pd.Timedelta(hours=36)]
        tmax = w.idxmax()
        rel = ((tmax - tmax[c.TARGET]) / pd.Timedelta(hours=1)).where(w.notna().sum() > 48)
        ok = w.max() > 1.5 * w.iloc[:24].median()            # stations that really had an event
        rel = rel.where(ok)
        land(ax)
        sc = ax.scatter(st.lon, st.lat, c=rel.reindex(st.index), cmap="RdBu", vmin=-36, vmax=36, s=12, zorder=3, edgecolor="k", lw=0.15)
        ax.scatter(st.loc[c.TARGET, "lon"], st.loc[c.TARGET, "lat"], marker="*", s=250, c="lime", edgecolor="k", zorder=4)
        fig.colorbar(sc, ax=ax, shrink=0.7, label="peak time minus Seoul's peak (h)")
        ax.set_title(f"Episode peaking {t_nat:%Y-%m-%d %H}h KST (Korea median {nat[t_nat]:.0f} ug/m3)")
        for cc in ("CN", "KR", "JP"):
            m = st.country.eq(cc).reindex(rel.index).fillna(False)
            rows.append({"episode": f"{t_nat:%Y-%m-%d %H}h", "country": cc, "stations_with_event": int(rel[m].notna().sum()),
                         "median_peak_vs_seoul_h": rel[m].median()})
    fig.suptitle("How the four largest Korea-wide episodes spread (red = peaked before Seoul, blue = after)", fontsize=13)
    c.save(fig, "e11_episode_propagation")
    c.table(pd.DataFrame(rows).round(1), "e11_episode_propagation", index=False)


def clusters(pm, st, k=10):
    d = np.log1p(c.train(pm)).resample("D").mean()
    keep = d.columns[d.notna().mean() >= 0.9]
    z = zscore(d[keep] - d[keep].rolling(7, center=True, min_periods=3).mean()).fillna(0.0)
    lab = pd.Series(KMeans(k, n_init=20, random_state=0).fit_predict(z.T.to_numpy()), index=keep)
    # name each group by its dominant administrative region
    name = {g: f"G{g + 1}: " + st.region.reindex(lab.index[lab.eq(g)]).value_counts().index[0] for g in range(k)}
    lab = lab.map(name)
    ct = pd.crosstab(st.region.reindex(lab.index), lab)
    c.table(ct, "e11_station_groups_vs_regions")
    fig, ax = c.plt.subplots(figsize=(11, 8))
    land(ax)
    cmap = c.plt.get_cmap("tab10")
    for i, g in enumerate(sorted(lab.unique())):
        s = st.loc[lab.index[lab.eq(g)]]
        ax.scatter(s.lon, s.lat, s=12, color=cmap(i), label=f"{g} ({len(s)})", zorder=3)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Data-driven station groups: k-means on daily PM2.5 anomaly series (2021-22)")
    c.save(fig, "e11_station_groups")


def main():
    pm, st = c.universe()
    station_lags(pm, st)
    region_pairs(pm, st)
    propagation(pm, st)
    clusters(pm, st)


if __name__ == "__main__":
    main()
