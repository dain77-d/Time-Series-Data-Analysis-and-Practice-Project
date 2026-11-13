"""E10 every station, not just Seoul: level, seasonality, daily cycle and how 'local' each station is (training 2021-22).

Per station (650 inputs + Seoul Jung-gu): mean, share of hours > 35, winter/summer ratio, daily-cycle amplitude and peak
hour, and the share of its daily variance shared with the whole network (EOF 1) -- 1 minus that is its 'local' part.
Plus monthly and daily cycles by region.
Outputs: figures/e10_*.png, tables/e10_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402
from plot_map import land  # noqa: E402

c.setup(__file__)


def profiles(pm, st):
    tr = c.train(pm)
    m = tr.index.month
    win, summ = tr[np.isin(m, [12, 1, 2])].mean(), tr[np.isin(m, [6, 7, 8])].mean()
    # daily cycle in each station's own local clock is what matters physically; KST is used for all (CN is KST-1)
    hourly = tr.groupby(tr.index.hour).mean()
    d = np.log1p(tr).resample("D").mean()
    keep = d.columns[d.notna().mean() >= 0.9]
    z = ((d[keep] - d[keep].mean()) / d[keep].std()).fillna(0.0)
    zc = z - z.mean()
    u, s, vt = np.linalg.svd(zc.to_numpy(), full_matrices=False)
    shared = pd.Series((vt[0] * s[0]) ** 2 / (zc.to_numpy() ** 2).sum(0), index=keep)
    out = pd.DataFrame({"mean": tr.mean(), "p_gt35": (tr > 35).sum() / tr.notna().sum(), "winter_summer_ratio": win / summ,
                        "diurnal_amplitude": hourly.max() - hourly.min(), "diurnal_peak_hour_kst": hourly.idxmax(),
                        "shared_with_network": shared, "availability": tr.notna().mean()})
    return st.join(out)


def maps(p):
    cols = [("mean", "mean PM2.5 (ug/m3)", "magma_r", None), ("p_gt35", "share of hours > 35", "magma_r", None),
            ("winter_summer_ratio", "winter / summer mean", "PuOr_r", (0.5, 3.5)),
            ("diurnal_amplitude", "daily-cycle amplitude (ug/m3)", "viridis", None),
            ("diurnal_peak_hour_kst", "hour of the daily peak (KST)", "twilight", (0, 23)),
            ("shared_with_network", "share of daily variance shared with the network (EOF 1)", "cividis", (0, 1))]
    fig, axs = c.plt.subplots(2, 3, figsize=(20, 11))
    for ax, (col, title, cmap, lim) in zip(axs.ravel(), cols):
        land(ax)
        x = p.dropna(subset=[col])
        vmin, vmax = lim if lim else (x[col].quantile(0.02), x[col].quantile(0.98))
        sc = ax.scatter(x.lon, x.lat, c=x[col], cmap=cmap, s=12, vmin=vmin, vmax=vmax, zorder=3, lw=0)
        t = p.loc[c.TARGET]
        ax.scatter(t.lon, t.lat, marker="*", s=180, facecolor="none", edgecolor="k", zorder=4)
        fig.colorbar(sc, ax=ax, shrink=0.75)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Station profiles across Korea, China, Japan and Mongolia (2021-22)", fontsize=13)
    c.save(fig, "e10_station_profile_maps")


def by_region(p, pm):
    order = [r for r in c.REGION_ORDER if r in p.region.values]
    s = p.groupby("region")[["mean", "p_gt35", "winter_summer_ratio", "diurnal_amplitude", "shared_with_network"]].median().reindex(order)
    s["stations"] = p.region.value_counts().reindex(order)
    c.table(s.round(3), "e10_profiles_by_region")
    tr = c.train(pm)
    rm = c.region_means(tr, p, log=False)
    mon = rm.groupby(rm.index.month).mean()
    mon = mon / mon.mean()
    hr = rm.groupby(rm.index.hour).mean()
    hr = hr / hr.mean()
    fig, axs = c.plt.subplots(1, 2, figsize=(16, 5))
    cmap = c.plt.get_cmap("tab20")
    for i, r in enumerate(order):
        ls = "-" if r.startswith("KR") else "--" if r.startswith("CN") else ":"
        axs[0].plot(mon.index, mon[r], ls=ls, c=cmap(i % 20), label=r)
        axs[1].plot(hr.index, hr[r], ls=ls, c=cmap(i % 20), label=r)
    axs[0].set_xticks(range(1, 13))
    axs[0].set_xlabel("month")
    axs[0].set_title("Seasonal cycle by region (relative to the annual mean)")
    axs[1].set_xlabel("hour (KST)")
    axs[1].set_title("Daily cycle by region (relative to the daily mean)")
    axs[1].legend(fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    c.save(fig, "e10_cycles_by_region")
    fig, axs = c.plt.subplots(1, 3, figsize=(18, 4.5))
    for ax, col, t in zip(axs, ["mean", "winter_summer_ratio", "shared_with_network"],
                          ["mean PM2.5", "winter / summer", "shared with network (EOF 1)"]):
        ax.boxplot([p.loc[p.region.eq(r), col].dropna() for r in order], tick_labels=order, showfliers=False)
        ax.tick_params(axis="x", rotation=70, labelsize=7)
        ax.set_title(t)
    fig.suptitle("Station profiles by region (each box = the region's stations)")
    c.save(fig, "e10_profiles_by_region")


def main():
    pm, st = c.universe()
    p = profiles(pm, st)
    c.table(p.round(3), "e10_station_profiles")
    maps(p)
    by_region(p, pm)


if __name__ == "__main__":
    main()
