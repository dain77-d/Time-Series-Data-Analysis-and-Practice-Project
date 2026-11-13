"""E5 where Seoul's air comes from: HYSPLIT 48 h back-trajectories arriving at 500 m (training 2021-22).

- density of trajectory points by season
- k-means route clusters (fit on 2021-22; labels also assigned to 2023 arrivals for E9, never 2024)
- Seoul PM2.5 at arrival by cluster, transport speed, mixing depth and rain along the path
- CWT (concentration-weighted trajectory) and PSCF (potential source contribution) maps
- hours spent near each region's stations vs Seoul PM2.5
Outputs: figures/e5_*.png, tables/e5_*.csv
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402

c.setup(__file__)
from plot_map import EXTENT, LAND, land


def coast(ax):
    """coastline drawn on top of a gridded field."""
    import json
    for f in json.loads(LAND.read_text(encoding="utf-8"))["features"]:
        g = f["geometry"]
        for poly in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            x, y = zip(*poly[0])
            if max(x) < EXTENT[0] or min(x) > EXTENT[1] or max(y) < EXTENT[2] or min(y) > EXTENT[3]:
                continue
            ax.plot(x, y, c="#333", lw=0.5, zorder=4)
    ax.set_xlim(EXTENT[:2])
    ax.set_ylim(EXTENT[2:])

K = 6
STEP = 3     # use every 3rd hour of the path as clustering features


def load():
    t = pd.read_parquet(c.root() / "Dataset" / "Trajectories" / "trajectory_points_2021_2024.parquet",
                        columns=["arrival_time_kst", "arrival_height_agl_m", "lookback_hours", "latitude", "longitude",
                                 "height_agl_m", "diagnostic_mixdepth", "diagnostic_rainfall"])
    t = t[t.arrival_height_agl_m.eq(500) & t.lookback_hours.between(0, 48)].copy()
    t["arrival"] = pd.DatetimeIndex(t.arrival_time_kst).tz_convert("Asia/Seoul")
    return t[t.arrival < "2024-01-01"]


def wide(t):
    lon = t.pivot_table(index="arrival", columns="lookback_hours", values="longitude")
    lat = t.pivot_table(index="arrival", columns="lookback_hours", values="latitude")
    ok = lon.notna().all(1) & lat.notna().all(1)
    return lon[ok], lat[ok]


def km(lon, lat):
    """great-circle length of each path (km) from hourly points."""
    la, lo = np.deg2rad(lat.to_numpy()), np.deg2rad(lon.to_numpy())
    a = np.sin(np.diff(la, axis=1) / 2) ** 2 + np.cos(la[:, :-1]) * np.cos(la[:, 1:]) * np.sin(np.diff(lo, axis=1) / 2) ** 2
    return (6371 * 2 * np.arcsin(np.sqrt(a))).sum(1)


def main():
    t = load()
    pm, st = c.universe()
    y = pm[c.TARGET]
    lon, lat = wide(t)
    tr = lon.index < "2023-01-01"
    cols = [h for h in lon.columns if h % STEP == 0]
    X = np.hstack([(lon[cols] - 126.98).to_numpy() * np.cos(np.deg2rad(37.5)), (lat[cols] - 37.57).to_numpy()])
    model = KMeans(K, n_init=20, random_state=0).fit(X[tr])
    lab = pd.Series(model.predict(X), index=lon.index)
    # name clusters by the mean position 24 h back (direction from Seoul) and order by distance travelled
    dist = pd.Series(km(lon, lat), index=lon.index)
    info = []
    for k in range(K):
        m = lab.eq(k) & tr
        dx, dy = lon.loc[m, 24].mean() - 126.98, lat.loc[m, 24].mean() - 37.57
        ang = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
        sector = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int(((ang + 22.5) % 360) // 45)]
        info.append({"cluster": k, "sector_24h": sector, "mean_path_km": dist[m].mean()})
    info = pd.DataFrame(info).sort_values("mean_path_km")
    speed = {r.cluster: ("slow" if r.mean_path_km < 900 else "fast" if r.mean_path_km > 1600 else "mid") for r in info.itertuples()}
    names = {r.cluster: f"C{i + 1} {r.sector_24h}-{speed[r.cluster]}" for i, r in enumerate(info.itertuples())}
    lab_named = lab.map(names)
    lab_named.rename("cluster").to_frame().assign(path_km=dist).to_csv(c.TAB / "e5_traj_clusters_2021_2023.csv", encoding="utf-8-sig")

    trn = lab_named[tr]
    ya = y.reindex(trn.index)
    order = sorted(names.values())
    pal = dict(zip(order, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]))

    # 1. mean paths
    fig, ax = c.plt.subplots(figsize=(10, 7))
    land(ax)
    for k, nm in names.items():
        m = lab.eq(k) & tr
        for i in np.random.default_rng(0).choice(np.flatnonzero(m), size=min(40, m.sum()), replace=False):
            ax.plot(lon.iloc[i], lat.iloc[i], c=pal[nm], lw=0.3, alpha=0.25)
        ax.plot(lon[m].mean(), lat[m].mean(), c=pal[nm], lw=3, label=f"{nm}: {100 * m.sum() / tr.sum():.0f}%, PM {ya[trn.eq(nm)].mean():.0f}")
    ax.scatter(126.98, 37.57, marker="*", s=250, c="k", zorder=5)
    ax.legend(fontsize=8, loc="lower right", title="cluster: share, mean Seoul PM2.5")
    ax.set_title("48 h back-trajectory clusters arriving at Seoul, 500 m (k-means, 2021-22)")
    c.save(fig, "e5_cluster_paths")

    # 2. cluster stats
    s = pd.DataFrame({"cluster": trn, "pm": ya, "season": c.season(trn.index), "km": dist[tr]})
    stat = s.groupby("cluster").agg(n=("pm", "size"), share=("pm", lambda x: len(x) / len(s)), pm_mean=("pm", "mean"),
                                    pm_median=("pm", "median"), p_gt35=("pm", lambda x: (x > 35).mean()), path_km=("km", "mean"))
    c.table(stat.round(3), "e5_cluster_stats")
    fig, ax = c.plt.subplots(1, 2, figsize=(13, 4.2))
    pd.crosstab(s.season, s.cluster, normalize="index").reindex(["DJF", "MAM", "JJA", "SON"])[order].plot.bar(
        stacked=True, ax=ax[0], color=[pal[o] for o in order], legend=False)
    ax[0].set_ylabel("share of arrivals")
    ax[0].set_title("Route cluster mix by season")
    ax[1].boxplot([s.pm[s.cluster.eq(o)].dropna() for o in order], tick_labels=order, showfliers=False)
    ax[1].tick_params(axis="x", rotation=30, labelsize=8)
    ax[1].set_ylabel("Seoul PM2.5 at arrival")
    ax[1].set_title("Seoul PM2.5 by route cluster")
    fig.legend([c.plt.Rectangle((0, 0), 1, 1, color=pal[o]) for o in order], order, loc="upper center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    c.save(fig, "e5_cluster_season_pm")

    # 3. density by season
    pts = t[t.arrival < "2023-01-01"]
    pts = pts.assign(season=c.season(pd.DatetimeIndex(pts.arrival)))
    fig, axs = c.plt.subplots(2, 2, figsize=(12, 9))
    for ax, sea in zip(axs.ravel(), ["DJF", "MAM", "JJA", "SON"]):
        land(ax)
        g = pts[pts.season.eq(sea) & pts.lookback_hours.gt(0)]
        h, xe, ye = np.histogram2d(g.longitude, g.latitude, bins=[np.arange(104, 138.5, 0.5), np.arange(30, 50.5, 0.5)])
        h = np.ma.masked_equal(h, 0)
        ax.pcolormesh(xe, ye, np.log10(h.T), cmap="YlOrRd", alpha=0.75, zorder=1)
        coast(ax)
        ax.scatter(126.98, 37.57, marker="*", s=150, c="k", zorder=5)
        ax.set_title(f"{sea}: where Seoul's air was in the past 48 h (log count)")
    c.save(fig, "e5_density_by_season")

    # 4. speed, mixing depth and rain vs PM
    arr = t[t.lookback_hours.eq(0)].set_index("arrival")
    rain = t[t.lookback_hours.between(1, 24)].groupby("arrival").diagnostic_rainfall.sum()
    f = pd.DataFrame({"pm": y.reindex(lon.index), "km48": dist, "mixdepth": arr.diagnostic_mixdepth.reindex(lon.index),
                      "rain24": rain.reindex(lon.index)})[tr]
    fig, ax = c.plt.subplots(1, 3, figsize=(15, 4))
    for a, col, bins, lab_ in [(ax[0], "km48", [0, 500, 800, 1100, 1400, 1800, 2400, 5000], "path length in 48 h (km)"),
                               (ax[1], "mixdepth", [0, 200, 400, 700, 1000, 1500, 2500, 6000], "mixing depth at arrival (m)"),
                               (ax[2], "rain24", [-0.01, 0, 0.5, 2, 5, 15, 1000], "rain along the last 24 h of the path")]:
        b = pd.cut(f[col], bins)
        g = f.groupby(b, observed=True).pm.agg(["mean", "count"])
        a.bar(range(len(g)), g["mean"], color="#34495e")
        a.set_xticks(range(len(g)), [str(i) for i in g.index], rotation=40, fontsize=7)
        a.set_xlabel(lab_)
        a.set_ylabel("mean Seoul PM2.5")
        for i, n in enumerate(g["count"]):
            a.text(i, 0.5, f"n={n}", ha="center", fontsize=6, color="w", rotation=90, va="bottom")
    fig.suptitle("Slow air, shallow mixing and no rain go with high Seoul PM2.5 (2021-22)")
    c.save(fig, "e5_speed_mixing_rain")
    c.table(f.corr(method="spearman").round(3), "e5_path_spearman")

    # 5. CWT and PSCF on a 1 degree grid (points 1-48 h back), weighted down where a cell has few points
    p = pts[pts.lookback_hours.gt(0)].copy()
    p["pm"] = y.reindex(pd.DatetimeIndex(p.arrival)).to_numpy()
    p = p.dropna(subset=["pm"])
    p["gx"], p["gy"] = np.floor(p.longitude).astype(int), np.floor(p.latitude).astype(int)
    thr = y.reindex(lon.index[tr]).quantile(0.75)
    g = p.groupby(["gx", "gy"]).agg(n=("pm", "size"), cwt=("pm", "mean"), m=("pm", lambda x: (x > thr).sum()))
    navg = g.n.mean()
    w = np.select([g.n > 3 * navg, g.n > 1.5 * navg, g.n > navg], [1.0, 0.7, 0.4], 0.2)
    g["cwt_w"], g["pscf_w"] = g.cwt * w, g.m / g.n * w
    c.table(g.round(3), "e5_cwt_pscf_grid")
    fig, axs = c.plt.subplots(1, 2, figsize=(15, 6))
    for ax, col, title, cmap in [(axs[0], "cwt_w", "CWT: mean Seoul PM2.5 of air that passed each cell", "magma_r"),
                                 (axs[1], "pscf_w", f"PSCF: share of passes ending in Seoul PM2.5 > {thr:.0f} (top 25%)", "viridis")]:
        land(ax)
        Z = np.full((20, 34), np.nan)
        for (gx, gy), v in g[col].items():
            if 104 <= gx < 138 and 30 <= gy < 50:
                Z[gy - 30, gx - 104] = v
        im = ax.pcolormesh(np.arange(104, 139), np.arange(30, 51), np.ma.masked_invalid(Z), cmap=cmap, alpha=0.8, zorder=1)
        fig.colorbar(im, ax=ax, shrink=0.7)
        coast(ax)
        ax.scatter(126.98, 37.57, marker="*", s=200, c="k", zorder=5)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Potential source areas for Seoul PM2.5, 2021-22 (500 m trajectories, cells weighted by number of points)")
    c.save(fig, "e5_cwt_pscf")

    # 6. hours near each region (within 100 km of any of its stations) vs Seoul PM
    from scipy.spatial import cKDTree
    pp = pts[pts.lookback_hours.between(1, 48)]
    xy = np.c_[pp.longitude.to_numpy() * np.cos(np.deg2rad(37.5)), pp.latitude.to_numpy()]
    res = {}
    for r in c.REGION_ORDER:
        s_ = st[st.region.eq(r)]
        if s_.empty:
            continue
        tree = cKDTree(np.c_[s_.lon * np.cos(np.deg2rad(37.5)), s_.lat])
        near = tree.query(xy, distance_upper_bound=0.9)[0] < 0.9      # ~100 km
        res[r] = pd.Series(near, index=pp.arrival.to_numpy()).groupby(level=0).sum()
    rt = pd.DataFrame(res).reindex(lon.index[tr]).fillna(0)
    yy = y.reindex(rt.index)
    rows = []
    for r in rt:
        visited = rt[r] > 0
        rows.append({"region": r, "share_of_arrivals_passing": visited.mean(), "mean_hours_when_passing": rt[r][visited].mean(),
                     "pm_if_passed": yy[visited].mean(), "pm_if_not": yy[~visited].mean(), "spearman_hours_pm": rt[r].corr(yy, method="spearman")})
    rr = pd.DataFrame(rows).set_index("region")
    c.table(rr.round(3), "e5_region_residence")
    fig, ax = c.plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(rr))
    ax.bar(x - 0.2, rr.pm_if_passed, 0.4, label="air passed the region in the last 48 h", color="#c0392b")
    ax.bar(x + 0.2, rr.pm_if_not, 0.4, label="did not pass", color="#95a5a6")
    for i, v in enumerate(rr.share_of_arrivals_passing):
        ax.text(i - 0.2, 1, f"{100 * v:.0f}%", ha="center", fontsize=7, color="w", rotation=90, va="bottom")
    ax.set_xticks(x, rr.index, rotation=50, fontsize=8)
    ax.set_ylabel("mean Seoul PM2.5 at arrival")
    ax.legend(fontsize=8)
    ax.set_title("Seoul PM2.5 when the air passed near each region (within ~100 km of its stations; % = how often)")
    c.save(fig, "e5_region_residence")


if __name__ == "__main__":
    main()
