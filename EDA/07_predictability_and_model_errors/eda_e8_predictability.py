"""E8 what makes Seoul PM2.5 hard to forecast, and what information predicts the change (training 2021-22).

A forecast has to predict the change y(t+h) - y(t); persistence predicts zero change.
- persistence RMSE by season, route cluster and current PM level
- Spearman of the future change with information available at t (Seoul trend, Baengnyeong, Chinese regions, weather)
- the same for upstream signals by route cluster
- future change after a sharp upstream rise vs otherwise
Outputs: figures/e8_*.png, tables/e8_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402

c.setup(__file__)
H = (3, 6, 12, 24)
BAENG = "KR:831492"
CN = ["CN_BTH", "CN_SHANDONG", "CN_LIAONING", "CN_JILIN", "CN_HEILONGJIANG", "CN_INNER_MONGOLIA"]


def main():
    pm, st = c.universe()
    y = c.train(pm[c.TARGET]).astype(float)
    ly = np.log1p(y)
    rm = c.region_means(c.train(pm), st)                 # log1p region means
    lb = np.log1p(c.train(pm[BAENG]).astype(float))
    a = c.train(c.asos())
    cl = c.clusters().reindex(y.index, method="ffill", limit=2)
    sea = pd.Series(c.season(y.index), index=y.index)
    dy = {h: ly.shift(-h) - ly for h in H}                # future change (log)

    # 1 persistence RMSE by condition
    rows = []
    level = pd.cut(y, [0, 15, 35, 75, 1000], labels=["<=15", "15-35", "35-75", ">75"])
    for name, grp in (("season", sea), ("route_cluster", cl), ("level_at_t", level)):
        for h in H:
            e = (y.shift(-h) - y)
            r = e.groupby(grp, observed=True).apply(lambda x: np.sqrt(np.nanmean(x ** 2)))
            for k, v in r.items():
                rows.append({"by": name, "group": k, "horizon": h, "persistence_RMSE": v})
    pr = pd.DataFrame(rows)
    c.table(pr.round(3), "e8_persistence_rmse_by_condition", index=False)
    fig, axs = c.plt.subplots(1, 3, figsize=(17, 4))
    for ax, name in zip(axs, ["season", "route_cluster", "level_at_t"]):
        p = pr[pr.by.eq(name)].pivot(index="group", columns="horizon", values="persistence_RMSE")
        p.plot.bar(ax=ax)
        ax.set_title(f"persistence RMSE by {name.replace('_', ' ')}")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    axs[0].set_ylabel("RMSE (ug/m3)")
    fig.suptitle("Where the naive forecast fails (2021-22)")
    c.save(fig, "e8_persistence_by_condition")

    # 2 what predicts the change
    X = {"Seoul change last 3 h": ly - ly.shift(3), "Seoul level (log)": ly,
         "Seoul minus 7-day mean": ly - ly.rolling(168, min_periods=72).mean(),
         "Baengnyeong minus Seoul": lb - ly, "Baengnyeong change last 6 h": lb - lb.shift(6),
         "KR Chungcheong minus Seoul": rm["KR_CHUNGCHEONG"] - ly,
         "China mean minus Seoul": rm[CN].mean(1) - ly, "China mean change last 12 h": rm[CN].mean(1) - rm[CN].mean(1).shift(12)}
    for r in CN:
        X[f"{r} change last 12 h"] = rm[r] - rm[r].shift(12)
    X.update({"wind speed 108": np.hypot(a.asos108_wind_u, a.asos108_wind_v), "u wind 102 (west+)": a.asos102_wind_u,
              "pressure 108 change 12 h": a.asos108_pressure - a.asos108_pressure.shift(12)})
    sp = pd.DataFrame({f"+{h} h": {k: x.corr(dy[h], method="spearman") for k, x in X.items()} for h in H})
    c.table(sp.round(3), "e8_change_predictors_spearman")
    fig, ax = c.plt.subplots(figsize=(8, 7))
    im = ax.imshow(sp.to_numpy(float), cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(H)), sp.columns)
    ax.set_yticks(range(len(sp)), sp.index, fontsize=8)
    for (i, j), v in np.ndenumerate(sp.to_numpy(float)):
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman with future change of log PM2.5")
    ax.set_title("Information at t vs Seoul's change over the next h hours (2021-22)")
    c.save(fig, "e8_change_predictors")

    # 3 upstream signals by route cluster
    keys = ["Baengnyeong minus Seoul", "China mean minus Seoul", "China mean change last 12 h", "KR Chungcheong minus Seoul"]
    rows = []
    for k in sorted(cl.dropna().unique()):
        m = cl.eq(k)
        for name in keys:
            for h in (6, 12, 24):
                rows.append({"cluster": k, "signal": name, "horizon": h, "spearman": X[name][m].corr(dy[h][m], method="spearman")})
    bc = pd.DataFrame(rows)
    c.table(bc.round(3), "e8_upstream_signal_by_cluster", index=False)
    fig, axs = c.plt.subplots(1, 3, figsize=(17, 4), sharey=True)
    for ax, h in zip(axs, (6, 12, 24)):
        p = bc[bc.horizon.eq(h)].pivot(index="cluster", columns="signal", values="spearman")
        p.plot.bar(ax=ax)
        ax.set_title(f"+{h} h")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend(fontsize=7)
    axs[0].set_ylabel("Spearman with Seoul future change")
    fig.suptitle("Upstream signals vs Seoul future change, by route cluster (2021-22): China trend helps most on W and NW routes at 24 h")
    c.save(fig, "e8_upstream_signal_by_cluster")

    # 4 after a sharp upstream rise
    rise = X["Baengnyeong change last 6 h"] > X["Baengnyeong change last 6 h"].quantile(0.9)
    rows = []
    fig, axs = c.plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, h in zip(axs, H):
        ch = y.shift(-h) - y
        ax.boxplot([ch[~rise].dropna(), ch[rise].dropna()], tick_labels=["otherwise", "Baengnyeong\nrose (top 10%)"], showfliers=False)
        ax.axhline(0, c="grey", lw=0.7)
        ax.set_title(f"Seoul change over +{h} h")
        rows.append({"horizon": h, "median_change_after_rise": ch[rise].median(), "median_change_otherwise": ch[~rise].median(),
                     "share_rising_after_rise": (ch[rise] > 0).mean(), "share_rising_otherwise": (ch[~rise] > 0).mean()})
    axs[0].set_ylabel("ug/m3")
    fig.suptitle("After Baengnyeong jumps, Seoul tends to rise (2021-22)")
    c.save(fig, "e8_after_upstream_rise")
    c.table(pd.DataFrame(rows).round(3), "e8_after_upstream_rise", index=False)


if __name__ == "__main__":
    main()
