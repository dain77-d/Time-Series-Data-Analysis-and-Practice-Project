"""E9 where the C models miss: 2023 validation errors by route cluster, season and PM level (never 2024).

C_GRU F (all stations) and DN (domestic only), 3-seed mean, on the team's fixed eligible grid; persistence from the grid.
Route cluster = E5 k-means label of the trajectory arriving at the forecast origin t (known at t).
Outputs: tables/e9_rmse_by_cluster.csv, e9_rmse_by_season.csv, e9_bias_by_level.csv
         figures/e9_rmse_by_cluster.png, e9_f_minus_dn_by_cluster.png, e9_bias_by_level.png, e9_obs_vs_pred.png
"""
import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402

c.setup(__file__)

RUNS = c.REPO / "results" / "C" / "runs" / "C_GRU"
H = (3, 6, 12, 24)


def preds(cond):
    p = pd.concat([pd.read_parquet(f) for f in sorted((RUNS / cond).glob("seed*/predictions.parquet"))])
    p = p[p.split.eq("validation")]
    return p.groupby(["origin_time_kst", "horizon"]).y_hat.mean().rename(cond)


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def main():
    g = pd.read_parquet(c.root() / "Dataset" / "Evaluation_Contract_20261107" / "evaluation_grid.parquet",
                        columns=["origin_time_kst", "split", "horizon", "y_true", "eligible", "persistence_y_hat"])
    g = g[g.split.eq("validation") & g.eligible].set_index(["origin_time_kst", "horizon"])
    d = g.join(preds("F")).join(preds("DN")).dropna(subset=["F", "DN"]).reset_index()
    cl = pd.read_csv(c.CLUSTERS, index_col=0, parse_dates=True).cluster
    cl.index = pd.DatetimeIndex(cl.index).tz_convert("Asia/Seoul")
    d["origin"] = pd.DatetimeIndex(d.origin_time_kst).tz_convert("Asia/Seoul")
    d["cluster"] = cl.reindex(d.origin).to_numpy()
    d["season"] = c.season(pd.DatetimeIndex(d.origin))
    d["level"] = pd.cut(d.y_true, [0, 15, 35, 75, 1000], labels=["<=15", "15-35", "35-75", ">75"])
    for m in ("F", "DN", "persistence_y_hat"):
        d[f"e_{m}"] = d[m] - d.y_true

    def table_by(col):
        r = d.groupby([col, "horizon"], observed=True).apply(lambda x: pd.Series(
            {"n": len(x), "RMSE_F": rmse(x.e_F), "RMSE_DN": rmse(x.e_DN), "RMSE_persistence": rmse(x.e_persistence_y_hat),
             "F_minus_DN": rmse(x.e_F) - rmse(x.e_DN), "bias_F": x.e_F.mean()}), include_groups=False)
        return r.round(3)

    bc = table_by("cluster")
    c.table(bc, "e9_rmse_by_cluster")
    c.table(table_by("season"), "e9_rmse_by_season")
    bl = table_by("level")
    c.table(bl, "e9_bias_by_level")

    order = sorted(d.cluster.dropna().unique())
    fig, axs = c.plt.subplots(1, 4, figsize=(17, 4), sharey=False)
    for ax, h in zip(axs, H):
        x = bc.xs(h, level="horizon").reindex(order)
        i = np.arange(len(x))
        ax.bar(i - 0.27, x.RMSE_persistence, 0.27, label="persistence", color="#bdc3c7")
        ax.bar(i, x.RMSE_DN, 0.27, label="C_GRU DN (domestic)", color="#e67e22")
        ax.bar(i + 0.27, x.RMSE_F, 0.27, label="C_GRU F (all)", color="#2c3e50")
        ax.set_xticks(i, [f"{o}\n(n={int(n)})" for o, n in zip(order, x.n)], rotation=45, fontsize=7)
        ax.set_title(f"{h} h ahead")
    axs[0].set_ylabel("RMSE (ug/m3)")
    axs[0].legend(fontsize=7)
    fig.suptitle("2023 validation RMSE by the route of the air arriving at the forecast origin")
    c.save(fig, "e9_rmse_by_cluster")

    piv = bc.F_minus_DN.unstack("horizon").reindex(order)
    fig, ax = c.plt.subplots(figsize=(7, 4))
    im = ax.imshow(piv.to_numpy(float), cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(H)), [f"{h} h" for h in H])
    ax.set_yticks(range(len(order)), order)
    for (i, j), v in np.ndenumerate(piv.to_numpy(float)):
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="RMSE F - RMSE DN")
    ax.set_title("Where foreign stations help (blue) or hurt (red), C_GRU, 2023")
    c.save(fig, "e9_f_minus_dn_by_cluster")

    fig, ax = c.plt.subplots(figsize=(8, 4))
    b = bl.bias_F.unstack("horizon")
    b.plot.bar(ax=ax)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xlabel("observed PM2.5 at the target time")
    ax.set_ylabel("mean error (pred - obs)")
    ax.set_title("C_GRU F bias by PM level: high PM is under-predicted, more at long horizons")
    ax.tick_params(axis="x", rotation=0)
    c.save(fig, "e9_bias_by_level")

    fig, axs = c.plt.subplots(1, 4, figsize=(16, 4))
    for ax, h in zip(axs, H):
        x = d[d.horizon.eq(h)]
        ax.scatter(x.y_true, x.F, s=3, alpha=0.3, c="#2c3e50")
        m = max(x.y_true.max(), x.F.max())
        ax.plot([0, m], [0, m], c="r", lw=0.8)
        ax.set_title(f"{h} h: RMSE {rmse(x.e_F):.2f}")
        ax.set_xlabel("observed")
    axs[0].set_ylabel("C_GRU F predicted")
    fig.suptitle("Observed vs predicted, 2023 validation")
    c.save(fig, "e9_obs_vs_pred")


if __name__ == "__main__":
    main()
