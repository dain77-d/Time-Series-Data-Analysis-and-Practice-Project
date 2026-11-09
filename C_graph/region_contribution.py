"""How much each region helps the C models predict Seoul PM2.5: tables and heatmaps.

Two complementary measures per region r, horizon h, on the fixed eligible rows (seed-ensemble, ug/m3):
  dependence  = RMSE(F with r occluded) - RMSE(F)           trained F model, r hidden at prediction time
  unique info = RMSE(DROP_r retrained)  - RMSE(F)           model retrained without r (substitution allowed)
  added info  = RMSE(DN) - RMSE(ADD_r retrained)            only for foreign regions (team ADD conditions)
Positive = the region helps. Dependence is also broken down by season, route sector, surface wind and
trajectory-level wind (when does the model lean on r?). Significance comes from score_all.py (DM, block CI, Holm).

Usage: python region_contribution.py [--split validation|test] [--final] [--data ROOT]
Output: results/C/analysis/<split>/region_contribution_*.csv and *.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cdata  # noqa: E402

RUNS = cdata.RUNS
CONDS = ["cold_season", "route_group", "wind_sector", "traj_wind_sector"]


def seeds(model, cond):
    return {p.parent.name for p in (RUNS / model / cond).glob("seed*/predictions.parquet")}


def ensemble(model, cond, split, only=None):
    """seed-mean prediction; `only` restricts to these seed dirs so both sides of a difference use the same seeds."""
    fs = sorted(f for f in (RUNS / model / cond).glob("seed*/predictions.parquet") if only is None or f.parent.name in only)
    if not fs:
        return None
    p = pd.concat([pd.read_parquet(f) for f in fs])
    p = p[p.split.eq(split)].groupby(["origin_time_kst", "horizon"]).y_hat.mean().clip(lower=0)
    return p.rename(f"{model}|{cond}")


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def main(split, root):
    grid = pd.read_parquet(root / "Dataset" / "Evaluation_Contract_20261107" / "evaluation_grid.parquet")
    grid = grid[grid.split.eq(split) & grid.eligible].set_index(["origin_time_kst", "horizon"])
    _, regions = cdata.conditions(root)
    conds, _ = cdata.conditions(root)
    out = RUNS.parent / "analysis" / split
    out.mkdir(parents=True, exist_ok=True)
    rows, cond_rows = [], []
    for m in ("C_GRU", "C_GCNGRU"):
        base = ensemble(m, "F", split)
        dn = ensemble(m, "DN", split)
        if base is None:
            continue
        g = grid.join(base, how="inner")
        for r, keys in regions.items():
            if not all(k in set(conds["F"]) for k in keys):
                continue
            occ, drop, add = ensemble(f"{m}_OCC", f"F_OCC_{r}", split), ensemble(m, f"DROP_{r}", split), ensemble(m, f"ADD_{r}", split)
            for h, gh in g.groupby(level="horizon"):
                y, f0 = gh.y_true.to_numpy(), gh[base.name].to_numpy()
                rec = {"model": m, "region": r, "n_stations": len(keys), "horizon": h, "rmse_F": rmse(f0 - y)}
                if occ is not None:
                    rec["dependence_occlusion"] = rmse(occ.reindex(gh.index).to_numpy() - y) - rec["rmse_F"]
                    rec["dependence_occlusion_pct"] = 100 * rec["dependence_occlusion"] / rec["rmse_F"]
                if drop is not None:            # F over the seeds DROP_r has (seed-ensemble size must match)
                    sd = seeds(m, f"DROP_{r}")
                    fb = ensemble(m, "F", split, sd).reindex(gh.index).to_numpy()
                    rec["unique_info_drop"] = rmse(drop.reindex(gh.index).to_numpy() - y) - rmse(fb - y)
                    rec["unique_info_drop_pct"] = 100 * rec["unique_info_drop"] / rmse(fb - y)
                    rec["n_seeds_drop"] = len(sd)
                if add is not None and dn is not None:
                    sa = seeds(m, f"ADD_{r}")
                    a_, d_ = add.reindex(gh.index).to_numpy(), ensemble(m, "DN", split, sa).reindex(gh.index).to_numpy()
                    rec["added_info_add"] = rmse(d_ - y) - rmse(a_ - y)
                    rec["n_seeds_add"] = len(sa)
                rows.append(rec)
                if occ is not None:
                    v = occ.reindex(gh.index).to_numpy()
                    for c in CONDS:
                        for lvl, idx in gh.groupby(c).indices.items():
                            cond_rows.append({"model": m, "region": r, "horizon": h, "condition": c, "level": lvl, "n": len(idx),
                                              "dependence_pct": 100 * (rmse(v[idx] - y[idx]) / rmse(f0[idx] - y[idx]) - 1)})
    if not rows:
        print("no F predictions yet")
        return
    t = pd.DataFrame(rows)
    t.to_csv(out / "region_contribution_by_horizon.csv", index=False, encoding="utf-8-sig")
    c = pd.DataFrame(cond_rows)
    if len(c):
        c.to_csv(out / "region_contribution_by_condition.csv", index=False, encoding="utf-8-sig")
    for m, tm in t.groupby("model"):
        cols = [x for x in ("dependence_occlusion_pct", "unique_info_drop_pct") if x in tm]
        if not cols:
            continue
        fig, axes = plt.subplots(1, len(cols), figsize=(5.5 * len(cols), 6), squeeze=False)
        for ax, col in zip(axes[0], cols):
            pv = tm.pivot(index="region", columns="horizon", values=col)
            lim = np.nanmax(np.abs(pv.to_numpy())) or 1
            im = ax.imshow(pv.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            ax.set(xticks=range(pv.shape[1]), xticklabels=[f"{h}h" for h in pv.columns], yticks=range(len(pv)), yticklabels=pv.index,
                   title=f"{m}: {col.replace('_pct', '')} (% RMSE)")
            for i in range(pv.shape[0]):
                for j in range(pv.shape[1]):
                    ax.text(j, i, f"{pv.iat[i, j]:+.1f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(out / f"region_contribution_{m}.png", dpi=150)
        plt.close(fig)
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--data")
    a = ap.parse_args()
    assert a.split == "validation" or a.final, "test opens only with --final"
    main(a.split, cdata.data_root(a.data))
