"""B figures, reusing C's map and forest helpers.

  b_shap_map_by_horizon.png   F model mean |SHAP| per region on the map (share of all station SHAP, %), 4 horizons
  b_regional_forest.png       ADD_r - DN and F - DROP_r RMSE differences with 95% CI
  b_vs_c_accuracy.png         best B and C configurations vs persistence by horizon

Usage: python bplots.py [--split validation] [--data ROOT]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "C_graph"))
import cdata  # noqa: E402
from plot_map import panel  # noqa: E402
from plot_summary import forest  # noqa: E402

B = REPO / "results" / "B"
H = (3, 6, 12, 24)


def main(split, root):
    out = B / "analysis" / split
    rh = pd.read_csv(out / "shap_region_horizon.csv", index_col=0)
    st = rh[~rh.index.str.startswith("COMMON_")]
    share = 100 * st / st.sum()                       # % of station-feature |SHAP| per horizon

    u = np.load(cdata.WORK / "c_universe.npz", allow_pickle=True)
    pos = dict(zip(u["keys"], zip(u["lon"], u["lat"])))
    _, regions = cdata.conditions(root)
    regions = {r: [k for k in ks if k in pos] for r, ks in regions.items() if r not in ("CN_EAST", "JP_EAST", "JP_WEST25")}
    stations = {r: np.array([pos[k] for k in ks]).T for r, ks in regions.items()}
    centres = {r: (np.median(xy[0]), np.median(xy[1])) for r, xy in stations.items()}
    centres["KR_CAPITAL"] = (126.6, 37.9)
    fig, axs = plt.subplots(2, 2, figsize=(17, 14))
    for ax, h in zip(axs.ravel(), H):
        panel(ax, centres, stations, share[str(h)].to_dict(), {}, f"{h} h ahead", vmax=40, fmt="{:.1f}%")
    fig.suptitle(f"B LightGBM (F, all regions): share of station-feature |SHAP| by region, {split}\n"
                 "what the prediction is built from (not accuracy). Seoul history / weather / calendar excluded from the 100%.",
                 fontsize=12)
    fig.tight_layout()
    (out / "maps").mkdir(exist_ok=True)
    fig.savefig(out / "maps" / "b_shap_map_by_horizon.png", dpi=120)
    plt.close(fig)

    t = pd.read_csv(B / "scores" / split / "regional" / "tests.csv")
    t = t[~t.versus_run_id.str.startswith(("persistence", "seasonal"))].copy()
    a, b = t.run_id.str.split("|").str[1], t.versus_run_id.str.split("|").str[1]
    t["lbl"] = [f"+{x[4:]} (ADD vs DN)" if x.startswith("ADD_") else f"-{y[5:]} (F vs DROP)" for x, y in zip(a, b)]
    fig, axs = plt.subplots(1, 4, figsize=(20, 7.5), sharey=True)
    for ax, h in zip(axs, H):
        forest(ax, t[t.horizon.eq(h)], f"LightGBM, {h} h")
    fig.suptitle("B regional value (left of 0 = the region helps; filled = significant by CI & Holm)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "b_regional_forest.png", dpi=110)
    plt.close(fig)

    def ens(path):
        m = pd.read_csv(path)
        return m, m[m.prediction_kind.eq("ensemble")]
    mb, eb = ens(B / "scores" / split / "core" / "metrics.csv")
    mc, ec = ens(cdata.RUNS.parent / "scores" / split / "core" / "metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    p = mb.groupby("horizon").persistence_RMSE.first()
    ax.plot(p.index, p.values, "k--", lw=2, label="Persistence")
    for lab, e, model, cond, c, mk in (("B LightGBM F", eb, "LightGBM", "F", "#1e8449", "D"), ("B LightGBM DN", eb, "LightGBM", "DN", "#82e0aa", "D"),
                                       ("C GRU F", ec, "C_GRU", "F", "#c0392b", "o"), ("C GCN-GRU F", ec, "C_GCNGRU", "F", "#2471a3", "s")):
        g = e[e.model.eq(model) & e.input_condition.eq(cond)].sort_values("horizon")
        ax.plot(g.horizon, g.RMSE, marker=mk, c=c, label=lab)
    ax.set_xticks(H)
    ax.set_xlabel("horizon (h)")
    ax.set_ylabel(f"RMSE (µg/m³), {split}")
    ax.set_title("B vs C accuracy (seed ensembles): LightGBM best to 12 h, GRU best at 24 h", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "b_vs_c_accuracy.png", dpi=130)
    plt.close(fig)
    print("B figures ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--data")
    a = ap.parse_args()
    main(a.split, cdata.data_root(a.data))
