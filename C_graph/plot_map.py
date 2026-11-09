"""C results on a map: how much the C_GRU F model leans on each region (occlusion dependence, % RMSE).

Dots = stations coloured by region; circle at each region's centre = dependence (area ~ value, number printed).
Thick outline = significant by the team rule (block-bootstrap CI upper < 0 and Holm p < 0.05).
  map_dependence_by_horizon.png   4 panels: 3, 6, 12, 24 h
  map_dependence_season_h12.png   cold vs warm season, 12 h (when is the model leaning on China?)
  map_unique_vs_dependence_h12.png  dependence vs unique info (DROP retrain), 12 h

Usage: python plot_map.py [--model C_GRU] [--split validation] [--data ROOT]
Land outline: assets/ne_50m_land.geojson (Natural Earth, public domain).
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cdata  # noqa: E402

LAND = cdata.REPO / "assets" / "ne_50m_land.geojson"
EXTENT = (104, 138, 30, 50)
COLORS = {"KR": "#d07a2d", "CN": "#a23b8f", "JP": "#2e7d5b", "MN": "#5b3f8f"}


def land(ax):
    for f in json.loads(LAND.read_text(encoding="utf-8"))["features"]:
        g = f["geometry"]
        for poly in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            x, y = zip(*poly[0])
            if max(x) < EXTENT[0] or min(x) > EXTENT[1] or max(y) < EXTENT[2] or min(y) > EXTENT[3]:
                continue
            ax.fill(x, y, color="#f4f4f1", ec="#888", lw=0.4, zorder=0)
    ax.set_facecolor("#e3ecf3")
    ax.set_xlim(EXTENT[:2])
    ax.set_ylim(EXTENT[2:])
    ax.set_aspect(1.25)
    ax.set_xticks([])
    ax.set_yticks([])


# Korean regions sit close together: their labels go out to the sea with a leader line (lon, lat of the label)
LABEL_AT = {"KR_CAPITAL": (123.3, 39.0), "KR_CHUNGCHEONG": (123.9, 35.9), "KR_HONAM": (123.6, 33.6), "KR_JEJU": (125.9, 31.4),
            "KR_GANGWON": (133.5, 40.0), "CN_SHANDONG": (117.6, 34.6), "KR_DAEGYEONG": (134.0, 37.4), "KR_DONGNAM": (134.0, 35.0)}
SHORT = {"KR_CAPITAL": "Capital", "KR_GANGWON": "Gangwon", "KR_CHUNGCHEONG": "Chungcheong", "KR_HONAM": "Honam",
         "KR_DAEGYEONG": "Daegu-Gyeongbuk", "KR_DONGNAM": "Busan-Ulsan-Gyeongnam", "KR_JEJU": "Jeju", "CN_SHANDONG": "Shandong",
         "CN_BTH": "Beijing-Tianjin-Hebei", "CN_LIAONING": "Liaoning", "CN_JILIN": "Jilin", "CN_HEILONGJIANG": "Heilongjiang",
         "CN_INNER_MONGOLIA": "Inner Mongolia", "JP": "W. Japan", "MN": "Mongolia"}


def panel(ax, centres, stations, values, sig, title, vmax, fmt="{:+.1f}"):
    land(ax)
    for r, (x, y) in stations.items():
        ax.scatter(x, y, s=5, c=COLORS[r[:2]], alpha=0.6, lw=0, zorder=1)
    ax.plot(126.997, 37.564, marker="*", ms=16, c="gold", mec="k", zorder=6)
    for r, (x, y) in centres.items():
        v = values.get(r, np.nan)
        if not np.isfinite(v):
            continue
        kr = r.startswith("KR_")
        size = (900 if kr else 2400) * np.sqrt(min(abs(v), vmax) / vmax) + 15
        ax.scatter(x, y, s=size, c="#c0392b" if v > 0 else "#2e86c1", alpha=0.5,
                   ec="k" if sig.get(r) else "none", lw=2.2 if sig.get(r) else 0, zorder=3)
        tx, ty = LABEL_AT.get(r, (x, y - 1.1))
        if r in LABEL_AT:
            ax.plot([x, tx], [y, ty], c="#555", lw=0.6, zorder=2)
        ax.text(tx, ty, f"{SHORT[r]}\n{fmt.format(v)}", ha="center", va="center", fontsize=8.5, zorder=5,
                weight="bold" if sig.get(r) else "normal",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    ax.set_title(title, fontsize=12)


def main(model, split, root):
    u = np.load(cdata.WORK / "c_universe.npz", allow_pickle=True)
    pos = dict(zip(u["keys"], zip(u["lon"], u["lat"])))
    _, regions = cdata.conditions(root)
    regions = {r: [k for k in ks if k in pos] for r, ks in regions.items() if r not in ("CN_EAST", "JP_EAST", "JP_WEST25")}
    stations = {r: np.array([pos[k] for k in ks]).T for r, ks in regions.items()}
    centres = {r: (np.median(xy[0]), np.median(xy[1])) for r, xy in stations.items()}
    centres["KR_CAPITAL"] = (126.6, 37.9)          # keep the label off the Seoul star
    a = cdata.RUNS.parent / "analysis" / split
    out = a / "maps"
    out.mkdir(parents=True, exist_ok=True)
    h = pd.read_csv(a / "region_contribution_by_horizon.csv")
    h = h[h.model.eq(model)]
    t = pd.read_csv(cdata.RUNS.parent / "scores" / split / f"occlusion_{model}" / "tests.csv")
    t = t[t.versus_run_id.str.contains("_OCC|", regex=False)]
    t["region"] = t.versus_run_id.str.split("|").str[1].str.replace("F_OCC_", "")
    legend = "circle area ~ RMSE increase when the region is hidden (%); thick outline = significant (CI & Holm); star = Seoul Jung-gu"

    fig, axs = plt.subplots(2, 2, figsize=(17, 14))
    for ax, hz in zip(axs.ravel(), (3, 6, 12, 24)):
        v = h[h.horizon.eq(hz)].set_index("region").dependence_occlusion_pct.to_dict()
        s = t[t.horizon.eq(hz)].set_index("region").improved.to_dict()
        panel(ax, centres, stations, v, s, f"{hz} h ahead", vmax=40)
    fig.suptitle(f"{model} (F, all 600 stations): dependence on each region, 2023 validation\n{legend}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "map_dependence_by_horizon.png", dpi=120)
    plt.close(fig)

    c = pd.read_csv(a / "region_contribution_by_condition.csv")
    c = c[c.model.eq(model) & c.horizon.eq(12) & c.condition.eq("cold_season")]
    fig, axs = plt.subplots(1, 2, figsize=(17, 7.5))
    for ax, lvl in zip(axs, ("cold", "warm")):
        cc = c[c.level.eq(lvl)]
        panel(ax, centres, stations, cc.set_index("region").dependence_pct.to_dict(), {}, f"{lvl} season (n={int(cc.n.iloc[0])})", vmax=40)
    fig.suptitle(f"{model}: dependence by season, 12 h ahead (% RMSE increase when hidden)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "map_dependence_season_h12.png", dpi=120)
    plt.close(fig)

    h12 = h[h.horizon.eq(12)].set_index("region")
    fig, axs = plt.subplots(1, 2, figsize=(17, 7.5))
    s12 = t[t.horizon.eq(12)].set_index("region").improved.to_dict()
    panel(axs[0], centres, stations, h12.dependence_occlusion_pct.to_dict(), s12, "dependence: region hidden in the trained model", vmax=40)
    panel(axs[1], centres, stations, h12.unique_info_drop_pct.to_dict(), {}, "unique info: model retrained without the region\n(circle scale x8 vs left)", vmax=5)
    fig.suptitle(f"{model}, 12 h: the model leans on many regions, but almost all of it can be replaced\n"
                 "(red = removing the region hurts, blue = removing it helps; % RMSE)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "map_unique_vs_dependence_h12.png", dpi=120)
    plt.close(fig)
    print("maps ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="C_GRU")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--data")
    a = ap.parse_args()
    main(a.model, a.split, cdata.data_root(a.data))
