"""B: score the LightGBM runs with the team scorer and turn the TreeSHAP tables into region figures.

Score families (same comparisons as C, one Holm family each):
  core      DN vs D67, F vs DN
  regional  ADD_r vs DN (8), F vs DROP_r (15)
  controls  CTRL_r vs DN (3)
SHAP (F model, seed mean; log1p scale, contribution to the prediction, not to accuracy):
  shap_region_horizon   mean |SHAP| per region and horizon
  shap_region_stat      mean |SHAP| per region x statistic (now / lagXh / meanXh / change ...) at 6 h
  shap_region_condition mean |SHAP| of foreign regions per season / route / surface wind / trajectory wind
  shap_high_pm          signed region contributions on the 10 highest-PM 2023 targets (6 h)
SHAP uses 2023 rows only unless --split test --final.

Usage: python banalysis.py [--split validation|test] [--final] [--data ROOT]
Output: results/B/scores/<split>/<family>/..., results/B/analysis/<split>/*.csv|png
"""
import argparse
import subprocess
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
from experiments import ADD, CTRL  # noqa: E402

B = REPO / "results" / "B"
RUNS = B / "runs" / "LightGBM"
H = (3, 6, 12, 24)
CONDS = ["cold_season", "route_group", "wind_sector", "traj_wind_sector"]


def e(c):
    return f"LightGBM|{c}|ensemble"


def score(root, split, final):
    conds, _ = cdata.conditions(root)
    drop = sorted(c for c in conds if c.startswith("DROP_"))
    fams = {"core": (["D67", "DN", "F"], [f"{e('DN')}:{e('D67')}", f"{e('F')}:{e('DN')}"]),
            "regional": (["DN", "F"] + ADD + drop, [f"{e(a)}:{e('DN')}" for a in ADD] + [f"{e('F')}:{e(d)}" for d in drop]),
            "controls": (["DN"] + CTRL, [f"{e(c)}:{e('DN')}" for c in CTRL])}
    lines = [f"# B scores ({split})\n"]
    for name, (cs, pairs) in fams.items():
        fs = [f for c in cs for f in sorted((RUNS / c).glob("seed*/predictions.parquet"))]
        have = {f.parts[-3] for f in fs}
        pairs = [p for p in pairs if all(x.split("|")[1] in have for x in p.split(":"))]
        d = B / "scores" / split / name
        cmd = [sys.executable, str(root / "Model" / "Common" / "score.py"), "--predictions", *map(str, fs), "--split", split,
               "--output-dir", str(d)] + (["--final"] if final else []) + (["--pairs", *pairs] if pairs else [])
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if r.returncode:
            print(name, "FAILED\n", r.stderr[-2000:])
            continue
        t = pd.read_csv(d / "tests.csv")
        t = t[~t.versus_run_id.astype(str).str.startswith(("persistence", "seasonal_naive"))]
        lines += [f"\n## {name}\n", t.to_markdown(index=False) if len(t) else "(no pair tests)", "\n"]
        print("scored", name, len(fs))
    (B / "scores" / split / f"summary_{split}.md").write_text("\n".join(lines), encoding="utf-8")


def heat(df, title, path, fmt="{:.3f}", cmap="viridis"):
    fig, ax = plt.subplots(figsize=(1.2 + 0.75 * df.shape[1], 0.8 + 0.38 * df.shape[0]))
    im = ax.imshow(df.to_numpy(float), aspect="auto", cmap=cmap)
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=40, ha="right")
    ax.set_yticks(range(df.shape[0]), df.index)
    for (i, j), v in np.ndenumerate(df.to_numpy(float)):
        if np.isfinite(v):
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7, color="w")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def shap(root, split):
    d = B / "shap" / "F"
    out = B / "analysis" / split
    out.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(root / "Dataset" / "Evaluation_Contract_20261107" / "evaluation_grid.parquet")
    grid = grid[grid.split.eq(split) & grid.eligible]
    per_h, cond_rows = {}, []
    for h in H:
        fs = sorted(d.glob(f"h{h}_seed*_region.parquet"))
        if not fs:
            continue
        s = pd.concat([pd.read_parquet(f) for f in fs])
        s = s[s.split.eq(split)].drop(columns="split").groupby("origin_time_kst").mean()   # seed mean, signed
        per_h[h] = s.abs().mean()
        g = grid[grid.horizon.eq(h)].set_index("origin_time_kst").join(s, how="inner")
        foreign = [c for c in s.columns if not c.startswith(("KR_", "COMMON_"))]
        for c in CONDS:
            m = g.groupby(c)[foreign].apply(lambda x: x.abs().mean())
            for lvl, row in m.iterrows():
                cond_rows += [{"horizon": h, "condition": c, "level": lvl, "region": r, "mean_abs_shap": v, "n": int((g[c] == lvl).sum())}
                              for r, v in row.items()]
        if h == 6:
            top = g.nlargest(10, "y_true")
            hp = top[s.columns].assign(y_true=top.y_true)
            hp.to_csv(out / "shap_high_pm_h6.csv", encoding="utf-8-sig")
            cols = [c for c in s.columns]
            fig, ax = plt.subplots(figsize=(10, 4.5))
            bottom_p, bottom_n = np.zeros(len(top)), np.zeros(len(top))
            cmap = plt.get_cmap("tab20")
            for k, c in enumerate(cols):
                v = top[c].to_numpy()
                b = np.where(v >= 0, bottom_p, bottom_n)
                ax.bar(range(len(top)), v, bottom=b, color=cmap(k % 20), label=c)
                bottom_p += np.clip(v, 0, None)
                bottom_n += np.clip(v, None, 0)
            ax.set_xticks(range(len(top)), [f"{t:%m-%d %H}h\n{y:.0f}" for t, y in zip(top.index, top.y_true)], fontsize=7)
            ax.set_ylabel("SHAP (log1p)")
            ax.set_title(f"B F model: region contributions on the 10 highest-PM2.5 targets, 6 h ({split})")
            ax.legend(fontsize=6, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
            fig.tight_layout()
            fig.savefig(out / "shap_high_pm_h6.png", dpi=130)
            plt.close(fig)
    if not per_h:
        print("no SHAP tables yet")
        return
    rh = pd.DataFrame(per_h).sort_values(6 if 6 in per_h else list(per_h)[0], ascending=False)
    rh.to_csv(out / "shap_region_horizon.csv", encoding="utf-8-sig")
    heat(rh.rename(columns=lambda h: f"{h} h"), f"B F model: mean |SHAP| by region and horizon ({split})", out / "shap_region_horizon.png")
    ct = pd.DataFrame(cond_rows)
    ct.to_csv(out / "shap_region_condition.csv", index=False, encoding="utf-8-sig")
    c6 = ct[ct.horizon.eq(6)].assign(col=lambda x: x.condition + "=" + x.level.astype(str))
    heat(c6.pivot(index="region", columns="col", values="mean_abs_shap"), f"B F model: foreign-region mean |SHAP| by condition, 6 h ({split})",
         out / "shap_region_condition_h6.png", cmap="magma")
    st = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("h6_seed*_region_stat.parquet"))]).groupby(["region", "stat"]).mean_abs_shap.mean()
    st = st.unstack("stat")
    st = st.loc[st.sum(1).sort_values(ascending=False).index]
    st.to_csv(out / "shap_region_stat_h6.csv", encoding="utf-8-sig")
    heat(st[st.columns[st.sum().argsort()[::-1]]], "B F model: mean |SHAP| by region x statistic, 6 h (2023+2024 rows)",
         out / "shap_region_stat_h6.png", fmt="{:.2f}", cmap="cividis")
    print("SHAP figures ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--data")
    a = ap.parse_args()
    if a.split == "test" and not a.final:
        sys.exit("2024 is opened once: pass --final")
    root = cdata.data_root(a.data)
    score(root, a.split, a.final)
    shap(root, a.split)
