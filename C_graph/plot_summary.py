"""C summary charts from the team scorer output (no recomputation).

  c_accuracy_by_horizon.png   RMSE of every C model x condition (seed ensemble) vs persistence, per horizon
  c_core_tests.png            the core pair tests (GCN vs GRU, real vs permuted graph, DN vs D67, F vs DN) with 95% CI
  c_regional_forest.png       ADD_r - DN and F - DROP_r RMSE differences with 95% CI, both models, 4 horizons

Negative difference = the first model has the smaller error. Filled marker = significant (CI upper < 0 and Holm p < 0.05).
Usage: python plot_summary.py [--split validation]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import cdata  # noqa: E402

H = (3, 6, 12, 24)
STYLE = {"C_GRU": ("#c0392b", "o"), "C_GCNGRU": ("#2471a3", "s"), "C_GCNGRU_PERM": ("#7f8c8d", "x")}


def tests(d, fam):
    t = pd.read_csv(d / fam / "tests.csv")
    t = t[~t.versus_run_id.str.startswith(("persistence", "seasonal"))].copy()
    t["a"] = t.run_id.str.split("|").str[:2].str.join("|")
    t["b"] = t.versus_run_id.str.split("|").str[:2].str.join("|")
    return t


def forest(ax, t, label):
    t = t.reset_index(drop=True)
    for i, r in t.iterrows():
        c = "#c0392b" if r.rmse_diff < 0 else "#2471a3"
        ax.plot([r.ci95_low, r.ci95_high], [i, i], c=c, lw=1.4)
        ax.plot(r.rmse_diff, i, "o", c=c, mfc=c if r.improved else "white", ms=6)
    ax.axvline(0, c="k", lw=0.8)
    ax.set_yticks(range(len(t)), t.lbl, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(label, fontsize=10)


def main(split):
    d = cdata.RUNS.parent / "scores" / split
    out = cdata.RUNS.parent / "analysis" / split
    out.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(d / "core" / "metrics.csv")
    e = m[m.prediction_kind.eq("ensemble")]
    fig, ax = plt.subplots(figsize=(8, 5))
    p = m.groupby("horizon").persistence_RMSE.first()
    ax.plot(p.index, p.values, "k--", lw=2, label="Persistence")
    for (model, cond), g in e.groupby(["model", "input_condition"]):
        col, mk = STYLE[model]
        ls = {"D67": ":", "DN": "-.", "F": "-"}[cond]
        ax.plot(g.horizon, g.RMSE, ls=ls, marker=mk, c=col, label=f"{model} {cond}")
    ax.set_xticks(H)
    ax.set_xlabel("horizon (h)")
    ax.set_ylabel("RMSE (µg/m³), 2023")
    ax.set_title("C accuracy by horizon (seed ensemble, 2023)\nplain GRU is best; GCN beats only its permuted-graph control", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "c_accuracy_by_horizon.png", dpi=130)
    plt.close(fig)

    t = tests(d, "core")
    t["lbl"] = t.a + "  vs  " + t.b
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, h in zip(axs, H):
        forest(ax, t[t.horizon.eq(h)], f"{h} h")
    fig.suptitle("C core tests: RMSE(first) - RMSE(second), 95% block-bootstrap CI (filled = significant improvement)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "c_core_tests.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(2, 4, figsize=(20, 13), sharey="row")
    for row, model in enumerate(("C_GRU", "C_GCNGRU")):
        t = tests(d, f"regional_{model}")
        t["lbl"] = [f"+{a.split('|')[1][4:]} (ADD vs DN)" if "ADD_" in a else f"-{b.split('|')[1][5:]} (F vs DROP)"
                    for a, b in zip(t.a, t.b)]
        for ax, h in zip(axs[row], H):
            forest(ax, t[t.horizon.eq(h)], f"{model}, {h} h")
    fig.suptitle("C regional value. ADD: adding a foreign region to all-Korea DN (left of 0 = helps).\n"
                 "F vs DROP: full model vs retrained without the region (left of 0 = the region carries unique info). "
                 "GCN DROP = seed 11 only in round 1.", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "c_regional_forest.png", dpi=110)
    plt.close(fig)
    print("summary charts ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="validation")
    main(ap.parse_args().split)
