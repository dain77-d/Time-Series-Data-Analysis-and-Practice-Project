"""Score every C family with the team's common scorer (Model/Common/score.py) and build hypothesis tables.

Families (each is one scorer invocation = one Holm family; comparisons fixed here, before results):
  core      E1+E2  H4a GCN vs GRU per condition; real vs permuted graph; DN vs D67; F vs DN
  regional  E3     per model: ADD_r vs DN (incremental value), F vs DROP_r (unique value)
  controls  E4     per model: CTRL_r vs DN (low-passage regions)
  occlusion        per model: F vs F_OCC_r (dependence of the trained F model)
Validation by default; --split test --final opens 2024 once, after everything is frozen.

Usage: python score_all.py [--split validation|test] [--final] [--data ROOT]
Output: results/C/scores/<split>/<family>/{metrics,tests,conditions,seed_metric_summary}.csv + summary_<split>.md
"""
import argparse
import subprocess
import sys

import pandas as pd

import cdata
from experiments import ADD, CTRL

RUNS = cdata.RUNS


def files(model, conds):
    out = []
    for c in conds:
        out += sorted((RUNS / model / c).glob("seed*/predictions.parquet"))
    return out


def e(model, cond):
    return f"{model}|{cond}|ensemble"


def n_seeds(model, cond):
    return len(list((RUNS / model / cond).glob("seed*/predictions.parquet")))


def fair(pair):
    """'a:b' of two ensembles -> seed-11 vs seed-11 when either side has fewer than 3 seeds (ensembles must match in size)."""
    a, b = pair.split(":")
    if all(n_seeds(*x.split("|")[:2]) == 3 for x in (a, b)):
        return pair
    return ":".join(x.replace("|ensemble", "|seed|seed=11") for x in (a, b))


def families(root):
    conds, regions = cdata.conditions(root)
    drop = sorted(c for c in conds if c.startswith("DROP_"))
    occ = [f"F_OCC_{r}" for r in regions if all(k in set(conds["F"]) for k in regions[r])]
    fam = {"core": (files("C_GRU", ["D67", "DN", "F"]) + files("C_GCNGRU", ["D67", "DN", "F"]) + files("C_GCNGRU_PERM", ["F"]),
                    [f"{e('C_GCNGRU', c)}:{e('C_GRU', c)}" for c in ("D67", "DN", "F")]
                    + [f"{e('C_GCNGRU', 'F')}:{e('C_GCNGRU_PERM', 'F')}"]
                    + [f"{e(m, 'DN')}:{e(m, 'D67')}" for m in ("C_GRU", "C_GCNGRU")]
                    + [f"{e(m, 'F')}:{e(m, 'DN')}" for m in ("C_GRU", "C_GCNGRU")])}
    for m in ("C_GRU", "C_GCNGRU"):
        fam[f"regional_{m}"] = (files(m, ["DN", "F"] + ADD + drop),
                                [f"{e(m, a)}:{e(m, 'DN')}" for a in ADD] + [f"{e(m, 'F')}:{e(m, d)}" for d in drop])
        fam[f"controls_{m}"] = (files(m, ["DN"] + CTRL), [f"{e(m, c)}:{e(m, 'DN')}" for c in CTRL])
        fam[f"occlusion_{m}"] = (files(m, ["F"]) + files(f"{m}_OCC", occ), [f"{e(m, 'F')}:{e(m + '_OCC', o)}" for o in occ])
    return fam


def main(split, final, root):
    score = root / "Model" / "Common" / "score.py"
    out = cdata.REPO / "results" / "C" / "scores" / split
    lines = [f"# C scores ({split})\n"]
    for name, (fs, pairs) in families(root).items():
        if not fs:
            print("skip (no predictions):", name)
            continue
        d = out / name
        cmd = [sys.executable, str(score), "--predictions", *map(str, fs), "--split", split, "--output-dir", str(d)]
        cmd += ["--final"] if final else []
        # keep only pairs whose runs exist
        have = {(p.parts[-4], p.parts[-3]) for p in fs}
        ok = [fair(p) for p in pairs if all(tuple(x.split("|")[:2]) in have for x in p.split(":"))]
        if ok:
            cmd += ["--pairs", *ok]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if r.returncode:
            print(name, "FAILED\n", r.stderr[-2000:])
            continue
        t = pd.read_csv(d / "tests.csv")
        t = t[~t.versus_run_id.astype(str).str.startswith(("persistence", "seasonal_naive"))]
        lines += [f"\n## {name}\n", t.to_markdown(index=False) if len(t) else "(no pair tests)", "\n"]
        print("scored", name, len(fs), "files")
    (out / f"summary_{split}.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--data")
    a = ap.parse_args()
    main(a.split, a.final, cdata.data_root(a.data))
