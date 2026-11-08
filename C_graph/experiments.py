"""The fixed C experiment list (decided before any result is seen) and a resumable runner.

E1 core (H4a)            C_GRU, C_GCNGRU x {D67, DN, F}
E2 graph control         C_GCNGRU_PERM x {F}          (random geography, same degree/weights)
E3 regional value (RQ1)  C_GRU, C_GCNGRU x {ADD_* 8, DROP_* 15}   (retrained, team conditions)
E4 low-passage controls  C_GRU, C_GCNGRU x {CTRL_CN_EAST, CTRL_JP_EAST, CTRL_JP_WEST25}
Every run uses seeds 11, 22, 33. Order: E1, E2, E4, E3 (the most important first).

Usage:
  python experiments.py --list                      # print the plan
  python experiments.py --workers 2 [--only E1,E2]  # run what is missing; safe to restart
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cdata

SEEDS = (11, 22, 33)
ADD = ["ADD_CN_SHANDONG", "ADD_CN_BTH", "ADD_CN_LIAONING", "ADD_CN_JILIN", "ADD_CN_HEILONGJIANG", "ADD_CN_INNER_MONGOLIA", "ADD_JP", "ADD_MN"]
CTRL = ["CTRL_CN_EAST", "CTRL_JP_EAST", "CTRL_JP_WEST25"]


def plan(root):
    conds, _ = cdata.conditions(root)
    drop = sorted(c for c in conds if c.startswith("DROP_"))
    add = [c for c in ADD if c in conds]
    assert len(drop) == 15 and len(add) == 8, (len(drop), len(add))
    jobs = []
    for fam, models, cs in (("E1", ["C_GRU", "C_GCNGRU"], ["D67", "DN", "F"]),
                            ("E2", ["C_GCNGRU_PERM"], ["F"]),
                            ("E4", ["C_GRU", "C_GCNGRU"], CTRL),
                            ("E3", ["C_GRU", "C_GCNGRU"], add + drop)):
        for c in cs:
            for m in models:
                for s in SEEDS:
                    jobs.append({"family": fam, "model": m, "condition": c, "seed": s})
    return jobs


def done(job, out):
    return (out / job["model"] / job["condition"] / f"seed{job['seed']}" / "predictions.parquet").exists()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    root = cdata.data_root(a.data)
    out = cdata.REPO / "results" / "C" / "runs"
    jobs = [j for j in plan(root) if not a.only or j["family"] in a.only.split(",")]
    if a.list:
        from collections import Counter
        print(json.dumps({"total": len(jobs), "by_family": Counter(j["family"] for j in jobs),
                          "by_model": Counter(j["model"] for j in jobs), "missing": sum(not done(j, out) for j in jobs)}, indent=1))
        sys.exit()
    log = cdata.REPO / "results" / "C" / "logs"
    log.mkdir(parents=True, exist_ok=True)
    todo, running = [j for j in jobs if not done(j, out)], []
    while todo or running:
        while todo and len(running) < a.workers:
            j = todo.pop(0)
            name = f"{j['model']}__{j['condition']}__seed{j['seed']}"
            f = open(log / f"{name}.log", "w", encoding="utf-8")
            p = subprocess.Popen([sys.executable, str(Path(__file__).parent / "train.py"), "--model", j["model"], "--condition", j["condition"],
                                  "--seed", str(j["seed"]), "--data", str(root), "--epochs", str(a.epochs)], stdout=f, stderr=subprocess.STDOUT)
            running.append((p, f, name, time.time()))
            print(f"start {name}  (left {len(todo)})", flush=True)
        for r in running[:]:
            if r[0].poll() is not None:
                r[1].close()
                running.remove(r)
                print(f"{'done' if r[0].returncode == 0 else 'FAILED'} {r[2]} {time.time() - r[3]:.0f}s", flush=True)
        time.sleep(5)
