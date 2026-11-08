"""Region occlusion on the trained full-input (F) models, no retraining.

For each region in F (15 team regions), its nodes are made unobserved over the whole 48 h window
(value 0, mask 0 -- exactly how missing data looks in training), and validation/test origins are re-predicted.
The rise in error vs the unmasked F model = how much the trained model relies on that region (dependence),
to be read next to the retrained DROP_* runs (information value). Also writes per-origin prediction changes
so the dependence can be broken down by season / route / wind (when).

Usage: python occlude.py [--models C_GRU C_GCNGRU] [--data ROOT]
Output: results/C/runs/<model>_OCC/F_OCC_<region>/seed<k>/predictions.parquet
"""
import argparse

import numpy as np
import torch

import cdata
import train


def main(models, root):
    _, regions = cdata.conditions(root)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = train.load_condition(root, "F", dev)
    idx = np.r_[train.origins(data, "validation", root), train.origins(data, "test", root)]
    pos = {k: i for i, k in enumerate(data["node_keys"])}
    for m in models:
        for seed in (11, 22, 33):
            ck = cdata.RUNS / m / "F" / f"seed{seed}" / "model.pt"
            if not ck.exists():
                print("missing", ck)
                continue
            c = torch.load(ck, map_location=dev)
            assert c["node_keys"] == data["node_keys"]
            model = train.build(m, data, seed).to(dev)
            model.load_state_dict(c["state_dict"])
            x0 = data["x"]
            for r, keys in regions.items():
                nodes = [pos[k] for k in keys if k in pos]
                if len(nodes) < len(keys):   # region not (fully) inside F, e.g. the 11-07 control regions
                    continue
                d = cdata.RUNS / f"{m}_OCC" / f"F_OCC_{r}" / f"seed{seed}"
                if (d / "predictions.parquet").exists():
                    continue
                x = x0.clone()
                x[:, nodes, :] = 0.0
                data["x"] = x
                z = train.predict(model, data, idx)
                d.mkdir(parents=True, exist_ok=True)
                train.to_frame(data, idx, z, f"{m}_OCC", f"F_OCC_{r}", seed).to_parquet(d / "predictions.parquet", index=False)
                print(m, seed, r, len(nodes), flush=True)
            data["x"] = x0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["C_GRU", "C_GCNGRU"])
    ap.add_argument("--data")
    a = ap.parse_args()
    main(a.models, cdata.data_root(a.data))
