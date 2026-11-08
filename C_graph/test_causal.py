"""Leakage check: a C model's prediction at origin t must not change when every value after t is changed.

Usage: python test_causal.py [--data ROOT]   (needs work/c_universe.npz; CPU, ~1 min)
"""
import argparse

import numpy as np
import torch

import cdata
import train

ap = argparse.ArgumentParser()
ap.add_argument("--data")
root = cdata.data_root(ap.parse_args().data)
data = train.load_condition(root, "D67", "cpu")
va = train.origins(data, "validation", root)[:64]
for name in ("C_GRU", "C_GCNGRU"):
    torch.manual_seed(0)
    model = train.build(name, data, 11)
    before = train.predict(model, data, va)
    x0, g0 = data["x"].clone(), data["glob"].clone()
    cut = int(va.max()) + 1
    data["x"][cut:] = 99.0
    data["glob"][cut:] = 99.0
    after = train.predict(model, data, va)
    data["x"], data["glob"] = x0, g0
    assert np.array_equal(before, after), name
    print(name, "causal OK on", len(va), "origins")
