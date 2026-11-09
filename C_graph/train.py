"""Train one C run: model x input condition x seed, then predict every validation/test grid origin.

Models (hyperparameters fixed in advance; same for every condition):
  C_GRU          all condition nodes concatenated per hour -> Linear -> GRU(64); no graph
  C_GCNGRU       2-layer GCN (16 ch) on the distance graph each hour -> node-wise GRU(32) -> [Seoul, node mean, globals]
  C_GCNGRU_PERM  same as C_GCNGRU but the graph's node labels are randomly permuted (Seoul fixed): same degree and
                 weight distribution, wrong geography. Control for "graph structure vs extra parameters".
Input per origin t: last 48 h x nodes x [standardized log1p PM2.5 (0 if missing), observed mask] + globals at t
(ASOS 108/102 + hour/annual sin-cos). Output: standardized log1p Seoul PM2.5 at t+3/6/12/24 (masked MSE), as last observed Seoul value + learned change.
Training on hourly 2021-22 origins; early stopping on the 2023 3-hourly grid; 2024 predicted with the frozen model.

Usage: python -I train.py --model C_GCNGRU --condition F --seed 11 [--data ROOT] [--epochs 40] [--max-batches N]
Output: results/C/runs/<model>/<condition>/seed<seed>/{predictions.parquet, meta.json, model.pt}
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import cdata

WINDOW, H = 48, np.array(cdata.HORIZONS)
OUT = cdata.RUNS


# ------------------------------------------------------------------ data
def load_condition(root, condition, device):
    u = np.load(cdata.WORK / "c_universe.npz")
    conds, _ = cdata.conditions(root)
    keys = list(u["keys"])
    pos = {k: i for i, k in enumerate(keys)}
    node_keys = [cdata.TARGET] + conds[condition]
    idx = np.array([pos[k] for k in node_keys])
    kst = pd.to_datetime(u["hours_utc"], unit="h", utc=True).tz_convert("Asia/Seoul")   # resolution-safe (pandas 2/3)
    year = kst.year.to_numpy()
    split = np.select([year <= 2020, year <= 2022, year == 2023, year == 2024], ["buffer", "train", "validation", "test"], "buffer")
    tr = split == "train"
    z = np.log1p(u["pm25"][:, idx].astype(np.float64))
    mu, sd = np.nanmean(z[tr], 0), np.nanstd(z[tr], 0)
    mu, sd = np.where(np.isfinite(mu), mu, 0.0), np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    zs = (z - mu) / sd
    x = np.stack([np.nan_to_num(zs), np.isfinite(zs)], -1).astype(np.float32)
    g = u["globals"].astype(np.float64)
    gm, gs = np.nanmean(g[tr], 0), np.nanstd(g[tr], 0)
    g = np.nan_to_num((g - gm) / np.where(gs > 0, gs, 1.0))
    hour = kst.hour.to_numpy()
    ph = ((kst - pd.Timestamp("2000-01-01", tz="Asia/Seoul")) / pd.Timedelta(days=1)).to_numpy() / 365.2425 * 2 * np.pi
    cal = np.column_stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), np.sin(ph), np.cos(ph), np.sin(2 * ph), np.cos(2 * ph)])
    glob = np.concatenate([g, cal], 1).astype(np.float32)
    T = len(kst)
    y = np.full((T, len(H)), np.nan, np.float32)
    for j, h in enumerate(H):     # team eligibility: same split for origin and target, inside 2021-2024
        ok = (split[:-h] == split[h:]) & (split[:-h] != "buffer")
        y[:-h, j] = np.where(ok, zs[h:, 0], np.nan)
    adj = cdata.distance_graph(u["lon"][idx], u["lat"][idx])
    to = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return dict(x=to(x), glob=to(glob), y=to(y), split=split, kst=kst, mu0=mu[0], sd0=sd[0], adj=adj,
                node_keys=node_keys, scale_mu=mu, scale_sd=sd)


def origins(data, which, root):
    if which == "train":
        i = np.flatnonzero(data["split"] == "train")
        i = i[i >= WINDOW - 1]
        return i[torch.isfinite(data["y"][torch.as_tensor(i, device=data["y"].device)]).any(1).cpu().numpy()]
    grid = pd.read_parquet(root / "Dataset" / "Evaluation_Contract_20261107" / "evaluation_grid.parquet", columns=["origin_time_kst", "split"])
    t = pd.DatetimeIndex(grid.origin_time_kst[grid.split.eq(which)].unique())
    p = pd.Index(data["kst"]).get_indexer(t)
    assert (p >= WINDOW - 1).all()
    return p


def batch(data, idx):
    dev = data["x"].device
    i = torch.as_tensor(idx, device=dev)
    w = i[:, None] - torch.arange(WINDOW - 1, -1, -1, device=dev)[None, :]
    return data["x"][w], data["glob"][i], data["y"][i]


# ------------------------------------------------------------------ models
def head(n_in):
    return nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, len(H)))


class GRUNet(nn.Module):
    def __init__(self, n, g, hid=64):
        super().__init__()
        self.inp, self.gru, self.out = nn.Linear(2 * n, hid), nn.GRU(hid, hid, batch_first=True), head(hid + g)

    def forward(self, x, glob):
        B, T, N, C = x.shape
        _, hn = self.gru(torch.relu(self.inp(x.reshape(B, T, N * C))))
        return self.out(torch.cat([hn[-1], glob], 1))


class GCNGRU(nn.Module):
    def __init__(self, adj, g, hid=32, gc=16):
        super().__init__()
        a = torch.as_tensor(adj, dtype=torch.float32) + torch.eye(len(adj))
        dinv = a.sum(1).rsqrt()
        self.register_buffer("A", dinv[:, None] * a * dinv[None, :])
        self.w1, self.w2 = nn.Linear(2, gc), nn.Linear(gc, gc)
        self.gru, self.out = nn.GRU(gc, hid, batch_first=True), head(2 * hid + g)

    def forward(self, x, glob):
        B, T, N, _ = x.shape
        h = torch.relu(torch.einsum("nm,btmc->btnc", self.A, self.w1(x)))
        h = torch.relu(torch.einsum("nm,btmc->btnc", self.A, self.w2(h)))
        _, hn = self.gru(h.permute(0, 2, 1, 3).reshape(B * N, T, -1))
        hn = hn[-1].reshape(B, N, -1)
        return self.out(torch.cat([hn[:, 0], hn.mean(1), glob], 1))


class Residual(nn.Module):
    """Predict the change from the last observed Seoul value in the window (persistence + learned correction).
    Applied identically to every C model, so it never favours one condition or graph."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, glob):
        v, m = x[:, :, 0, 0], x[:, :, 0, 1]
        last = (m * torch.arange(1, x.shape[1] + 1, device=x.device)).argmax(1)
        base = v.gather(1, last[:, None]) * m.gather(1, last[:, None])   # 0 (= train mean) if never observed
        return self.net(x, glob) + base


def permuted(adj, seed):
    p = np.r_[0, 1 + np.random.default_rng(seed).permutation(len(adj) - 1)]
    return adj[np.ix_(p, p)]


def build(model, data, seed):
    n, g = data["x"].shape[1], data["glob"].shape[1]
    if model == "C_GRU":
        return Residual(GRUNet(n, g))
    if model == "C_GCNGRU":
        return Residual(GCNGRU(data["adj"], g))
    if model == "C_GCNGRU_PERM":
        return Residual(GCNGRU(permuted(data["adj"], 1000 + seed), g))
    raise ValueError(model)


def masked_mse(pred, y):
    m = torch.isfinite(y)
    return ((pred - torch.nan_to_num(y)) ** 2 * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def predict(model, data, idx, bs=128):
    model.eval()
    return torch.cat([model(*batch(data, idx[i:i + bs])[:2]) for i in range(0, len(idx), bs)]).float().cpu().numpy()


def to_frame(data, idx, z, model, condition, seed):
    yhat = np.expm1(z * data["sd0"] + data["mu0"])
    t = data["kst"][idx]
    return pd.concat([pd.DataFrame({"origin_time_kst": t, "horizon": int(h), "model": model, "y_hat": yhat[:, j],
                                    "split": data["split"][idx], "seed": int(seed), "prediction_kind": "seed",
                                    "input_condition": condition, "data_version": cdata.DATA_VERSION})
                      for j, h in enumerate(H)], ignore_index=True)


def run(model_name, condition, seed, root, epochs=40, patience=6, bs=64, lr=1e-3, max_batches=0, device=None, out=OUT):
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    data = load_condition(root, condition, device)
    model = build(model_name, data, seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    tr, va, te = origins(data, "train", root), origins(data, "validation", root), origins(data, "test", root)
    yva = data["y"][torch.as_tensor(va, device=device)]
    best, state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        model.train()
        perm, losses, te0 = rng.permutation(tr), [], time.time()
        for b, i in enumerate(range(0, len(perm), bs)):
            if max_batches and b >= max_batches:
                break
            xb, gb, yb = batch(data, perm[i:i + bs])
            loss = masked_mse(model(xb, gb), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss))
        val = float(masked_mse(torch.from_numpy(predict(model, data, va)).to(device), yva))
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_loss": val, "seconds": round(time.time() - te0, 2)})
        print(f"{model_name} {condition} seed{seed} ep{ep} train={hist[-1]['train_loss']:.4f} val={val:.4f} {hist[-1]['seconds']}s", flush=True)
        if val < best - 1e-4:
            best, bad, state = val, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    d = out / model_name / condition / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    idx = np.r_[va, te]
    to_frame(data, idx, predict(model, data, idx), model_name, condition, seed).to_parquet(d / "predictions.parquet", index=False)
    torch.save({"state_dict": state, "node_keys": data["node_keys"], "model": model_name, "condition": condition, "seed": seed}, d / "model.pt")
    meta = {"model": model_name, "condition": condition, "seed": seed, "nodes": len(data["node_keys"]),
            "parameters": int(sum(p.numel() for p in model.parameters())), "best_val_loss": best,
            "best_epoch": int(np.argmin([h["val_loss"] for h in hist])), "epochs_run": len(hist), "history": hist,
            "hyper": {"window": WINDOW, "epochs": epochs, "patience": patience, "batch": bs, "lr": lr, "weight_decay": 1e-4,
                      "dist_sigma_km": cdata.DIST_SIGMA_KM, "dist_cutoff_km": cdata.DIST_CUTOFF_KM, "max_batches": max_batches},
            "train_origins": len(tr), "validation_origins": len(va), "test_origins": len(te),
            "device": device, "seconds": round(time.time() - t0, 1), "data_version": cdata.DATA_VERSION}
    (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    m = run(a.model, a.condition, a.seed, cdata.data_root(a.data), a.epochs, max_batches=a.max_batches, out=Path(a.out))
    print(json.dumps({k: v for k, v in m.items() if k != "history"}, indent=1))
