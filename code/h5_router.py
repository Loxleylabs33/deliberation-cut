#!/usr/bin/env python3
"""H5 step 2: router v2 retrained on the 32k pool (hyper/pool_text32), same method as h3b_value_router.py: one
logistic model per setting predicts P(correct) from the question vector (Instruct layer 18, PCA-64), C by 5-fold CV
log-loss; choice = argmax p_k - lambda * tokens_k / 1000. Writes hyper/h5/router_v2.pt (same keys as the H3b file)
and prints what changed against the old pool."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h3b_value_router as v2  # noqa: E402
from cutconfig import ROOT  # noqa: E402

SETS = ["dial0", "dial75", "dial100"]


def read(d):
    res = {}
    for k, s in enumerate(SETS):
        for r in map(json.loads, open(f"{ROOT}/hyper/{d}/{s}.jsonl")):
            res.setdefault(r["id"], {})[k] = r
    return {i: v for i, v in res.items() if len(v) == 3}


def main():
    os.makedirs(f"{ROOT}/hyper/h5", exist_ok=True)
    emb = torch.load(f"{ROOT}/hyper/embed.pt", weights_only=True)
    idx = {i: n for n, i in enumerate(emb["ids"])}
    pool = read("pool_text32")
    # the earlier-caps pool is optional: it is only used for the comparison printed below
    old = read("pool_text") if os.path.isdir(f"{ROOT}/hyper/pool_text") else {}
    pids = sorted(pool)
    task = {i: pool[i][0]["task"] for i in pids}
    for t in ("gsm8k", "math", "bbh"):
        ids = [i for i in pids if task[i] == t]
        new = " / ".join(f"{100 * sum(pool[i][k]['ok'] for i in ids) / len(ids):.1f}" for k in range(3))
        if all(i in old for i in ids):
            before = " / ".join(f"{100 * sum(old[i][k]['ok'] for i in ids) / len(ids):.1f}" for k in range(3))
            print(f"pool {t} (n {len(ids)}) accuracy D / 0.75 / Thinking: old {before} -> 32k {new}", flush=True)
        else:
            print(f"pool {t} (n {len(ids)}) accuracy D / 0.75 / Thinking at the 32k budget: {new}", flush=True)
    Xp = emb["x"][[idx[i] for i in pids]]
    mu, sd = Xp.mean(0), Xp.std(0).clamp(min=1e-6)
    Z = (Xp - mu) / sd
    _, _, V = torch.pca_lowrank(Z, q=64, center=False)
    proj = V[:, :64]
    Zp = Z @ proj
    Y = {k: torch.tensor([float(pool[i][k]["ok"]) for i in pids]) for k in range(3)}
    mean_tok = {k: sum((pool[i][k]["tokens"] or 0) for i in pids) / len(pids) for k in range(3)}
    perm = torch.randperm(len(pids), generator=torch.Generator().manual_seed(0))
    folds = [perm[f::5] for f in range(5)]
    best = (None, 1e9)
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        ll = 0.0
        for f in folds:
            fs = set(f.tolist())
            tr = torch.tensor([j for j in range(len(pids)) if j not in fs])
            for k in range(3):
                w, b = v2.fit_logistic(Zp[tr], Y[k][tr], C)
                p = torch.sigmoid(Zp[f] @ w + b).clamp(1e-4, 1 - 1e-4)
                ll += float(-(Y[k][f] * p.log() + (1 - Y[k][f]) * (1 - p).log()).sum())
        print(f"C {C}: CV log-loss {ll / (len(pids) * 3):.4f}", flush=True)
        if ll < best[1]:
            best = (C, ll)
    C = best[0]
    models = {k: v2.fit_logistic(Zp, Y[k], C) for k in range(3)}
    torch.save({"mu": mu, "sd": sd, "proj": proj, "models": models, "mean_tokens": mean_tok, "C": C,
                "settings": SETS, "layer": emb["layer"], "pool": "pool_text32"}, f"{ROOT}/hyper/h5/router_v2.pt")
    print(f"router v2 (32k pool) saved: C {C}; pool mean tokens {[round(v) for v in mean_tok.values()]}", flush=True)


if __name__ == "__main__":
    main()
