#!/usr/bin/env python3
"""Hypernetwork H3, version 2: a value router.

v1 classified the cheapest-correct setting per question: 420 of 440 labels said 'no thinking', because thinking's gain
is a RATE over a question type (web of lies: about 1 question in 9 flips), not a property single noisy labels show.
v2 predicts, for every setting k, the probability that setting answers the question correctly (one logistic model per
setting, trained on every setting's actual right/wrong on the pool), and the expected length (mean per setting on the
pool), then picks  argmax_k  p_k - LAMBDA * tokens_k / 1000.  Same inputs, test scoring and baselines as v1.
Writes hyper/H3b_RESULTS.md and hyper/router_v2.pt; sweeps LAMBDA to show the accuracy/length trade.
"""
import collections
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h3_train_eval as v1  # noqa: E402

ROOT = v1.ROOT
S = v1.SETTINGS


def fit_logistic(X, y, C, iters=300):
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + (w ** 2).sum() / (2 * C * len(y))
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


def main():
    emb = torch.load(f"{ROOT}/hyper/embed.pt", weights_only=True)
    idx = {i: n for n, i in enumerate(emb["ids"])}
    task_of = dict(zip(emb["ids"], emb["tasks"]))
    pool = v1.read_pool()
    pids = sorted(pool)
    Xp = emb["x"][[idx[i] for i in pids]]
    mu, sd = Xp.mean(0), Xp.std(0).clamp(min=1e-6)
    Z = (Xp - mu) / sd
    _, _, V = torch.pca_lowrank(Z, q=64, center=False)
    proj = V[:, :64]
    Zp = Z @ proj
    Y = {k: torch.tensor([float(pool[i][k]["ok"]) for i in pids]) for k in range(len(S))}
    mean_tok = {k: sum((pool[i][k]["tokens"] or 0) for i in pids) / len(pids) for k in range(len(S))}
    # C by 5-fold CV on log-loss, shared across settings
    perm = torch.randperm(len(pids), generator=torch.Generator().manual_seed(0))
    folds = [perm[f::5] for f in range(5)]
    best = (None, 1e9)
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        ll = 0.0
        for f in folds:
            fs = set(f.tolist())
            tr = torch.tensor([j for j in range(len(pids)) if j not in fs])
            for k in range(len(S)):
                w, b = fit_logistic(Zp[tr], Y[k][tr], C)
                p = torch.sigmoid(Zp[f] @ w + b).clamp(1e-4, 1 - 1e-4)
                ll += float(-(Y[k][f] * p.log() + (1 - Y[k][f]) * (1 - p).log()).sum())
        print(f"C {C}: CV log-loss {ll / (len(pids) * len(S)):.4f}", flush=True)
        if ll < best[1]:
            best = (C, ll)
    C = best[0]
    models = {k: fit_logistic(Zp, Y[k], C) for k in range(len(S))}
    torch.save({"mu": mu, "sd": sd, "proj": proj, "models": models, "mean_tokens": mean_tok, "C": C, "settings": S,
                "layer": emb["layer"]}, f"{ROOT}/hyper/router_v2.pt")

    tr_res = v1.test_results()
    test_ids = [i for i in emb["ids"] if not i.startswith("pool-") and len(tr_res.get(i, {})) == len(S)]
    Xt = ((emb["x"][[idx[i] for i in test_ids]] - mu) / sd) @ proj
    P = torch.stack([torch.sigmoid(Xt @ models[k][0] + models[k][1]) for k in range(len(S))], dim=1)

    def score(choice):
        acc = collections.defaultdict(list)
        toks, thinks = [], 0
        for i, k in zip(test_ids, choice):
            acc[task_of[i]].append(tr_res[i][k][0])
            toks.append(tr_res[i][k][1])
            thinks += k > 0
        return ([100 * sum(acc[t]) / len(acc[t]) for t in ("gsm8k", "math", "bbh")], sum(toks) / len(toks),
                100 * thinks / len(choice))

    def tkey(i):
        return task_of[i] if task_of[i] != "bbh" else "bbh-" + i[4:].rsplit("-", 1)[0]

    lines = [f"# H3 router v2: value router (pool {len(pids)}; C {C}; pool mean tokens by setting "
             f"{ {S[k]: round(v) for k, v in mean_tok.items()} })\n",
             "| policy | gsm8k | math | bbh | mean tokens | thinks on |", "|---|---|---|---|---|---|"]
    base = {"always D (0)": [0] * len(test_ids), "always Thinking (1)": [2] * len(test_ids)}
    for name, ch in base.items():
        a, t, th = score(ch)
        lines.append(f"| {name} | {a[0]:.1f} | {a[1]:.1f} | {a[2]:.1f} | {t:.0f} | {th:.0f}% |")
    chosen = None
    for lam in (0.0, 0.005, 0.01, 0.02, 0.05):
        util = P - lam * torch.tensor([mean_tok[k] for k in range(len(S))]) / 1000
        ch = util.argmax(1).tolist()
        a, t, th = score(ch)
        lines.append(f"| router v2, lambda {lam} | {a[0]:.1f} | {a[1]:.1f} | {a[2]:.1f} | {t:.0f} | {th:.0f}% |")
        if lam == 0.01:
            chosen = ch
    think_by = collections.defaultdict(list)
    for i, k in zip(test_ids, chosen):
        think_by[tkey(i)].append(k)
    lines.append("\nRouter v2 (lambda 0.01) setting chosen by question type (share choosing 0 / 0.75 / 1): " + "; ".join(
        f"{t} " + "/".join(f"{100 * sum(1 for x in v if x == k) / len(v):.0f}" for k in range(len(S)))
        for t, v in sorted(think_by.items())))
    open(f"{ROOT}/hyper/H3b_RESULTS.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
