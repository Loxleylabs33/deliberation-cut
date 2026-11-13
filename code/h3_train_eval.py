#!/usr/bin/env python3
"""Hypernetwork H3 (version 1): a small network reads the question vector and sets the dial for that question.

Settings: 0 = D (Thinking minus the deliberation piece), 1 = D + 0.75 P, 2 = D + P (Thinking).
Training labels (hyper/pool/dial0|dial75|dial100.jsonl, 440 questions disjoint from all tests): for each question,
pick the setting maximising  correct - LAMBDA * tokens / 1000  (cheapest correct; if none correct, the cheapest).
Model: standardise -> PCA to 64 dims -> multinomial logistic regression (L2), C chosen by 5-fold cross-validation on
the pool. Implemented in torch, no extra packages.
Test (no retraining): each test question is routed to the chosen setting; its expected correctness and length come
from the existing per-item test results (seeds averaged where several exist). Compared with always-0, always-0.75,
always-1, a task-identity router (perfect knowledge of the puzzle type, best setting per type on the POOL), and the
oracle. Writes hyper/H3_RESULTS.md.
"""
import collections
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cutconfig import ROOT  # noqa: E402

SETTINGS = ["dial0", "dial75", "dial100"]
LAMBDA = 0.01  # 1 point of accuracy is worth 1,000 tokens / 10 -> tokens matter only as a tie-breaker


def read_pool():
    res = {}
    for k, s in enumerate(SETTINGS):
        for r in map(json.loads, open(f"{ROOT}/hyper/pool/{s}.jsonl")):
            res.setdefault(r["id"], {})[k] = r
    return {i: v for i, v in res.items() if len(v) == len(SETTINGS)}


def label_of(per_setting):
    score = [per_setting[k]["ok"] - LAMBDA * (per_setting[k]["tokens"] or 0) / 1000 for k in range(len(SETTINGS))]
    return max(range(len(SETTINGS)), key=lambda k: score[k])


def test_results():
    """Per test item, per setting: (expected correctness, mean tokens), averaging every seed found."""
    dirs = {0: ["q80-neg", "q80-neg-seed1", "q80-neg-seed2"],
            1: ["dial75-full", "dial75-full-seed1", "dial75-full-seed2"],
            2: ["thinking", "thinking-seed1", "thinking-seed2"]}
    out = collections.defaultdict(dict)
    for k, labs in dirs.items():
        acc = collections.defaultdict(list)
        for lab in labs:
            d = f"{ROOT}/step4/{lab}"
            rerun = {os.path.basename(f)[6:-6] for f in glob.glob(f"{d}/items_*.jsonl")}  # tasks redone under new rules
            for f in glob.glob(f"{d}/items_*.jsonl") + glob.glob(f"{d}/items.jsonl"):
                old = f.endswith("/items.jsonl")
                for r in map(json.loads, open(f)):
                    t = r.get("task", "")
                    if t not in ("gsm8k", "math", "bbh"):
                        continue
                    if old and t in rerun:  # the first harness's MATH/BBH used a 4,096 cap: never mix with reruns
                        continue
                    acc[r["id"]].append((r["ok"], r.get("tokens") or 0))
        for i, v in acc.items():
            out[i][k] = (sum(a for a, _ in v) / len(v), sum(t for _, t in v) / len(v))
    return out


def fit(X, y, C, epochs=400):
    W = torch.zeros(X.shape[1], len(SETTINGS), requires_grad=True)
    b = torch.zeros(len(SETTINGS), requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=epochs, line_search_fn="strong_wolfe")
    cw = torch.bincount(y, minlength=len(SETTINGS)).float()
    cw = (len(y) / (len(SETTINGS) * cw.clamp(min=1)))  # balance rare "think" labels

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W + b, y, weight=cw) + (W ** 2).sum() / (2 * C * len(y))
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach(), b.detach()


def main():
    emb = torch.load(f"{ROOT}/hyper/embed.pt", weights_only=True)
    idx = {i: n for n, i in enumerate(emb["ids"])}
    task_of = dict(zip(emb["ids"], emb["tasks"]))
    pool = read_pool()
    pids = sorted(pool)
    y = torch.tensor([label_of(pool[i]) for i in pids])
    Xp = emb["x"][[idx[i] for i in pids]]
    mu, sd = Xp.mean(0), Xp.std(0).clamp(min=1e-6)
    Z = (Xp - mu) / sd
    U, S, V = torch.pca_lowrank(Z, q=64, center=False)
    proj = V[:, :64]
    Zp = Z @ proj
    # 5-fold CV for C
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(pids), generator=g)
    folds = [perm[k::5] for k in range(5)]
    best_c, best_acc = None, -1
    for C in (0.01, 0.1, 1.0, 10.0):
        hits = 0
        for f in folds:
            tr = torch.tensor([j for j in range(len(pids)) if j not in set(f.tolist())])
            W, b = fit(Zp[tr], y[tr], C)
            hits += int(((Zp[f] @ W + b).argmax(1) == y[f]).sum())
        acc = hits / len(pids)
        print(f"C {C}: CV label accuracy {acc:.3f}", flush=True)
        if acc > best_acc:
            best_c, best_acc = C, acc
    W, b = fit(Zp, y, best_c)
    torch.save({"mu": mu, "sd": sd, "proj": proj, "W": W, "b": b, "C": best_c, "settings": SETTINGS,
                "layer": emb["layer"]}, f"{ROOT}/hyper/router_v1.pt")

    # per-task best setting on the POOL (the task-identity router)
    by_task = collections.defaultdict(lambda: [[0.0, 0.0] for _ in SETTINGS])
    for i in pids:
        t = task_of[i] if task_of[i] != "bbh" else "bbh-" + i.split("pool-bbh-")[1].rsplit("-", 1)[0]
        for k in range(len(SETTINGS)):
            by_task[t][k][0] += pool[i][k]["ok"] - LAMBDA * (pool[i][k]["tokens"] or 0) / 1000
            by_task[t][k][1] += 1
    task_choice = {t: max(range(len(SETTINGS)), key=lambda k: v[k][0]) for t, v in by_task.items()}

    tr = test_results()
    test_ids = [i for i in emb["ids"] if not i.startswith("pool-") and len(tr.get(i, {})) == len(SETTINGS)]
    Xt = ((emb["x"][[idx[i] for i in test_ids]] - mu) / sd) @ proj
    pred = (Xt @ W + b).argmax(1).tolist()

    def tkey(i):
        return task_of[i] if task_of[i] != "bbh" else "bbh-" + i[4:].rsplit("-", 1)[0]

    policies = {"always D (0)": [0] * len(test_ids), "always 0.75": [1] * len(test_ids),
                "always Thinking (1)": [2] * len(test_ids), "router v1": pred,
                "task-identity router": [task_choice.get(tkey(i), 0) for i in test_ids],
                "oracle": [max(range(len(SETTINGS)), key=lambda k: tr[i][k][0] - LAMBDA * tr[i][k][1] / 1000)
                           for i in test_ids]}
    lines = [f"# H3 router v1 ({os.path.basename(ROOT)}; pool {len(pids)} questions, labels "
             f"{dict(collections.Counter(y.tolist()))}, C {best_c}, CV label accuracy {best_acc:.3f})\n",
             "| policy | " + " | ".join(f"{t} acc" for t in ("gsm8k", "math", "bbh")) + " | mean tokens | thinks on |",
             "|---|---|---|---|---|---|"]
    for name, choice in policies.items():
        acc = collections.defaultdict(list)
        toks, thinks = [], 0
        for i, k in zip(test_ids, choice):
            acc[task_of[i]].append(tr[i][k][0])
            toks.append(tr[i][k][1])
            thinks += k > 0
        lines.append(f"| {name} | " + " | ".join(f"{100 * sum(acc[t]) / len(acc[t]):.1f}" for t in ("gsm8k", "math", "bbh"))
                     + f" | {sum(toks) / len(toks):.0f} | {100 * thinks / len(choice):.0f}% |")
    # where the router chose to think, by BBH task
    think_by = collections.defaultdict(list)
    for i, k in zip(test_ids, pred):
        think_by[tkey(i)].append(k > 0)
    lines.append("\nRouter v1 'think' rate by question type: " + ", ".join(
        f"{t} {100 * sum(v) / len(v):.0f}%" for t, v in sorted(think_by.items())))
    lines.append(f"\nTask-identity router choices (from the pool): {task_choice}")
    open(f"{ROOT}/hyper/H3_RESULTS.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
