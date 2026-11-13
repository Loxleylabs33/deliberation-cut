#!/usr/bin/env python3
"""Recompute the original-model, D and router rows of the long-budget table (reply cap 32,768 tokens) from the
per-item results in this release. The rows for the per-layer dial are not reproduced here: the dial's weights are not
part of this release.

    export CUT_ROOT=WORKDIR            # made by code/setup_workdir.py
    python3 -I score_long_budget.py [OLD_ROUTER.pt NEW_ROUTER.pt]

The routers default to the two shipped files: the one trained on the 440-question pool at the original token caps
("old pool") and the one trained on the same pool re-answered at the 32,768 cap ("32k pool"). Question vectors come
from hyper/c32/x.pt (Instruct, layer 18, as in router/README.md). Policies:
  accuracy first = argmax_k p_k (lambda 0);  balanced = argmax_k p_k - 0.01 * mean_tokens_k / 1000.
A routed question is scored with the answer(s) the model at the chosen setting gave to that question in the
long-budget runs (every setting was run on every question it could be routed to); the router does not run in the loop.
Each question's correctness is averaged over the runs present (seed 0 = c32-<model>, seed 1 = c32-<model>-s1), then
averaged over questions. Pairs: questions where A's average beats B's and the reverse, exact two-sided sign test.
"""
import collections
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
from cutconfig import ROOT  # noqa: E402

TASKS = ("mathhard", "bbhhard", "math", "gsm8k")
NAMES = {"mathhard": "hard maths (96)", "bbhhard": "hard logic (200)", "math": "MATH test (50)", "gsm8k": "GSM8K (300)"}
SEEDS = ("", "-s1")
SETTING = {0: "dial0", 1: "dial75", 2: "thinking"}
task_of = lambda i: i.split("-", 1)[0]  # noqa: E731
_cache = {}


def load(label, task):
    if (label, task) not in _cache:
        f = f"{ROOT}/step4/{label}/items_{task}.jsonl"
        rows = list(map(json.loads, open(f))) if os.path.exists(f) else []
        _cache[(label, task)] = {r["id"]: (float(r["ok"]), r.get("tokens") or 0, bool(r.get("truncated")))
                                 for r in rows}
    return _cache[(label, task)]


def answers(model, task):
    """Per seed, {id: (ok, tokens, truncated)} merged over every label that holds this model's answers."""
    labels = {"dial75": ["c32-dial75", "c32-dial75x"]}.get(model, [f"c32-{model}"])
    out = []
    for s in SEEDS:
        d = {}
        for lab in labels:
            d.update(load(f"{lab}{s}", task))
        if d:
            out.append(d)
    return out


def sign_p(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)


def choices(router_path, vec):
    """{id: (setting at lambda 0, setting at lambda 0.01)} for a saved router."""
    r = torch.load(router_path, map_location="cpu", weights_only=True)
    Z = ((vec["x"] - r["mu"]) / r["sd"]) @ r["proj"]
    P = torch.stack([torch.sigmoid(Z @ w + b) for w, b in (r["models"][k] for k in range(3))], 1)
    tokc = torch.tensor([r["mean_tokens"][k] for k in range(3)])
    a, b = P.argmax(1).tolist(), (P - 0.01 * tokc / 1000).argmax(1).tolist()
    return {i: (a[n], b[n]) for n, i in enumerate(vec["ids"])}


def main():
    default = os.path.join(os.path.dirname(HERE), "router")
    old_path = sys.argv[1] if len(sys.argv) > 1 else f"{default}/router_pool440_h3b/router_v2_pool440.pt"
    new_path = sys.argv[2] if len(sys.argv) > 2 else f"{default}/router_pool32k_h5/router_v2_pool32k.pt"
    vec = torch.load(f"{ROOT}/hyper/c32/x.pt", weights_only=True)
    ro, rn = choices(old_path, vec), choices(new_path, vec)
    pick = {
        "Instruct (original)": lambda i: "instruct",
        "Thinking (original)": lambda i: "thinking",
        "D (the cut, never thinks)": lambda i: "dial0",
        "router v2, old pool, accuracy first": lambda i: SETTING[ro[i][0]],
        "router v2, 32k pool, accuracy first": lambda i: SETTING[rn[i][0]],
        "router v2, 32k pool, balanced": lambda i: SETTING[rn[i][1]],
    }
    ids = {t: sorted(i for i in vec["ids"] if task_of(i) == t) for t in TASKS}
    score, toks, nruns = {}, {}, {}
    for p, f in pick.items():
        score[p], toks[p], nruns[p] = {}, {}, {}
        for t in TASKS:
            for i in ids[t]:
                rs = [d[i] for d in answers(f(i), t) if i in d]
                if rs:
                    score[p][i] = sum(r[0] for r in rs) / len(rs)
                    toks[p][i] = sum(r[1] for r in rs) / len(rs)
                    nruns[p].setdefault(t, set()).add(len(rs))
    L = ["# Long-budget test: cap 32,768, KV 8-bit, 5 slots. Cell: accuracy % (runs per question) / mean tokens.", "",
         "| policy | " + " | ".join(NAMES[t] for t in TASKS) + " |", "|---|" + "---|" * len(TASKS)]
    for p in pick:
        cells = []
        for t in TASKS:
            v = [score[p][i] for i in ids[t] if i in score[p]]
            if len(v) < len(ids[t]):
                cells.append(f"pending ({len(v)}/{len(ids[t])})")
                continue
            tk = sum(toks[p][i] for i in ids[t]) / len(ids[t])
            cells.append(f"{100 * sum(v) / len(v):.1f} ({'/'.join(map(str, sorted(nruns[p][t])))}) / {tk:,.0f}")
        L.append(f"| {p} | " + " | ".join(cells) + " |")
    L += ["", "Question by question: A better / B better (averaged over runs), exact two-sided sign test p.", "",
          "| A vs B | " + " | ".join(NAMES[t] for t in TASKS) + " |", "|---|" + "---|" * len(TASKS)]
    for a, b in [("router v2, 32k pool, accuracy first", "router v2, old pool, accuracy first"),
                 ("router v2, 32k pool, accuracy first", "Thinking (original)"),
                 ("router v2, 32k pool, accuracy first", "D (the cut, never thinks)"),
                 ("Thinking (original)", "D (the cut, never thinks)"),
                 # added after the first draft of this page, because the paper cites it (hard maths)
                 ("Instruct (original)", "D (the cut, never thinks)")]:
        cells = []
        for t in TASKS:
            both = [i for i in ids[t] if i in score[a] and i in score[b]]
            if len(both) < len(ids[t]):
                cells.append("pending")
                continue
            wa = sum(score[a][i] > score[b][i] for i in both)
            wb = sum(score[b][i] > score[a][i] for i in both)
            cells.append(f"{wa} / {wb} (p {sign_p(wa, wb):.3f})")
        L.append(f"| {a} vs {b} | " + " | ".join(cells) + " |")
    L += ["", "Hard logic by puzzle type (accuracy %, averaged over runs):"]
    for p in pick:
        bt = collections.defaultdict(list)
        for i in ids["bbhhard"]:
            if i in score[p]:
                bt[i[len("bbhhard-"):].rsplit("-", 1)[0]].append(score[p][i])
        if bt:
            L.append(f"- {p}: " + ", ".join(f"{t} {100 * sum(v) / len(v):.0f} (n {len(v)})" for t, v in sorted(bt.items())))
    print("\n".join(L))


if __name__ == "__main__":
    main()
