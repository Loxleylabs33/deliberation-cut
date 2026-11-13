#!/usr/bin/env python3
"""Score a saved router on the first test set (GSM8K 300, MATH 50, BBH logic 90) from per-item results. No retraining.

    export CUT_ROOT=WORKDIR            # made by code/setup_workdir.py
    python3 -I score_router.py router_pool440_h3b/router_v2_pool440.pt

Prints the table in results/router_h3b.md: always-D, always-Thinking, and the router at several values of lambda (the
price of 1,000 tokens in probability points; the router picks the setting with the highest p_correct - lambda * mean
tokens / 1000). Each test question is scored by looking up what the model at the chosen setting actually did on that
question in the per-item results (BBH: averaged over its three runs); the router itself does not run in the loop.
The settings are 0 = D (the released model), 1 = D + 0.75 P, 2 = Thinking.
"""
import collections
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import h3_train_eval as v1  # noqa: E402
from cutconfig import ROOT  # noqa: E402


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    r = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
    emb = torch.load(f"{ROOT}/hyper/embed.pt", weights_only=True)
    S = r["settings"]
    idx = {i: n for n, i in enumerate(emb["ids"])}
    task_of = dict(zip(emb["ids"], emb["tasks"]))
    mean_tok = r["mean_tokens"]
    tr_res = v1.test_results()
    test_ids = [i for i in emb["ids"] if not i.startswith("pool-") and len(tr_res.get(i, {})) == len(S)]
    Xt = ((emb["x"][[idx[i] for i in test_ids]] - r["mu"]) / r["sd"]) @ r["proj"]
    P = torch.stack([torch.sigmoid(Xt @ r["models"][k][0] + r["models"][k][1]) for k in range(len(S))], dim=1)

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

    lines = [f"# router {os.path.basename(sys.argv[1])}: test questions {len(test_ids)}; C {r['C']}; pool mean tokens "
             f"by setting { {S[k]: round(v) for k, v in mean_tok.items()} }\n",
             "| policy | gsm8k | math | bbh | mean tokens | thinks on |", "|---|---|---|---|---|---|"]
    for name, ch in {"always D (0)": [0] * len(test_ids), "always Thinking (1)": [2] * len(test_ids)}.items():
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
    by = collections.defaultdict(list)
    for i, k in zip(test_ids, chosen):
        by[tkey(i)].append(k)
    lines.append("\nRouter v2 (lambda 0.01) setting chosen by question type (share choosing 0 / 0.75 / 1): " + "; ".join(
        f"{t} " + "/".join(f"{100 * sum(1 for x in v if x == k) / len(v):.0f}" for k in range(len(S)))
        for t, v in sorted(by.items())))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
