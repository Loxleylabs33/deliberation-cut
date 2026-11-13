#!/usr/bin/env python3
"""H4: train the per-layer dial hypernetwork.

Teacher: router v2's choice for each pool question, CROSS-FITTED (5 folds, so no question is labelled by a router that
saw it), lambda 0.01. Target text: the response from the chosen setting if it is correct and not capped; otherwise
the shortest correct response from any setting; questions with none are skipped. Targets over 4,096 tokens are
skipped (memory).
Hypernetwork: question vector (Instruct layer 18, pooled) -> standardise -> PCA-64 -> MLP 64-128-37 -> sigmoid = beta
(36 layers + embedding/head). Starts near beta 0.12 everywhere (close to D).
Loss: next-token loss of the target under the composed model with beta(x), + MU * mean(beta) (deliberation costs).
Only the hypernetwork trains; both 4B models stay frozen. Writes hyper/h4/hypernet.pt, betas.pt (beta for every pool
and test question), train_log.jsonl.
Usage: h4_train.py [--epochs 3] [--mu 0.02] [--limit N]
"""
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h3_train_eval as v1  # noqa: E402
import h3b_value_router as v2  # noqa: E402
from h4_compose import Composed, T_DIR  # noqa: E402

ROOT = v1.ROOT
# H4_OUT and H4_POOL_TEXT redirect output and teacher data (the 32k-budget retrain, "H5", sets both); unset = H4.
OUT = os.environ.get("H4_OUT", f"{ROOT}/hyper/h4")
POOL_TEXT = os.environ.get("H4_POOL_TEXT", f"{ROOT}/hyper/pool_text")
SETS = ["dial0", "dial75", "dial100"]


def arg(n, d):
    return type(d)(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d


def teacher_choices(emb, pids, pool):
    idx = {i: n for n, i in enumerate(emb["ids"])}
    X = emb["x"][[idx[i] for i in pids]]
    choice = {}
    perm = torch.randperm(len(pids), generator=torch.Generator().manual_seed(1))
    for f in range(5):
        test = perm[f::5]
        tset = set(test.tolist())
        tr = torch.tensor([j for j in range(len(pids)) if j not in tset])
        mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-6)
        Z = (X - mu) / sd
        _, _, V = torch.pca_lowrank(Z[tr], q=64, center=False)
        Zp = Z @ V[:, :64]
        tok = torch.tensor([sum((pool[pids[j]][k]["tokens"] or 0) for j in tr.tolist()) / len(tr) for k in range(3)])
        P = []
        for k in range(3):
            y = torch.tensor([float(pool[pids[j]][k]["ok"]) for j in tr.tolist()])
            w, b = v2.fit_logistic(Zp[tr], y, 0.01)
            P.append(torch.sigmoid(Zp[test] @ w + b))
        util = torch.stack(P, 1) - 0.01 * tok / 1000
        for j, k in zip(test.tolist(), util.argmax(1).tolist()):
            choice[pids[j]] = k
    return choice


class HyperNet(torch.nn.Module):
    def __init__(self, mu, sd, proj, n_out=37):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.register_buffer("proj", proj)
        self.net = torch.nn.Sequential(torch.nn.Linear(proj.shape[1], 128), torch.nn.GELU(), torch.nn.Linear(128, n_out))
        torch.nn.init.zeros_(self.net[2].weight)
        torch.nn.init.constant_(self.net[2].bias, -2.0)  # beta ~ 0.12 at the start

    def forward(self, x):
        return torch.sigmoid(self.net(((x - self.mu) / self.sd) @ self.proj))


def main():
    epochs, mu_cost, limit = arg("--epochs", 3), arg("--mu", 0.02), arg("--limit", 0)
    maxlen = arg("--maxlen", 4096)  # H5: 24,576 so long Thinking answers (maths at 32k) can be targets
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda:0")
    emb = torch.load(f"{ROOT}/hyper/embed.pt", weights_only=True)
    idx = {i: n for n, i in enumerate(emb["ids"])}
    texts = {}
    for k, s in enumerate(SETS):
        for r in map(json.loads, open(f"{POOL_TEXT}/{s}.jsonl")):
            texts.setdefault(r["id"], {})[k] = r
    pool = {i: v for i, v in texts.items() if len(v) == 3}
    pids = sorted(pool)
    choice = teacher_choices(emb, pids, pool)
    tok = AutoTokenizer.from_pretrained(T_DIR)
    data = []
    for i in pids:
        c = pool[i][choice[i]]
        if not (c["ok"] and not c["truncated"]):
            good = [pool[i][k] for k in range(3) if pool[i][k]["ok"] and not pool[i][k]["truncated"]]
            if not good:
                continue
            c = min(good, key=lambda r: r["tokens"] or 0)
        p_ids = tok(c["prompt"], add_special_tokens=False)["input_ids"]
        t_ids = tok(c["text"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(t_ids) > maxlen:
            continue
        data.append(dict(id=i, x=emb["x"][idx[i]], ids=torch.tensor(p_ids + t_ids), start=len(p_ids),
                         teacher=choice[i], src=SETS.index(next(s for s in SETS if pool[i][SETS.index(s)] is c))))
    random.Random(3).shuffle(data)
    if limit:
        data = data[:limit]
    n_val = max(4, len(data) // 10)
    val, train = data[:n_val], data[n_val:]
    print(f"targets: {len(data)} (train {len(train)}, val {len(val)}); teacher choices "
          f"{ {s: sum(1 for d in data if d['teacher'] == k) for k, s in enumerate(SETS)} }; target sources "
          f"{ {s: sum(1 for d in data if d['src'] == k) for k, s in enumerate(SETS)} }", flush=True)

    Xp = torch.stack([d["x"] for d in train])
    mu, sd = Xp.mean(0), Xp.std(0).clamp(min=1e-6)
    _, _, V = torch.pca_lowrank((Xp - mu) / sd, q=64, center=False)
    hyper = HyperNet(mu, sd, V[:, :64]).to(dev)
    opt = torch.optim.AdamW(hyper.parameters(), lr=1e-3, weight_decay=1e-2)
    comp = Composed(dev)
    print(comp.summary, flush=True)
    lm_head = comp.model.lm_head

    def chunk_loss(h, y):
        return F.cross_entropy(lm_head(h).float(), y, reduction="sum")

    def sample_loss(d):
        beta = hyper(d["x"].to(dev).unsqueeze(0))[0]
        comp.set_beta(beta)
        ids = d["ids"].to(dev).unsqueeze(0)
        h = comp.model.model(input_ids=ids).last_hidden_state[0]
        y, hs = ids[0, d["start"]:], h[d["start"] - 1:-1]
        tot = 0.0
        for s in range(0, y.numel(), 512):
            tot = tot + checkpoint(chunk_loss, hs[s:s + 512], y[s:s + 512], use_reentrant=False)
        return tot / y.numel(), beta

    log = open(f"{OUT}/train_log.jsonl", "a")
    t0, step = time.time(), 0
    for ep in range(epochs):
        random.Random(ep).shuffle(train)
        opt.zero_grad()
        run_ce, run_b = 0.0, 0.0
        skipped = 0
        for n, d in enumerate(train, 1):
            try:
                ce, beta = sample_loss(d)
                loss = (ce + mu_cost * beta.mean()) / 8
                loss.backward()
                run_ce += float(ce)
                run_b += float(beta.mean())
            except torch.cuda.OutOfMemoryError:  # H5: one target too long for memory; drop this accumulation group
                skipped += 1
                ce = beta = loss = None
                opt.zero_grad()
                torch.cuda.empty_cache()
                print(f"ep {ep} {n}: out of memory on {d['id']} ({d['ids'].numel()} tokens), skipped", flush=True)
            if n % 8 == 0 or n == len(train):
                opt.step()
                opt.zero_grad()
                step += 1
            if n % 40 == 0 or n == len(train):
                print(f"ep {ep} {n}/{len(train)} ce {run_ce / n:.4f} mean beta {run_b / n:.3f} "
                      f"({time.time() - t0:.0f} s, peak {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB)", flush=True)
        with torch.no_grad():
            vals = []
            for d in val:
                try:
                    vals.append(float(sample_loss(d)[0]))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
            vce = sum(vals) / max(1, len(vals))
        rec = dict(epoch=ep, train_ce=run_ce / len(train), val_ce=vce, mean_beta=run_b / len(train),
                   seconds=round(time.time() - t0), skipped_oom=skipped)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print("EPOCH", json.dumps(rec), flush=True)
        torch.save(hyper.state_dict(), f"{OUT}/hypernet.pt")
    with torch.no_grad():
        B = hyper(emb["x"].to(dev)).cpu()
    torch.save({"ids": emb["ids"], "tasks": emb["tasks"], "beta": B}, f"{OUT}/betas.pt")
    by = {}
    for i, t, b in zip(emb["ids"], emb["tasks"], B):
        key = ("pool " if i.startswith("pool-") else "test ") + (t if t != "bbh" else "bbh-" + i.split("bbh-")[1].rsplit("-", 1)[0])
        by.setdefault(key, []).append(b)
    for key in sorted(by):
        m = torch.stack(by[key]).mean(0)
        print(f"{key:48s} mean beta {float(m[:36].mean()):.2f} | early {float(m[:12].mean()):.2f} mid "
              f"{float(m[12:24].mean()):.2f} late {float(m[24:36].mean()):.2f} | emb {float(m[36]):.2f}", flush=True)


if __name__ == "__main__":
    main()
