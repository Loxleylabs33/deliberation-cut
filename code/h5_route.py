#!/usr/bin/env python3
"""H5 step 4: route the long-budget test questions (GSM8K 300, MATH test 50, hard maths 96, hard logic 200) through
the dial and router retrained on the 32k pool. As H4: the hypernetwork's 37 dials per question are grouped by k-means
(K=4, no labels) into profiles; a profile whose middle layers average above 0.5 thinks, and runs with its early
layers at full (the H4b rule). Question vectors (Instruct layer 18) are cached in hyper/c32/x.pt.
Writes hyper/h5/profiles.json, hyper/h5/routes.json, hyper/h5/ids_<model>.json, and hyper/h5/ids_dial75x.json (the
0.75 answers the new router needs that the long-budget test did not already produce)."""
import collections
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cutconfig import INSTRUCT_DIR, ROOT  # noqa: E402

H5 = f"{ROOT}/hyper/h5"
TASKS = ("gsm8k", "math", "mathhard", "bbhhard")
K = 4


def vectors():
    f = f"{ROOT}/hyper/c32/x.pt"
    if os.path.exists(f):
        return torch.load(f, weights_only=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import step4_eval as s4
    items = [it for t in TASKS for it in s4.items_for(t)]
    dev = torch.device(os.environ.get("EMBED_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))
    hf = INSTRUCT_DIR
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16).to(dev).eval()
    xs = []
    with torch.no_grad():
        for it in items:
            enc = tok(f"<|im_start|>user\n{it['msg']}<|im_end|>\n", add_special_tokens=False, return_tensors="pt").to(dev)
            xs.append(model.model(**enc, output_hidden_states=True).hidden_states[18][0].float().mean(0).cpu())
    del model
    torch.cuda.empty_cache()
    out = dict(ids=[it["id"] for it in items], x=torch.stack(xs))
    torch.save(out, f)
    return out


def main():
    from h4_train import HyperNet
    v = vectors()
    ids, X = v["ids"], v["x"]
    r2 = torch.load(f"{H5}/router_v2.pt", map_location="cpu", weights_only=True)
    Z = ((X - r2["mu"]) / r2["sd"]) @ r2["proj"]
    P = torch.stack([torch.sigmoid(Z @ w + b) for w, b in (r2["models"][k] for k in range(3))], 1)
    tokc = torch.tensor([r2["mean_tokens"][k] for k in range(3)])
    v2_l0, v2_l01 = P.argmax(1).tolist(), (P - 0.01 * tokc / 1000).argmax(1).tolist()
    sd = torch.load(f"{H5}/hypernet.pt", map_location="cpu", weights_only=True)
    hyper = HyperNet(sd["mu"], sd["sd"], sd["proj"])
    hyper.load_state_dict(sd)
    with torch.no_grad():
        B = hyper(X)
    g = torch.Generator().manual_seed(0)
    C = B[torch.randperm(len(B), generator=g)[:K]].clone()
    for _ in range(100):
        a = torch.cdist(B, C).argmin(1)
        C = torch.stack([B[a == k].mean(0) if (a == k).any() else C[k] for k in range(K)])
    a = torch.cdist(B, C).argmin(1).tolist()
    think = sorted(k for k in range(K) if float(C[k][12:24].mean()) > 0.5)
    json.dump(dict(K=K, centroids=C.tolist(), sizes=[a.count(k) for k in range(K)], thinking_profiles=think),
              open(f"{H5}/profiles.json", "w"))
    name = {k: f"h5-prof{k}-early1" if k in think else f"h5-prof{k}" for k in range(K)}
    routes = {i: dict(v2_l0=v2_l0[n], v2_l01=v2_l01[n], h5=a[n], model=name[a[n]], beta_mid=float(B[n, 12:24].mean()),
                      beta_early=float(B[n, :12].mean())) for n, i in enumerate(ids)}
    json.dump(dict(routes=routes, thinking_profiles=think), open(f"{H5}/routes.json", "w"))
    for k in range(K):
        json.dump([i for i, r in routes.items() if r["h5"] == k], open(f"{H5}/ids_{name[k]}.json", "w"))
    # ids already answered at 0.75 in the first long-budget run; optional, an absent file means none were
    f75 = f"{ROOT}/hyper/c32/ids_dial75.json"
    old75 = set(json.load(open(f75))) if os.path.exists(f75) else set()
    json.dump([i for i, r in routes.items() if 1 in (r["v2_l0"], r["v2_l01"]) and i not in old75],
              open(f"{H5}/ids_dial75x.json", "w"))
    for k in range(K):
        c = C[k]
        print(f"profile {k} ({name[k]}): {a.count(k)} questions; early {float(c[:12].mean()):.2f} mid "
              f"{float(c[12:24].mean()):.2f} late {float(c[24:36].mean()):.2f} emb {float(c[36]):.2f}", flush=True)
    task_of = lambda i: i.split("-", 1)[0]  # noqa: E731
    for t in TASKS:
        rs = [r for i, r in routes.items() if task_of(i) == t]
        print(f"{t}: n {len(rs)} | dial profiles {dict(sorted(collections.Counter(r['h5'] for r in rs).items()))} | "
              f"v2 l0 {dict(sorted(collections.Counter(r['v2_l0'] for r in rs).items()))} | "
              f"v2 l0.01 {dict(sorted(collections.Counter(r['v2_l01'] for r in rs).items()))}", flush=True)


if __name__ == "__main__":
    main()
