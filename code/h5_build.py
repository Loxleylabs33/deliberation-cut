#!/usr/bin/env python3
"""H5 step 5: build each H5 profile as a bf16 GGUF, as h4_profiles.py / h4_variant.py do:
W = Thinking + (beta_l - 1) * P_l per layer (P from hyper/P_q80.safetensors; the embedding/head use beta_36).
Thinking profiles (middle layers above 0.5) are built with layers 0-11 at 1 (the H4b rule) as h5-prof<k>-early1;
the others as h5-prof<k>. Skips models already built."""
import json
import os
import re
import shutil
import subprocess
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cutconfig import CONVERT_SCRIPT, ROOT, THINKING_DIR as T_DIR  # noqa: E402


def build(name, c):
    gguf = f"{ROOT}/models/{name}.gguf"
    os.makedirs(f"{ROOT}/models", exist_ok=True)
    if os.path.exists(gguf):
        print(f"{name}: already built", flush=True)
        return
    P = safe_open(f"{ROOT}/hyper/P_q80.safetensors", framework="pt", device="cpu")
    pkeys = set(P.keys())
    wm = json.load(open(f"{T_DIR}/model.safetensors.index.json"))["weight_map"]
    hf = f"{ROOT}/hyper/h5/{name}-hf"
    os.makedirs(hf, exist_ok=True)
    handles, out = {}, {}
    for n in sorted(wm):
        f = os.path.join(T_DIR, wm[n])
        if f not in handles:
            handles[f] = safe_open(f, framework="pt", device="cpu")
        W = handles[f].get_tensor(n).to(torch.float32)
        m = re.search(r"layers\.(\d+)\.", n)
        beta = float(c[int(m.group(1))]) if m else float(c[36])
        if n + ".A" in pkeys:
            W = W + (beta - 1.0) * (P.get_tensor(n + ".A").float() @ P.get_tensor(n + ".B").float())
        elif n in pkeys:
            W = W + (beta - 1.0) * P.get_tensor(n).float()
        out[n] = W.to(torch.bfloat16).contiguous()
    save_file(out, f"{hf}/model.safetensors", metadata={"format": "pt"})
    for fname in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                  "merges.txt"):
        if os.path.exists(os.path.join(T_DIR, fname)):
            shutil.copy(os.path.join(T_DIR, fname), hf)
    # the converter imports a sibling package, so do not run it in isolated mode
    subprocess.run([sys.executable, CONVERT_SCRIPT, hf, "--outtype", "bf16", "--outfile", gguf], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(hf)
    print(f"built {gguf}; dials {[round(float(x), 2) for x in c]}", flush=True)


def main():
    prof = json.load(open(f"{ROOT}/hyper/h5/profiles.json"))
    for k in range(prof["K"]):
        c = torch.tensor(prof["centroids"][k])
        if k in prof["thinking_profiles"]:
            c[0:12] = 1.0
            build(f"h5-prof{k}-early1", c)
        else:
            build(f"h5-prof{k}", c)


if __name__ == "__main__":
    main()
