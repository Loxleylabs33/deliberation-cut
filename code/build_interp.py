#!/usr/bin/env python3
"""Plain blends along the cut, no selection: Instruct + alpha * R, with R = Thinking - Instruct (alpha = 1 is the
Thinking model, except for 3,471 of about 4.0 billion elements where float32 cannot hold the difference exactly).
Baseline for the direction-selected cuts: published work on blending thinking and instruct checkpoints reports smooth
behaviour along this line. R is computed on the fly in float32 from the two Hugging Face folders.
Writes models/interpNN.gguf (bf16).
Usage: build_interp.py 25 50 75
"""
import json
import os
import shutil
import subprocess
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cutconfig import CONVERT_SCRIPT, INSTRUCT_DIR as I_DIR, ROOT, THINKING_DIR as T_DIR  # noqa: E402


def weight_map(d):
    with open(f"{d}/model.safetensors.index.json") as fh:
        return json.load(fh)["weight_map"]


def blend(alpha, names=None):
    """{name: bf16 tensor} of Instruct + alpha * (Thinking - Instruct), float32 arithmetic, for every tensor
    (or only for `names`)."""
    wi, wt = weight_map(I_DIR), weight_map(T_DIR)
    handles, out = {}, {}

    def get(d, wm, n):
        f = os.path.join(d, wm[n])
        if f not in handles:
            handles[f] = safe_open(f, framework="pt", device="cpu")
        return handles[f].get_tensor(n).to(torch.float32)

    for n in (sorted(wi) if names is None else names):
        w_i = get(I_DIR, wi, n)
        r = get(T_DIR, wt, n) - w_i  # the raw cut for this tensor
        out[n] = (w_i + alpha * r).to(torch.bfloat16).contiguous()
    return out


def main():
    for pct in (int(a) for a in sys.argv[1:]):
        alpha = pct / 100
        hf = f"{ROOT}/step3e/interp{pct}-hf"
        os.makedirs(hf, exist_ok=True)
        save_file(blend(alpha), f"{hf}/model.safetensors", metadata={"format": "pt"})
        for fname in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                      "vocab.json", "merges.txt"):
            if os.path.exists(os.path.join(T_DIR, fname)):
                shutil.copy(os.path.join(T_DIR, fname), hf)
        os.makedirs(f"{ROOT}/models", exist_ok=True)
        gguf = f"{ROOT}/models/interp{pct}.gguf"
        # the converter imports a sibling package, so do not run it in isolated mode
        subprocess.run([sys.executable, CONVERT_SCRIPT, hf, "--outtype", "bf16", "--outfile", gguf], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.rmtree(hf)
        print(f"built {gguf} (alpha {alpha})", flush=True)


if __name__ == "__main__":
    main()
