#!/usr/bin/env python3
"""Hypernetwork H3 input: what the language model 'sees' in each question.
For every pool and test question: the Instruct model reads the user turn, and the hidden states of its middle layer
(18 of 36) are mean-pooled over the question's tokens -> one 2,560-number vector. Runs on EMBED_DEVICE (default: the
first CUDA device if there is one, else the CPU). Writes hyper/embed.pt: {"ids": [...], "tasks": [...], "x": tensor}.
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h2_pool  # noqa: E402
import step4_eval as s4  # noqa: E402
from cutconfig import INSTRUCT_DIR  # noqa: E402

ROOT = s4.ROOT
HF = INSTRUCT_DIR
LAYER = 18


def all_questions():
    qs = [(t, it["id"], it["msg"]) for t, it in h2_pool.pool_items()]
    for t in ("gsm8k", "math", "bbh"):
        qs += [(t, it["id"], it["msg"]) for it in s4.items_for(t)]
    return qs


def main():
    os.makedirs(f"{ROOT}/hyper", exist_ok=True)
    dev = torch.device(os.environ.get("EMBED_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))
    print("device:", torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu", flush=True)
    tok = AutoTokenizer.from_pretrained(HF)
    model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16).to(dev).eval()
    qs = all_questions()
    xs, ids, tasks = [], [], []
    with torch.no_grad():
        for n, (t, i, msg) in enumerate(qs):
            enc = tok(f"<|im_start|>user\n{msg}<|im_end|>\n", add_special_tokens=False, return_tensors="pt").to(dev)
            out = model.model(**enc, output_hidden_states=True)
            h = out.hidden_states[LAYER][0].float().mean(dim=0).cpu()
            xs.append(h)
            ids.append(i)
            tasks.append(t)
            if (n + 1) % 100 == 0:
                print(f"{n + 1}/{len(qs)}", flush=True)
    torch.save({"ids": ids, "tasks": tasks, "x": torch.stack(xs), "layer": LAYER}, f"{ROOT}/hyper/embed.pt")
    print(f"saved {len(ids)} question vectors", flush=True)


if __name__ == "__main__":
    main()
