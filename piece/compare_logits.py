#!/usr/bin/env python3
"""Compare next-token distributions of three weight sets on the same text, teacher-forced on the CPU:

  A  the released GGUF (Thinking minus P, built from float32 factors)
  B  the model rebuilt from Thinking and P_q80.safetensors with apply_piece.py (bf16 factors)
  C  Qwen3-4B-Thinking-2507 itself, for scale

For every token of every reply it reports KL(A || B) and KL(A || C) in nats, how often the most likely next token
agrees, and the total negative log-likelihood of the reply under each set. A and B differ only by the bf16 rounding of
the stored factors (see the model card); C differs from A by the whole piece.

  python3 compare_logits.py --thinking-dir DIR --piece P_q80.safetensors --gguf RELEASED.gguf --texts texts.jsonl

texts.jsonl holds one JSON object per line, {"prompt": "...", "reply": "..."}. "prompt" is the text up to and including
the opening of the assistant turn (for example "<|im_start|>user\\n2+2?<|im_end|>\\n<|im_start|>assistant\\n<think>\\n");
"reply" is the text to score. sample_texts.jsonl is a three-line example. Needs torch, transformers, safetensors, gguf.
About 8 GB of memory for the model, plus the logits (150k floats per token per weight set).
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apply_piece as ap  # noqa: E402


def load_gguf_into(model, path):
    import gguf
    reader = gguf.GGUFReader(path)
    by_name = {t.name: t for t in reader.tensors}
    sd = model.state_dict()
    n = 0
    for hf_name in ap.weight_map(ARGS.thinking_dir):
        t = by_name[ap.gguf_name(hf_name)]
        raw = t.data.reshape(-1)
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            vals = torch.from_numpy(raw.astype("float32")).to(torch.bfloat16)
        else:
            vals = torch.from_numpy(raw.view("int16").copy()).view(torch.bfloat16)
        with torch.no_grad():
            sd[hf_name].copy_(vals.reshape(sd[hf_name].shape))
        n += 1
    return n


def load_rebuilt_into(model, thinking_dir, piece, beta):
    sd = model.state_dict()
    n = 0
    for name, w, _kind, _ss in ap.rebuilt(thinking_dir, piece, beta):
        with torch.no_grad():
            sd[name].copy_(w)
        n += 1
    return n


def logprobs(model, batches):
    """List of [reply_tokens, vocab] float32 log-probabilities, one per text."""
    out = []
    with torch.no_grad():
        for ids, start in batches:
            logits = model(input_ids=ids.unsqueeze(0), use_cache=False).logits[0, start - 1:-1].float()
            out.append(torch.log_softmax(logits, dim=-1))
    return out


def main():
    global ARGS
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--thinking-dir", required=True)
    p.add_argument("--piece", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--texts", required=True)
    p.add_argument("--max-tokens", type=int, default=300, help="score at most this many reply tokens per text")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--report", default="")
    ARGS = p.parse_args()
    if ARGS.threads:
        torch.set_num_threads(ARGS.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ARGS.thinking_dir)
    batches, refs = [], []
    for line in open(ARGS.texts):
        if not line.strip():
            continue
        r = json.loads(line)
        pid = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        rid = tok(r["reply"], add_special_tokens=False)["input_ids"][:ARGS.max_tokens]
        batches.append((torch.tensor(pid + rid), len(pid)))
        refs.append(torch.tensor(rid))
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(ARGS.thinking_dir, dtype=torch.bfloat16).eval()
    lp_c = logprobs(model, batches)  # C: Thinking, as loaded
    print(f"C done {time.time() - t0:.0f} s", flush=True)
    load_rebuilt_into(model, ARGS.thinking_dir, ARGS.piece, 0.0)
    lp_b = logprobs(model, batches)
    print(f"B done {time.time() - t0:.0f} s", flush=True)
    load_gguf_into(model, ARGS.gguf)
    lp_a = logprobs(model, batches)
    print(f"A done {time.time() - t0:.0f} s", flush=True)

    def kl(p_log, q_log):  # KL(p || q), per position
        return (p_log.exp() * (p_log - q_log)).sum(-1)

    rows = {}
    for name, other in (("A_vs_B (released vs rebuilt from bf16 P)", lp_b), ("A_vs_C (released vs Thinking)", lp_c)):
        kls = torch.cat([kl(a, b) for a, b in zip(lp_a, other)])
        agree = torch.cat([(a.argmax(-1) == b.argmax(-1)).float() for a, b in zip(lp_a, other)])
        rows[name] = dict(tokens=int(kls.numel()), mean_kl_nats=float(kls.mean()), median_kl_nats=float(kls.median()),
                          max_kl_nats=float(kls.max()), top1_agreement_pct=100 * float(agree.mean()))
    nll = {}
    for name, lps in (("A released", lp_a), ("B rebuilt", lp_b), ("C Thinking", lp_c)):
        tot = sum(-float(lp[torch.arange(len(ref)), ref].sum()) for lp, ref in zip(lps, refs))
        nll[name] = dict(total_nll_nats=tot, mean_nll_per_token=tot / sum(len(r) for r in refs))
    report = dict(texts=len(batches), reply_tokens=sum(len(r) for r in refs), comparisons=rows, nll=nll,
                  seconds=round(time.time() - t0))
    print(json.dumps(report, indent=1))
    if ARGS.report:
        with open(ARGS.report, "w") as fh:
            json.dump(report, fh, indent=1)


ARGS = None

if __name__ == "__main__":
    main()
