#!/usr/bin/env python3
"""Ask a saved router which setting to use for a question.

    python3 -I route_question.py --router router_pool32k_h5/router_v2_pool32k.pt \
        --instruct-dir /path/to/Qwen3-4B-Instruct-2507 "Janet has 3 apples and buys 5 more. How many now?"

What it does. The router reads the question as the first 18 decoder layers of Qwen3-4B-Instruct-2507 see it: the user
turn  <|im_start|>user\\n{question}<|im_end|>\\n  goes through the model and the layer-18 hidden states are averaged over
the question's tokens (one 2,560-number vector). A stored mean and standard deviation standardise the vector, a stored
64-direction PCA projection shrinks it, and three logistic models give p(correct) for the three settings:
    0  D, the released model (Thinking minus P)           1  D + 0.75 P (Thinking minus 0.25 P)        2  Thinking
The choice is argmax_k  p_k - lambda * mean_tokens_k / 1000  (lambda 0 = accuracy first; 0.01 = balanced). The mean
token counts are stored in the router file (averages over its training pool). Needs torch and transformers; runs on the
CPU by default (a few seconds per short question after the model loads) and stops after layer 18.

What the router was trained on: questions in the three prompt styles below, from GSM8K, MATH and six BIG-Bench Hard
logic tasks. Anything else is untested. Use --wrap to put your question in the same style:
  --wrap math   <question>\\nPlease reason step by step, and put your final answer within \\boxed{}.     (default)
  --wrap logic  <question>\\n\\nThink it through, then put your final answer (the option letter in parentheses, or the
                exact answer) within \\boxed{}.
  --wrap none   the question exactly as given

What to do with the answer: setting 0 is the released GGUF; setting 2 is Qwen/Qwen3-4B-Thinking-2507; setting 1 is built
with piece/apply_piece.py --beta 0.75. This script only chooses; it does not switch models.
"""
import argparse
import json
import os
import sys

import torch

LAYER = 18
WRAPS = {
    "math": "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    "logic": "{q}\n\nThink it through, then put your final answer (the option letter in parentheses, or the exact "
             "answer) within \\boxed{{}}.",
    "none": "{q}",
}
NAMES = ["0 (D, the released model)", "1 (D + 0.75 P)", "2 (Thinking)"]


class _Done(Exception):
    pass


def embed(texts, instruct_dir, device, dtype):
    """Mean layer-18 hidden state of the user turn, one row per text; the forward pass stops after layer 18."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(instruct_dir)
    model = AutoModelForCausalLM.from_pretrained(instruct_dir, dtype=dtype).to(device).eval()
    store = {}

    def hook(_module, _inputs, output):
        store["h"] = output[0] if isinstance(output, tuple) else output
        raise _Done

    handle = model.model.layers[LAYER - 1].register_forward_hook(hook)
    rows = []
    try:
        for t in texts:
            enc = tok(f"<|im_start|>user\n{t}<|im_end|>\n", add_special_tokens=False, return_tensors="pt").to(device)
            try:
                with torch.no_grad():
                    model.model(**enc, use_cache=False)
            except _Done:
                pass
            rows.append(store["h"][0].float().mean(dim=0).cpu())
    finally:
        handle.remove()
    return torch.stack(rows)


def decide(router, X, lam):
    z = ((X - router["mu"]) / router["sd"]) @ router["proj"]
    p = torch.stack([torch.sigmoid(z @ w + b) for w, b in (router["models"][k] for k in range(3))], dim=1)
    tok = torch.tensor([router["mean_tokens"][k] for k in range(3)])
    return p, (p - lam * tok / 1000).argmax(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("question", nargs="*", help="one or more questions")
    ap.add_argument("--file", help="text file with one question per line")
    ap.add_argument("--router", required=True)
    ap.add_argument("--instruct-dir", required=True, help="Hugging Face folder of Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.01)
    ap.add_argument("--wrap", choices=sorted(WRAPS), default="math")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--json", action="store_true", help="print one JSON object per question")
    ap.add_argument("--compare-with", metavar="EMBED_PT:ID", help="check the first question's vector against a stored "
                    "row of embed.pt (use the exact message text the row was made from)")
    args = ap.parse_args()
    qs = list(args.question)
    if args.file:
        qs += [l.rstrip("\n") for l in open(args.file) if l.strip()]
    if not qs:
        sys.exit("give a question or --file")
    router = torch.load(args.router, map_location="cpu", weights_only=True)
    msgs = [WRAPS[args.wrap].format(q=q) for q in qs]
    X = embed(msgs, args.instruct_dir, torch.device(args.device), getattr(torch, args.dtype))
    p, choice = decide(router, X, args.lam)
    if args.compare_with:
        path, rid = args.compare_with.rsplit(":", 1)
        stored = torch.load(path, weights_only=True)
        ref = stored["x"][stored["ids"].index(rid)]
        cos = float(torch.nn.functional.cosine_similarity(X[0], ref, dim=0))
        print(f"vector vs stored row {rid}: cosine {cos:.6f}, max abs diff {float((X[0] - ref).abs().max()):.4f}, "
              f"relative L2 diff {float((X[0] - ref).norm() / ref.norm()):.4f}")
    for q, row, c in zip(qs, p.tolist(), choice.tolist()):
        tokens = [router["mean_tokens"][k] for k in range(3)]
        if args.json:
            print(json.dumps(dict(question=q, setting=c, p_correct=row, mean_tokens=tokens, lam=args.lam)))
        else:
            print(f"\n{q[:100]}{'...' if len(q) > 100 else ''}\n  chosen setting: {NAMES[c]}  (lambda {args.lam})")
            for k in range(3):
                print(f"    {NAMES[k]:28s} p(correct) {row[k]:.3f}   mean tokens in the training pool {tokens[k]:.0f}")


if __name__ == "__main__":
    main()
