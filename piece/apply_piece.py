#!/usr/bin/env python3
"""Apply the removed piece P (P_q80.safetensors) to Qwen3-4B-Thinking-2507.

    W(beta) = Thinking + (beta - 1) * P

  beta = 0    Thinking minus P: the released model, up to the bf16 rounding described below
  beta = 1    the Thinking model itself
  0 < beta < 1  the intermediate "dial" settings (the routers choose among 0, 0.75 and 1)

P is stored in factored form. For each of the 252 weight matrices M the file holds M.A (out x r) and M.B (r x in) with
P_M = A @ B, r between 34 and 986 (73,871 directions in total). Forty-seven tensors are stored whole: the embedding
table (bf16) and 46 norm vectors (float32). The arithmetic is done in float32 and the result is rounded to bf16, the
same recipe the released file was built with.

The rounding caveat. The released file was built from float32 factors. P_q80.safetensors holds the factors rounded to
bf16 (and the embedding difference rounded to bf16), so applying it to the original weights reproduces the released
weights only to within that rounding. --compare-gguf measures the gap tensor by tensor; the model card quotes the result.

Usage:
  python3 apply_piece.py --thinking-dir DIR --piece P_q80.safetensors --beta 0 --out OUTDIR
      writes OUTDIR/model.safetensors (bf16) and copies the config and tokenizer files, ready for transformers or for
      llama.cpp's convert_hf_to_gguf.py
  python3 apply_piece.py --thinking-dir DIR --piece P_q80.safetensors --beta 0 --compare-gguf RELEASED.gguf [--report r.json]
      rebuilds the weights in memory and compares them with the GGUF, tensor by tensor (nothing is written but the report)
DIR is a Hugging Face folder of Qwen/Qwen3-4B-Thinking-2507. Needs torch, safetensors and (for --compare-gguf) gguf.
"""
import argparse
import json
import os
import re
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SIDE_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
              "merges.txt")
GGUF_LAYER = {
    "input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight", "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}
CHUNK = 1 << 26  # elements per comparison slice


def weight_map(d):
    with open(os.path.join(d, "model.safetensors.index.json")) as fh:
        return json.load(fh)["weight_map"]


def gguf_name(n):
    """Hugging Face tensor name -> llama.cpp tensor name for the Qwen3 architecture (tied embeddings: no lm_head)."""
    fixed = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
             "lm_head.weight": "output.weight"}
    if n in fixed:
        return fixed[n]
    layer, rest = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", n).groups()
    return f"blk.{layer}.{GGUF_LAYER[rest]}"


def rebuilt(thinking_dir, piece_path, beta):
    """Yield (name, bf16 tensor, kind, piece_sumsq) for every tensor of the rebuilt model.
    kind is 'matrix' (factored in P), 'unit' (stored whole in P) or 'untouched'."""
    wm = weight_map(thinking_dir)
    piece = safe_open(piece_path, framework="pt", device="cpu")
    keys = set(piece.keys())
    handles = {}
    for n in sorted(wm):
        f = os.path.join(thinking_dir, wm[n])
        if f not in handles:
            handles[f] = safe_open(f, framework="pt", device="cpu")
        w = handles[f].get_tensor(n).to(torch.float32)
        kind, ss = "untouched", 0.0
        if n + ".A" in keys:
            delta = (beta - 1.0) * (piece.get_tensor(n + ".A").float() @ piece.get_tensor(n + ".B").float())
            w, kind, ss = w + delta, "matrix", float((delta.double() ** 2).sum())
        elif n in keys:
            delta = (beta - 1.0) * piece.get_tensor(n).float()
            w, kind, ss = w + delta, "unit", float((delta.double() ** 2).sum())
        yield n, w.to(torch.bfloat16).contiguous(), kind, ss


def ordered(bits16):
    """int16 bit patterns of bf16 values -> integers that increase with the value (so |difference| = distance in ulps)."""
    i = bits16.to(torch.int32)
    return torch.where(i >= 0, i, -(i & 0x7FFF))


def compare_tensor(mine, ref_np, ref_is_f32):
    """Statistics of one rebuilt bf16 tensor against the released tensor (numpy array from the GGUF)."""
    a = mine.reshape(-1)
    if ref_is_f32:
        b = torch.from_numpy(ref_np.reshape(-1).astype("float32")).to(torch.bfloat16)
        exact_f32 = bool(torch.equal(b.float(), torch.from_numpy(ref_np.reshape(-1).astype("float32"))))
    else:
        b = torch.from_numpy(ref_np.reshape(-1).view("int16").copy()).view(torch.bfloat16)
        exact_f32 = True
    assert a.numel() == b.numel(), "element count differs"
    n, n_eq, n_1, max_abs, ss_diff, ss_ref = a.numel(), 0, 0, 0.0, 0.0, 0.0
    for s in range(0, n, CHUNK):
        xa, xb = a[s:s + CHUNK], b[s:s + CHUNK]
        d = (ordered(xa.view(torch.int16)) - ordered(xb.view(torch.int16))).abs()
        n_eq += int((d == 0).sum())
        n_1 += int((d == 1).sum())
        diff = (xa.float() - xb.float()).abs()
        max_abs = max(max_abs, float(diff.max()))
        ss_diff += float((diff.double() ** 2).sum())
        ss_ref += float((xb.float().double() ** 2).sum())
    return dict(n=n, equal=n_eq, one_ulp=n_1, more=n - n_eq - n_1, max_abs=max_abs, ss_diff=ss_diff, ss_ref=ss_ref,
                gguf_f32_exactly_bf16=exact_f32)


def compare(args):
    import gguf
    reader = gguf.GGUFReader(args.compare_gguf)
    tensors = {t.name: t for t in reader.tensors}
    seen, groups, t0 = set(), {}, time.time()
    for n, w, kind, ss in rebuilt(args.thinking_dir, args.piece, args.beta):
        t = tensors[gguf_name(n)]
        seen.add(t.name)
        is_f32 = t.tensor_type == gguf.GGMLQuantizationType.F32
        if not is_f32 and t.tensor_type != gguf.GGMLQuantizationType.BF16:
            raise SystemExit(f"{t.name}: unexpected GGUF type {t.tensor_type.name}")
        s = compare_tensor(w, t.data, is_f32)
        g = groups.setdefault(kind, dict(tensors=0, n=0, equal=0, one_ulp=0, more=0, max_abs=0.0, ss_diff=0.0,
                                         ss_ref=0.0, piece_ss=0.0, not_exact_bf16=0))
        g["tensors"] += 1
        for k in ("n", "equal", "one_ulp", "more", "ss_diff", "ss_ref"):
            g[k] += s[k]
        g["max_abs"] = max(g["max_abs"], s["max_abs"])
        g["piece_ss"] += ss
        g["not_exact_bf16"] += 0 if s["gguf_f32_exactly_bf16"] else 1
    missing = sorted(set(tensors) - seen)
    rows = {}
    for kind, g in groups.items():
        rows[kind] = dict(
            tensors=g["tensors"], elements=g["n"], identical=g["equal"], differ_by_one_ulp=g["one_ulp"],
            differ_by_more=g["more"], pct_identical=round(100 * g["equal"] / g["n"], 4),
            pct_one_ulp=round(100 * g["one_ulp"] / g["n"], 4), pct_more=round(100 * g["more"] / g["n"], 6),
            max_abs_diff=g["max_abs"], rel_frobenius_vs_released=(g["ss_diff"] / g["ss_ref"]) ** 0.5,
            rel_frobenius_vs_piece=((g["ss_diff"] / g["piece_ss"]) ** 0.5) if g["piece_ss"] else None,
            gguf_f32_tensors_not_exactly_bf16=g["not_exact_bf16"])
    tot = {k: sum(g[k] for g in groups.values()) for k in ("n", "equal", "one_ulp", "more", "ss_diff", "ss_ref")}
    rows["all"] = dict(elements=tot["n"], identical=tot["equal"], differ_by_one_ulp=tot["one_ulp"],
                       differ_by_more=tot["more"], pct_identical=round(100 * tot["equal"] / tot["n"], 4),
                       pct_one_ulp=round(100 * tot["one_ulp"] / tot["n"], 4),
                       pct_more=round(100 * tot["more"] / tot["n"], 6),
                       max_abs_diff=max(g["max_abs"] for g in groups.values()),
                       rel_frobenius_vs_released=(tot["ss_diff"] / tot["ss_ref"]) ** 0.5)
    report = dict(beta=args.beta, gguf=os.path.basename(args.compare_gguf), tensors_compared=len(seen),
                  gguf_tensors_not_matched=missing, groups=rows, seconds=round(time.time() - t0))
    print(json.dumps(report, indent=1))
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=1)


def write_model(args):
    os.makedirs(args.out, exist_ok=True)
    out = {n: w for n, w, _, _ in rebuilt(args.thinking_dir, args.piece, args.beta)}
    save_file(out, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})
    for f in SIDE_FILES:
        if os.path.exists(os.path.join(args.thinking_dir, f)):
            shutil.copy(os.path.join(args.thinking_dir, f), args.out)
    print(f"wrote {args.out}/model.safetensors ({len(out)} tensors, beta {args.beta})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--thinking-dir", required=True, help="Hugging Face folder of Qwen3-4B-Thinking-2507")
    ap.add_argument("--piece", required=True, help="P_q80.safetensors")
    ap.add_argument("--beta", type=float, default=0.0, help="0 = Thinking minus P (default), 1 = Thinking")
    ap.add_argument("--out", default="", help="write the rebuilt model here")
    ap.add_argument("--compare-gguf", default="", help="compare the rebuilt weights with this GGUF")
    ap.add_argument("--report", default="", help="with --compare-gguf: also write the report as JSON")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = library default)")
    args = ap.parse_args()
    if not args.out and not args.compare_gguf:
        sys.exit("give --out and/or --compare-gguf")
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.compare_gguf:
        compare(args)
    if args.out:
        write_model(args)


if __name__ == "__main__":
    main()
