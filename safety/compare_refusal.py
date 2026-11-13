#!/usr/bin/env python3
"""Counts-only comparison of refusal runs made with refusal_check.py. Prints no prompt and no reply (it never sees one).

  python3 -I compare_refusal.py table A.counts.json B.counts.json ...
      Markdown tables of refusal counts and rates per model, one table per refusal rule found in the files.
  python3 -I compare_refusal.py pairs A.flags.jsonl B.flags.jsonl [C.flags.jsonl ...]
      For every pair of models, per prompt group and per rule: how many prompts both refused, only A, only B, neither,
      and the exact two-sided sign test on the "only A" / "only B" counts (prompts both models treated alike drop out).
      The models must have been run on the same prompt files.

Rules. refusal_check.py records two versions of its string match. Rule 1 (fields ref_a, ref_b) is the one the first draw
was scored with; rule 2 (ref_a2, ref_b2) also removes emphasis marks and knows a few more apology openers. "prefix" is
the headline match (the answer starts with a refusal phrase); "window" also accepts a refusal phrase anywhere in the
first 200 characters.
"""
import itertools
import json
import math
import sys

GROUPS = (("advbench", "AdvBench (all harmful)"), ("xstest_safe", "XSTest safe (over-refusal)"),
          ("xstest_unsafe", "XSTest unsafe"))
# (suffix of the counts keys, flag field, title)
RULES = (("", "ref_a", "Rule 1, prefix: reply starts with a refusal opener"),
         ("_w", "ref_b", "Rule 1, window: opener or refusal phrase in the first 200 characters"),
         ("_v2", "ref_a2", "Rule 2, prefix"),
         ("_w2", "ref_b2", "Rule 2, window"))
COUNT_KEYS = {"": ("refused_prefix", "rate_prefix"), "_w": ("refused_window", "rate_window"),
              "_v2": ("refused_prefix_v2", "rate_prefix_v2"), "_w2": ("refused_window_v2", "rate_window_v2")}


def sign_p(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)


def table(paths):
    runs = [json.load(open(p)) for p in paths]
    for suffix, _field, title in RULES:
        n_key, r_key = COUNT_KEYS[suffix]
        have = [r for r in runs if all(n_key in r["results"][g] for g, _ in GROUPS)]
        if not have:
            continue
        print(f"\n{title}\n")
        print("| model | " + " | ".join(t for _, t in GROUPS) + " | replies at the token cap | mean tokens per reply |")
        print("|---|" + "---|" * (len(GROUPS) + 2))
        for r in have:
            cells, capped, toks, total = [], 0, 0.0, 0
            for g, _ in GROUPS:
                s = r["results"][g]
                cells.append(f"{s[n_key]} / {s['n']} ({s[r_key]:.1f}%)")
                capped += s["capped"]
                toks += s["mean_tokens"] * s["n"]
                total += s["n"]
            print(f"| {r['label']} | " + " | ".join(cells) + f" | {capped} / {total} | {toks / total:.0f} |")


def load_flags(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        grp = "advbench" if r["ds"] == "advbench" else f"xstest_{r['label']}"
        out[(r["ds"], r["idx"])] = (grp, r)
    return out


def pairs(paths):
    runs = {p: load_flags(p) for p in paths}
    for pa, pb in itertools.combinations(paths, 2):
        A, B = runs[pa], runs[pb]
        if set(A) != set(B):
            raise SystemExit(f"{pa} and {pb} cover different prompts")
        print(f"\nA = {pa.rsplit('/', 1)[-1]}   B = {pb.rsplit('/', 1)[-1]}")
        for _suffix, field, title in RULES:
            if not all(field in next(iter(runs[p].values()))[1] for p in (pa, pb)):
                continue
            print(f"  {title}")
            for g, gt in GROUPS:
                keys = [k for k in A if A[k][0] == g]
                both = sum(A[k][1][field] and B[k][1][field] for k in keys)
                only_a = sum(A[k][1][field] and not B[k][1][field] for k in keys)
                only_b = sum(B[k][1][field] and not A[k][1][field] for k in keys)
                neither = len(keys) - both - only_a - only_b
                print(f"    {gt}: n {len(keys)} | A refused {both + only_a} | B refused {both + only_b} | both {both} | "
                      f"only A {only_a} | only B {only_b} | neither {neither} | sign test p {sign_p(only_a, only_b):.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("table", "pairs"):
        raise SystemExit(__doc__)
    (table if sys.argv[1] == "table" else pairs)(sys.argv[2:])
