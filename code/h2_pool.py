#!/usr/bin/env python3
"""Hypernetwork H2: the training pool. Questions disjoint from every test item:
  BBH examples 15..54 of each of the 6 tasks (240; tests use 0..14), MATH-500 problems not in the test set (every 10th
  is the test; take 100 of the rest, fixed seed), GSM8K TRAIN problems 1000..1099 (100; step 3a used a shuffled 400).
Answered at one dial setting per run with the same prompts, caps and scoring as step 4. Writes
hyper/pool/<label>.jsonl: {id, task, ok, tokens, thought, truncated}.
Usage: h2_pool.py LABEL GGUF [--seed-offset N]
"""
import concurrent.futures as cf
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step4_eval as s4  # noqa: E402  (same prompts, caps, scoring)

ROOT = s4.ROOT
OUT = f"{ROOT}/hyper/pool"


def pool_items():
    items = []
    for f in sorted(os.listdir(f"{ROOT}/data/bbh")):
        t = f[:-5]
        ex = json.load(open(f"{ROOT}/data/bbh/{f}"))["examples"][15:55]
        items += [("bbh", dict(id=f"pool-bbh-{t}-{i}", msg=f"{e['input']}\n\nThink it through, then put your final answer "
                   f"(the option letter in parentheses, or the exact answer) within \\boxed{{}}.", gold=e["target"]))
                  for i, e in enumerate(ex, start=15)]
    math_all = [json.loads(l) for l in open(f"{ROOT}/data/math500/test.jsonl")]
    rest = [r for k, r in enumerate(math_all) if k % 10 != 0]
    random.Random(5).shuffle(rest)
    items += [("math", dict(id=f"pool-math-{r.get('unique_id', i)}", msg=f"{r['problem']}\nPlease reason step by step, and "
               f"put your final answer within \\boxed{{}}.", gold=r["answer"])) for i, r in enumerate(rest[:100])]
    gsm = [json.loads(l) for l in open(f"{ROOT}/data/gsm8k/train.jsonl")][1000:1100]
    items += [("gsm8k", dict(id=f"pool-gsm8k-{i}", msg=f"{r['question']}\nPlease reason step by step, and put your final "
               f"answer within \\boxed{{}}.", gold=r["answer"].split("####")[-1].strip()))
              for i, r in enumerate(gsm, start=1000)]
    return items


def main():
    label, gguf = sys.argv[1], sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    out_file = f"{OUT}/{label}.jsonl"
    done = {json.loads(l)["id"] for l in open(out_file)} if os.path.exists(out_file) else set()
    todo = [(t, it) for t, it in pool_items() if it["id"] not in done]
    print(f"[{label}] {len(todo)} to answer ({len(done)} already done)", flush=True)
    if not todo:
        return
    srv = s4.start(gguf)
    try:
        with cf.ThreadPoolExecutor(s4.SLOTS) as ex, open(out_file, "a") as fh:
            futs = {ex.submit(s4.run_item, t, it, "THINK"): (t, it) for t, it in todo}
            for n, fu in enumerate(cf.as_completed(futs), 1):
                t, it = futs[fu]
                r = fu.result()
                fh.write(json.dumps(dict(id=it["id"], task=t, ok=r["ok"], tokens=r["tokens"], thought=r["thought"],
                                         truncated=r["truncated"])) + "\n")
                fh.flush()
                if n % 50 == 0:
                    print(f"[{label}] {n}/{len(todo)}", flush=True)
    finally:
        srv.terminate()
        srv.wait(timeout=60)
    rows = [json.loads(l) for l in open(out_file)]
    by = {}
    for r in rows:
        by.setdefault(r["task"], []).append(r["ok"])
    print(f"[{label}] done: " + ", ".join(f"{k} {sum(v) / len(v):.3f} (n {len(v)})" for k, v in sorted(by.items())),
          flush=True)


if __name__ == "__main__":
    main()
