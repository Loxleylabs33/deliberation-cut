#!/usr/bin/env python3
"""H4 step 1: re-answer the 440-question pool at one dial setting, KEEPING the completion text (H2 kept only
right/wrong). Same prompts, caps, sampling and scoring as step 4 / H2. Writes hyper/pool_text/<label>.jsonl:
{id, task, prompt, text, ok, tokens, truncated}. Resumable.
Usage: h4_pool_text.py LABEL GGUF
"""
import concurrent.futures as cf
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h2_pool  # noqa: E402
import step4_eval as s4  # noqa: E402

s4.KEEP_TEXT = True
OUT = f"{s4.ROOT}/hyper/pool_text"


def main():
    label, gguf = sys.argv[1], sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    out_file = f"{OUT}/{label}.jsonl"
    done = {json.loads(l)["id"] for l in open(out_file)} if os.path.exists(out_file) else set()
    todo = [(t, it) for t, it in h2_pool.pool_items() if it["id"] not in done]
    print(f"[{label}] {len(todo)} to answer ({len(done)} done)", flush=True)
    if not todo:
        return
    srv = s4.start(gguf)
    try:
        with cf.ThreadPoolExecutor(s4.SLOTS) as ex, open(out_file, "a") as fh:
            futs = {ex.submit(s4.run_item, t, it, "THINK"): (t, it) for t, it in todo}
            for n, fu in enumerate(cf.as_completed(futs), 1):
                t, it = futs[fu]
                r = fu.result()
                fh.write(json.dumps(dict(id=it["id"], task=t, prompt=s4.wrap(it["msg"], "THINK"), text=r["text"],
                                         ok=r["ok"], tokens=r["tokens"], truncated=r["truncated"])) + "\n")
                fh.flush()
                if n % 100 == 0:
                    print(f"[{label}] {n}/{len(todo)}", flush=True)
    finally:
        srv.terminate()
        srv.wait(timeout=60)
    rows = [json.loads(l) for l in open(out_file)]
    print(f"[{label}] done: {sum(r['ok'] for r in rows)}/{len(rows)} correct", flush=True)


if __name__ == "__main__":
    main()
