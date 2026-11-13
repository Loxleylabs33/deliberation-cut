#!/usr/bin/env python3
"""H5 step 1: the dial's teacher data at the 32k budget. The 440-question pool was answered at the old caps (MATH
8,192, GSM8K 4,096): 53 of 1,320 answers hit a cap (D 3, 0.75 16, Thinking 34), mostly MATH, which taught every
router that thinking loses at maths. Only those are re-answered here, at cap 32,768 (8-bit KV, 5 slots, as the
long-budget test); every other answer is kept, because an answer that finished under its cap does not depend on the
cap (c32: Thinking's MATH answers that finished under 8,192 agreed 35/35 at 32,768).
Writes hyper/pool_text32/<label>.jsonl (same rows as hyper/pool_text/, re-answered rows marked rerun32). Resumable.
Usage: h5_pool32.py LABEL GGUF
"""
import sys

sys.argv += ["--cap", "32768", "--slots", "5", "--kvq8"]  # read by step4_eval at import
import concurrent.futures as cf  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h2_pool  # noqa: E402
import step4_eval as s4  # noqa: E402

s4.KEEP_TEXT = True
OLD, OUT = f"{s4.ROOT}/hyper/pool_text", f"{s4.ROOT}/hyper/pool_text32"


def main():
    label, gguf = sys.argv[1], sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    out_file = f"{OUT}/{label}.jsonl"
    old = [json.loads(l) for l in open(f"{OLD}/{label}.jsonl")]
    if not os.path.exists(out_file):  # start with every answer that finished under its old cap
        with open(out_file, "w") as fh:
            for r in old:
                if not r["truncated"]:
                    fh.write(json.dumps(r) + "\n")
    have = {json.loads(l)["id"] for l in open(out_file)}
    redo = {r["id"] for r in old if r["truncated"]} - have
    todo = [(t, it) for t, it in h2_pool.pool_items() if it["id"] in redo]
    print(f"[{label}] {len(todo)} capped answers to redo at {s4.CAP} ({len(have)} kept)", flush=True)
    if todo:
        srv = s4.start(gguf)
        try:
            with cf.ThreadPoolExecutor(s4.SLOTS) as ex, open(out_file, "a") as fh:
                futs = {ex.submit(s4.run_item, t, it, "THINK"): (t, it) for t, it in todo}
                for fu in cf.as_completed(futs):
                    t, it = futs[fu]
                    r = fu.result()
                    fh.write(json.dumps(dict(id=it["id"], task=t, prompt=s4.wrap(it["msg"], "THINK"), text=r["text"],
                                             ok=r["ok"], tokens=r["tokens"], truncated=r["truncated"],
                                             rerun32=True)) + "\n")
                    fh.flush()
        finally:
            srv.terminate()
            srv.wait(timeout=60)
    rows = [json.loads(l) for l in open(out_file)]
    re32 = [r for r in rows if r.get("rerun32")]
    print(f"[{label}] done: {sum(r['ok'] for r in rows)}/{len(rows)} correct; re-answered {len(re32)}: "
          f"{sum(r['ok'] for r in re32)} now right, {sum(r['truncated'] for r in re32)} still capped", flush=True)


if __name__ == "__main__":
    main()
