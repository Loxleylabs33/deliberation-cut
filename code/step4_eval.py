#!/usr/bin/env python3
"""Reasoning cut, step 4: the scorecard. Serves one GGUF with llama.cpp's server and runs fixed test sets.

Every model gets the SAME raw prompts (the GGUF's own template is not used):
  THINK   = user turn, then "<|im_start|>assistant\n<think>\n"                (invites reasoning)
  NOTHINK = user turn, then "<|im_start|>assistant\n<think>\n\n</think>\n\n"  (answer directly)
  NATIVE  = user turn, then "<|im_start|>assistant\n"                          (Instruct's own format; --native)
Tasks (fixed subsets, same items for every model):
  gsm8k   first 300 of GSM8K test, THINK, answer in \boxed{}          (on target: everyday maths)
  math    every 5th of MATH-500 (100), THINK, \boxed{}                (on target: harder maths)
  bbh     first 30 of 6 BIG-Bench Hard logic tasks (180), THINK       (on target: logic)
  mmlu    500 MMLU test questions, NOTHINK, next-token choice A-D     (off target: knowledge)
  ifeval  first 200 IFEval prompts, NOTHINK, strict check             (off target: instruction following)
Reasoning tasks sample at the Thinking card's settings (T 0.6, top_p 0.95, top_k 20) with a fixed seed per item and
a 4096-token cap; IFEval at T 0.7, top_p 0.8. Also recorded: whether a real think block was written (>= 50 tokens
before </think>), output length, truncations.
Usage: step4_eval.py LABEL GGUF [--tasks gsm8k,math,bbh,mmlu,ifeval] [--native]
Writes step4/LABEL/{results.json,items.jsonl}.
"""
import concurrent.futures as cf
import csv
import glob
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cutconfig import LLAMA_SERVER, ROOT  # noqa: E402

PORT = 18931
SLOTS = int(sys.argv[sys.argv.index("--slots") + 1]) if "--slots" in sys.argv else 8
# At a 4,096 cap the Thinking model ran out of room on 54% of MATH and 32% of BBH items, which mixes reasoning
# length into the score; GSM8K (5% capped) keeps 4,096, MATH and BBH get 8,192 on smaller subsets.
CAPS = {"gsm8k": 4096, "math": 8192, "bbh": 8192, "bbh2": 8192, "mathhard": 8192, "bbhhard": 8192}
# --cap N sets every reasoning task's cap (32,768 = the Qwen3-4B-Thinking-2507 card's setting for most queries);
# --kvq8 stores the KV cache at 8 bits (flash attention on) so 5 slots of 32k fit beside the bf16 weights on a
# 24 GB card.
if "--cap" in sys.argv:
    CAPS = {k: int(sys.argv[sys.argv.index("--cap") + 1]) for k in CAPS}
CAP = max(CAPS.values())
PAD = 1024 if "--cap" in sys.argv else 512  # prompt room per slot
KVQ8 = "--kvq8" in sys.argv
REASON = ("gsm8k", "math", "bbh", "bbh2", "mathhard", "bbhhard")
sys.path.insert(0, f"{ROOT}/ifeval_pkg")


def user(msg):
    return f"<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"


def wrap(msg, mode):
    if mode == "THINK":
        return user(msg) + "<think>\n"
    if mode == "NOTHINK":
        return user(msg) + "<think>\n\n</think>\n\n"
    return user(msg)


def start(gguf):
    # GPU choice and library paths come from the caller's environment (CUDA_VISIBLE_DEVICES, LD_LIBRARY_PATH, ...)
    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    log = open(f"{ROOT}/logs/server-step4.log", "w")
    kv = ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"] if KVQ8 else []
    srv = subprocess.Popen([LLAMA_SERVER, "-m", gguf, "-ngl", "99", "-c", str(SLOTS * (CAP + PAD)),
                            "-np", str(SLOTS), "--host", "127.0.0.1", "--port", str(PORT)] + kv, stdout=log,
                           stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return srv
        except Exception:
            pass
        if srv.poll() is not None:
            raise SystemExit("llama-server exited; see logs/server-step4.log")
        time.sleep(1)
    raise SystemExit("llama-server not healthy")


def complete(prompt, **kw):
    body = dict(prompt=prompt, stop=["<|im_end|>"], cache_prompt=True, **kw)
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return s[i + 7:j - 1] if depth == 0 else None


def norm_math(a):
    if a is None:
        return None
    a = a.strip().replace(" ", "").replace("\\left", "").replace("\\right", "").replace("\\!", "")
    a = a.replace("dfrac", "frac").replace("tfrac", "frac").replace("^\\circ", "").replace("^{\\circ}", "")
    a = a.replace("\\$", "").replace("$", "").rstrip(".")
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    return a


def num(a):
    try:
        return float(str(a).replace(",", "").replace("\\$", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def think_info(text):
    if "</think>" not in text:
        return False, text
    pre, post = text.split("</think>", 1)
    return len(pre.split()) >= 40, post


SCREEN = "--screen" in sys.argv  # quick triage: GSM8K 50, BBH 6 x 5, MMLU 500; see main()


def items_for(task):
    if task == "gsm8k":
        rows = [json.loads(l) for l in open(f"{ROOT}/data/gsm8k/test.jsonl")][:50 if SCREEN else 300]
        return [dict(id=f"gsm8k-{i}", msg=f"{r['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
                     gold=r["answer"].split("####")[-1].strip()) for i, r in enumerate(rows)]
    if task == "math":
        rows = [json.loads(l) for l in open(f"{ROOT}/data/math500/test.jsonl")][::10]  # 50 problems
        return [dict(id=f"math-{r.get('unique_id', i)}", msg=f"{r['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
                     gold=r["answer"]) for i, r in enumerate(rows)]
    if task == "bbh":
        out = []
        for f in sorted(glob.glob(f"{ROOT}/data/bbh/*.json")):
            t = os.path.basename(f)[:-5]
            for i, e in enumerate(json.load(open(f))["examples"][:5 if SCREEN else 15]):  # 6 x 15 = 90 (screen 6 x 5)
                out.append(dict(id=f"bbh-{t}-{i}", msg=f"{e['input']}\n\nThink it through, then put your final answer "
                                f"(the option letter in parentheses, or the exact answer) within \\boxed{{}}.", gold=e["target"]))
        return out
    if task == "bbhhard":  # harder logic tasks (fetch_data.sh); examples 0-49 of each task
        out = []
        for f in sorted(glob.glob(f"{ROOT}/data/bbh_hard/*.json")):
            t = os.path.basename(f)[:-5]
            ask = ("Think it through, then write only the closing brackets you add, separated by spaces, on the last "
                   "line after \"Answer:\"." if t == "dyck_languages" else
                   "Think it through, then put your final answer (the option letter in parentheses, or the exact "
                   "answer) within \\boxed{}.")
            for i, e in enumerate(json.load(open(f))["examples"][:50]):
                out.append(dict(id=f"bbhhard-{t}-{i}", msg=f"{e['input']}\n\n{ask}", gold=e["target"]))
        return out
    if task == "mathhard":  # the 96 level-5 MATH-500 problems in neither the test set nor the pool
        rows = [json.loads(l) for l in open(f"{ROOT}/data/math500/test.jsonl")]
        rest = [k for k in range(len(rows)) if k % 10 != 0]
        used = {k for k in range(len(rows)) if k % 10 == 0} | set(rest[:100])  # test set | h2_pool's MATH share
        return [dict(id=f"mathhard-{rows[k].get('unique_id', k)}", msg=f"{rows[k]['problem']}\nPlease reason step by "
                     f"step, and put your final answer within \\boxed{{}}.", gold=rows[k]["answer"])
                for k in range(len(rows)) if k not in used and rows[k]["level"] == 5]
    if task == "bbh2":  # held-out logic set: examples 55-69 of each task, untouched by any earlier step
        out = []
        for f in sorted(glob.glob(f"{ROOT}/data/bbh/*.json")):
            t = os.path.basename(f)[:-5]
            for i, e in enumerate(json.load(open(f))["examples"][55:70], start=55):
                out.append(dict(id=f"bbh2-{t}-{i}", msg=f"{e['input']}\n\nThink it through, then put your final answer "
                                f"(the option letter in parentheses, or the exact answer) within \\boxed{{}}.", gold=e["target"]))
        return out
    if task == "mmlu":
        rows = []
        for f in sorted(glob.glob(f"{ROOT}/data/mmlu/data/test/*_test.csv")):
            for r in csv.reader(open(f)):
                if len(r) == 6:
                    rows.append(r)
        random.Random(42).shuffle(rows)
        return [dict(id=f"mmlu-{i}", msg=f"{q}\n\n(A) {a}\n(B) {b}\n(C) {c}\n(D) {d}\n\nAnswer with the letter of the correct option.",
                     gold=ans) for i, (q, a, b, c, d, ans) in enumerate(rows[:500])]
    if task == "ifeval":
        rows = [json.loads(l) for l in open(f"{ROOT}/data/ifeval/input_data.jsonl")][:200]
        return [dict(id=f"ifeval-{r['key']}", msg=r["prompt"], raw=r) for r in rows]
    raise SystemExit(f"unknown task {task}")


KEEP_TEXT = "--keep-text" in sys.argv  # callers that need the completion text (H4 pool) also set it True


def grade_bbhhard(it, pred):
    """Per-task graders for the downloaded harder logic tasks; formatting cannot decide right or wrong."""
    if pred is None:
        return False
    g = it["gold"].strip()
    p = re.sub(r"\\text\{([^}]*)\}", r"\1", pred).strip().strip("$`").strip()
    if "-dyck_languages-" in it["id"]:
        if p.startswith("\\boxed{") and p.endswith("}"):
            p = p[7:-1]
        clean = lambda s: re.sub(r"[\s`$]", "", s.replace("\\{", "{").replace("\\}", "}"))  # noqa: E731
        return clean(p) == clean(g)
    if "-multistep_arithmetic_two-" in it["id"]:
        a, b = num(p.replace("\u2212", "-")), num(g)
        return a is not None and b is not None and abs(a - b) < 1e-6
    m = re.match(r"\(?([A-G])\)?(\s|$|\.)", p + " ")  # seven-option multiple choice: the letter decides
    return bool(m) and m.group(1) == g.strip("()")
SEED_OFFSET = int(sys.argv[sys.argv.index("--seed-offset") + 1]) if "--seed-offset" in sys.argv else 0
# --only FILE = JSON list of item ids; each task runs only those (routed subsets for the comparison
# with the original models). Seeds are per id as in a full run (parallel slots can still batch differently).
ONLY = set(json.load(open(sys.argv[sys.argv.index("--only") + 1]))) if "--only" in sys.argv else None


def run_item(task, it, mode):
    seed = (zlib.crc32(it["id"].encode()) + SEED_OFFSET) % (2**31)  # stable across runs; offset = a fresh draw
    if task in REASON:
        cap = CAPS[task]
        out = complete(wrap(it["msg"], mode), n_predict=cap, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, seed=seed)
        text = out["content"]
        capped = (out.get("tokens_predicted") or 0) >= cap
        if "</think>" in text:
            pre, post = text.split("</think>", 1)
            thought = len(pre.split()) >= 40  # a real, closed think block
        elif capped:
            thought, post = False, ""  # ran out of room before answering
        else:
            # a model never trained on think tags (Instruct) answers straight after the opening tag and
            # stops without closing it; its whole reply is the answer. Not counted as a think block.
            thought, post = False, text
        if task == "bbhhard" and "-dyck_languages-" in it["id"]:
            m = post.rfind("Answer:")  # bracket answers can contain "}", which \boxed{} cannot hold
            pred = post[m + 7:].strip().split("\n")[0].strip() if m >= 0 else None
        else:
            pred = boxed(post)
        if task == "gsm8k":
            ok = num(pred) is not None and num(it["gold"]) is not None and abs(num(pred) - num(it["gold"])) < 1e-6
        elif task in ("math", "mathhard"):
            ok = pred is not None and norm_math(pred) == norm_math(it["gold"])
        elif task == "bbhhard":
            ok = grade_bbhhard(it, pred)
        else:
            g = it["gold"].strip()
            p = (pred or "").strip()
            ok = p == g or p.strip("()") == g.strip("()") or p.lower() == g.lower()
        res = dict(id=it["id"], ok=bool(ok), pred=pred, thought=thought, tokens=out.get("tokens_predicted"),
                   truncated=(out.get("tokens_predicted") or 0) >= cap, closed="</think>" in text)
        if KEEP_TEXT:  # H4 training targets need the full completion
            res["text"] = text
        return res
    if task == "mmlu":
        out = complete(wrap(it["msg"], mode) + "The answer is (", n_predict=1, n_probs=20, temperature=0.0)
        first = (out.get("completion_probabilities") or [{}])[0]
        probs = first.get("top_probs") or first.get("top_logprobs") or []
        best = None
        for p in probs:  # listed most likely first
            t = p.get("token", "").strip()
            if t in ("A", "B", "C", "D"):
                best = t
                break
        return dict(id=it["id"], ok=best == it["gold"], pred=best)
    if task == "ifeval":
        from instruction_following_eval import evaluation_lib as ev
        out = complete(wrap(it["msg"], mode), n_predict=1536, temperature=0.7, top_p=0.8, top_k=20, seed=seed)
        text = out["content"]
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        r = it["raw"]
        res = ev.test_instruction_following_strict(
            ev.InputExample(key=r["key"], instruction_id_list=r["instruction_id_list"], prompt=r["prompt"],
                            kwargs=r["kwargs"]), {r["prompt"]: text.strip()})
        return dict(id=it["id"], ok=bool(res.follow_all_instructions), tokens=out.get("tokens_predicted"))


def main():
    label, gguf = sys.argv[1], sys.argv[2]
    tasks = sys.argv[sys.argv.index("--tasks") + 1].split(",") if "--tasks" in sys.argv else \
        ["gsm8k", "math", "bbh", "mmlu", "ifeval"]
    native = "--native" in sys.argv
    if SCREEN and "--tasks" not in sys.argv:
        tasks = ["gsm8k", "bbh", "mmlu"]
    out_dir = f"{ROOT}/{'step4_screen' if SCREEN else 'step4'}/{label}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    srv = start(gguf)
    res_file = f"{out_dir}/results.json"
    results = json.load(open(res_file))["results"] if os.path.exists(res_file) else {}  # merge, per task
    t_all = time.time()
    try:
        if True:
            for task in tasks:
                mode = "NATIVE" if native else ("THINK" if task in REASON else "NOTHINK")
                its, t0 = items_for(task), time.time()
                if ONLY is not None:
                    its = [it for it in its if it["id"] in ONLY]
                    if not its:
                        print(f"[{label}] {task}: no ids in --only, skipped", flush=True)
                        continue
                with cf.ThreadPoolExecutor(SLOTS) as ex:
                    res = list(ex.map(lambda it: run_item(task, it, mode), its))
                with open(f"{out_dir}/items_{task}.jsonl", "w") as fi:
                    for r in res:
                        fi.write(json.dumps(dict(task=task, mode=mode, **r)) + "\n")
                acc = sum(r["ok"] for r in res) / len(res)
                summ = dict(mode=mode, n=len(res), accuracy=round(acc, 4), seconds=round(time.time() - t0))
                if task in REASON:
                    toks = sorted(r["tokens"] or 0 for r in res)
                    summ.update(thought_rate=round(sum(r["thought"] for r in res) / len(res), 4),
                                closed_rate=round(sum(r["closed"] for r in res) / len(res), 4),
                                median_tokens=toks[len(toks) // 2], truncated=sum(r["truncated"] for r in res),
                                cap=CAPS[task])
                results[task] = summ
                print(f"[{label}] {task}: {json.dumps(summ)}", flush=True)
                json.dump(dict(label=label, gguf=gguf, results=results), open(f"{out_dir}/results.json", "w"), indent=1)
    finally:
        srv.terminate()
        srv.wait(timeout=60)
    print(f"[{label}] done in {time.time() - t_all:.0f} s", flush=True)


if __name__ == "__main__":
    main()
