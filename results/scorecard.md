# Scorecard: the released model against the two models it came from

Runs of 30 October 2026. Each model was served as a bf16 GGUF by llama.cpp (build b11669) on one RTX 3090, 8 requests
in parallel. One draw per test item, with a fixed seed per item.

Names used below. **Instruct** is Qwen/Qwen3-4B-Instruct-2507. **Thinking** is Qwen/Qwen3-4B-Thinking-2507. **D** is the
released model, Thinking minus the q80 piece P. **Blend 3/4** is Instruct + 0.75 x (Thinking - Instruct), every tensor,
no selection (built with `code/build_interp.py 75`).

## Test sets

These are subsets. None of them is a full benchmark run, and they differ in size by a factor of ten, so read the accuracy
columns together with the standard errors in the last section. Every model in the tables ran on the same items under
the same settings. One setting changed during the work: a first Thinking run with a 4,096-token cap hit the cap on 54%
of 100 MATH replies and 32% of 180 BBH replies, so before the final runs we raised the MATH and BBH caps to 8,192 and cut
those two sets to 50 and 90 items; Thinking was rerun under the final settings and the tables use only final-setting runs.
The six BBH tasks were in the harness from the first scorecard run and never changed. The cut used a fixed rule (the first fifteen examples of each task;
every tenth MATH-500 problem) but was made after seeing the first Thinking run, so it was not blind. Dyck languages is not
among the six; it appears only in the harder 32,768-token set (`long_budget_H5.md`).

| Set | Items | How the items were picked |
|---|---|---|
| GSM8K | 300 | first 300 problems of the test file |
| MATH | 50 | every 10th problem of MATH-500 |
| BIG-Bench Hard, six tasks | 90 | first 15 examples of boolean_expressions, formal_fallacies, logical_deduction_five_objects, web_of_lies, tracking_shuffled_objects_three_objects, date_understanding |
| MMLU | 500 | 500 test questions, shuffled with seed 42 across all subjects |
| IFEval | 200 | first 200 prompts, strict prompt-level accuracy (every instruction in the prompt followed) |

## Prompts and settings

No chat template was applied. Apart from the last table row, every model got the same raw text.

- GSM8K, MATH, BBH: `<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n`, temperature 0.6,
  top_p 0.95, top_k 20 (the Thinking model card's settings). Reply cap 4,096 tokens for GSM8K and 8,192 for MATH and BBH.
  The question ends with an instruction to put the final answer in `\boxed{}` (BBH: the option letter or exact answer).
- MMLU: the same prompt with an empty think block (`<think>\n\n</think>\n\n`), then `The answer is (`; the answer is the
  most likely of A to D for the next token.
- IFEval: empty think block, temperature 0.7, top_p 0.8, top_k 20, reply cap 1,536 tokens, scored with Google's checker
  (patched as described in `code/README.md`).
- "Instruct, its own chat format" drops the `<think>` line and uses `<|im_start|>assistant\n` alone, which is how
  Instruct was trained. The think-tag prompt costs Instruct about 9.5 points on IFEval.

## Accuracy (%)

| Model | GSM8K (300) | MATH (50) | BBH (90) | MMLU (500) | IFEval (200) |
|---|---|---|---|---|---|
| Instruct | 96.0 | 86 | 75.6 | 69.2 | 73.0 |
| Thinking | 93.0 | 64 | 95.6 | 64.8 | 66.0 |
| **D, the released model** | 93.7 | 86 | 90.0 | 66.8 | 69.5 |
| Blend 3/4 | 96.7 | 72 | 86.7 | 66.8 | 73.0 |
| Instruct, its own chat format | 94.7 | 92 | 75.6 | 68.8 | 82.5 |

## Length and think blocks

Median tokens per reply (replies that reached the cap, out of the set size), and the number of replies with a think
block, meaning a closed `<think>` section of 40 words or more. Thinking's counts come from the first version of the
harness, which also counted a reply that ran into the cap; Thinking opened every reply with deliberation.

| Model | GSM8K median (capped of 300) | MATH median (capped of 50) | BBH median (capped of 90) | Think blocks, GSM8K / MATH / BBH |
|---|---|---|---|---|
| Instruct | 281 (2) | 788 (4) | 530 (0) | 0 / 0 / 0 |
| Thinking | 1,069 (15) | 5,178 (15) | 2,868 (0) | 300 / 50 / 90 |
| **D** | 204 (0) | 637 (2) | 362 (0) | 0 / 0 / 0 |
| Blend 3/4 | 944 (6) | 3,711 (12) | 2,017 (0) | 297 / 40 / 90 |

## Reading the table

On the six BBH tasks D scores 90.0 against Instruct's 75.6 and Thinking's 95.6, which is 14.4 of the 20-point gap, with a
median reply of 362 tokens against Thinking's 2,868. On MATH at an 8,192-token cap D matches Instruct (86) and beats
Thinking (64), but Thinking's 64 is a budget effect: 15 of its 50 replies hit the cap, and with a 32,768-token cap
Thinking scores 91 on the same 50 problems (a two-run average, `long_budget_H5.md`). Instruct in its own format scores 92 on MATH and 82.5
on IFEval, both above D.

On MMLU D scores 66.8, between Thinking (64.8) and Instruct (69.2). On IFEval D scores 69.5, between Thinking (66.0) and
Instruct (73.0 with the think-tag prompt, 82.5 in its own format).

## Standard errors

For an accuracy p on n items the standard error is sqrt(p(1-p)/n): about 1.4 points for GSM8K, 4.9 for MATH, 3.2 for BBH,
2.1 for MMLU and 3.3 for IFEval. The standard error of a gap between two models is roughly 1.4 times that (independent
samples), so a 5-point gap on BBH or a 10-point gap on MATH is inside two standard errors. The 14.4-point BBH gap between
D and Instruct is about 3.2 standard errors.

Per-item results for every row are in `items/` (one folder per run, `items_<task>.jsonl`). `items/thinking` also holds
`items.jsonl` from the first harness version, which `code/h3_train_eval.py` reads for Thinking's GSM8K.
