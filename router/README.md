# Routers: choosing how much of P to keep, per question

The piece P (see `../piece/`) can be added back to the released model D at any strength. A router reads a question and
picks one of three settings:

| Index in the code | Strength of P (beta) | Weights | What it is |
|---|---|---|---|
| 0 | 0 | D | the released model; it wrote no closed think block in any reply we scored where it was free to |
| 1 | 0.75 | D + 0.75 P (Thinking minus 0.25 P) | thinks on most questions and stops on its own |
| 2 | 1 | D + P = Thinking | Qwen/Qwen3-4B-Thinking-2507 |

The result tables name the settings by beta (0, 0.75, 1); the scripts and the router files count them 0, 1, 2.

A router holds about 170,000 numbers (a 2,560-number mean, the same for the spread, a 2,560 x 64 projection, three
65-number logistic models). It does not run the three models. It predicts, from a vector describing the question, how
likely each setting is to answer correctly, and it picks the setting with the best predicted probability after a penalty
for length.

## Two routers

They share a file format and a training method and differ in one thing: the token cap under which their training answers
were produced. Both file names carry the pool, so they cannot be confused.

| Folder | File | SHA-256 (first 16) | Trained on | Matches |
|---|---|---|---|---|
| `router_pool440_h3b/` | `router_v2_pool440.pt` | `7cb1e22d6ae947bc` | 440 questions, each answered at the three settings under the original caps (GSM8K 4,096 tokens, MATH and BBH 8,192) | `../results/router_H3b.md`, and the "old pool" rows of `../results/long_budget_H5.md` |
| `router_pool32k_h5/` | `router_v2_pool32k.pt` | `6cd3a4a5c94ba08f` | the same 440 questions, with the 53 answers that hit a cap (3 of D's, 16 of the 0.75 setting's, 34 of Thinking's) redone at a 32,768-token cap | the "32k pool" rows of `../results/long_budget_H5.md` |

The first router learned from answers in which Thinking often ran out of room, so it learned that thinking does not pay
on maths (see the long-budget table). The second is the better of the two on the long-budget test and is the one to
start from. Both are kept because the first one produced the first table.

## Files

| Path | Contents |
|---|---|
| `embed.pt` | 880 question vectors: the 440 pool questions (ids start with `pool-`) and the 440 questions of the first test set |
| `embed_long_budget.pt` | 646 vectors for the long-budget test questions (hard maths 96, hard logic 200, MATH 50, GSM8K 300) |
| `pool_labels/pool440/`, `pool_labels/pool32k/` | for each pool question and each setting (`dial0.jsonl` = D, `dial75.jsonl` = 0.75, `dial100.jsonl` = Thinking): correct or not, tokens, whether the reply hit its cap. The fields the training scripts read; no prompts, no replies |
| `route_question.py` | choose a setting for a question you type |
| `score_router.py` | recompute the first test table from a saved router |
| `score_long_budget.py` | recompute the long-budget rows for the originals, D and both routers |

The `.pt` files hold only tensors, dictionaries, lists, strings and numbers. They load with `torch.load(path,
weights_only=True)`, which is how every script here loads them; use PyTorch 2.6 or later, where that is the default.

## How a question is read

The question is wrapped in a user turn, `<|im_start|>user\n{message}<|im_end|>\n`, and run through the first 18 decoder
layers of Qwen/Qwen3-4B-Instruct-2507. The layer-18 hidden states are averaged over the message's tokens, giving one
2,560-number vector. The router standardises it with a stored mean and standard deviation, projects it onto 64 stored
directions (a PCA fitted on the pool), and applies three logistic models, one per setting, to get p(correct) for each.
The choice is `argmax_k  p_k - lambda * mean_tokens_k / 1000`, where `mean_tokens_k` is the average reply length of
setting k on the training pool, stored in the router file, and lambda is the price of 1,000 tokens in probability.
Lambda 0 is "accuracy first"; 0.01 is "balanced" (a thousand tokens are worth one point).

The vectors were made from the benchmark prompts themselves, which end with an instruction to put the answer in
`\boxed{}`. `route_question.py --wrap math` (the default) and `--wrap logic` add the same endings to your question.
The pool covers three families of question: GSM8K, MATH and six BIG-Bench Hard logic tasks. How the routers behave on
other kinds of question has not been tested.

## Using a router

```
python3 -I route_question.py --router router_pool32k_h5/router_v2_pool32k.pt \
    --instruct-dir /path/to/Qwen3-4B-Instruct-2507 --lambda 0.01 \
    "A shop sells pens at 3 for 2 dollars. How much do 12 pens cost?"
```

```
A shop sells pens at 3 for 2 dollars. How much do 12 pens cost?
  chosen setting: 0 (D, the released model)  (lambda 0.01)
    0 (D, the released model)    p(correct) 0.994   mean tokens in the training pool 565
    1 (D + 0.75 P)               p(correct) 0.958   mean tokens in the training pool 2299
    2 (Thinking)                 p(correct) 0.996   mean tokens in the training pool 3647
```

It prints the chosen setting and the three predicted probabilities. It runs on the CPU in bfloat16 and stops after layer
18, a few seconds per short question once the model is loaded. It chooses; it does not switch models. Index 0 is the GGUF
in `../model/`, index 1 is `../piece/apply_piece.py --beta 0.75` followed by a GGUF conversion, index 2 is the Qwen
checkpoint. A server that holds the three variants (or D plus P as a run-time low-rank update) and switches per request
is not part of this package.

## What we checked in this package

- `score_router.py` on the first router reproduces every number in the first test table (`../results/router_H3b.md`),
  including the per-question-type choices.
- `score_long_budget.py` reproduces the long-budget rows for the originals, D and the 32k-pool router exactly. For the
  first router one cell moves from 2,556 to 2,549 mean tokens (hard logic): one of 646 routing decisions differs at each
  lambda (see `../results/long_budget_H5.md`).
- The 32k-pool router, applied to the stored question vectors, makes the same choice as the saved routes of the original
  run on all 646 long-budget questions, at lambda 0 and at 0.01.
- `route_question.py` on the CPU reproduces stored question vectors to a cosine of at least 0.99995 (relative L2 distance
  0.4% to 3.5%, six questions). The CPU and the GPU round differently, so a question close to a tie between settings can
  go the other way: in that six-question check 23 of 24 decisions (two routers, two lambdas) were the same.
- `code/h3_embed.py` run on the CPU for 15 of the 880 questions reproduced the stored vectors to a cosine of at least
  0.99995; the ids and task labels matched.
- `h3b_value_router.py` and `h5_router.py` run end to end on the labels in `pool_labels/`.

## Retraining gives a different router

`code/h3b_value_router.py` and `code/h5_router.py` rebuild a router from `pool_labels/` and `embed.pt`. The PCA step
(`torch.pca_lowrank`) starts from random numbers and the script sets no seed, so a retrained router is a little different
every time. We retrained both once from the files in this package and compared with the shipped routers on the 646
long-budget questions: the 32k-pool retrain made the same choice on 559 questions at lambda 0 (86.5%) and 558 at lambda
0.01; the first-router retrain on 458 (70.9%) and 559 (86.5%). Mean absolute difference in predicted probability was
0.009 to 0.028. The three settings often have nearly equal predicted probabilities, so a small change flips the choice.
The first test table, rebuilt this way, moved by at most 2 points per column (lambda 0.01: 94.0 / 84.0 / 94.1 against
94.0 / 82.0 / 94.8). The published tables belong to the shipped files.

## Limits

- Three families of question, 440 training questions, one embedding model. The pool has no code, no multi-turn chat and
  nothing in other languages.
- Per-question decisions are unstable (see above). What the tables show is the average effect over a few hundred
  questions.
- Test items are single or double runs of fixed seeds, and the routed rows are scored by lookup, so a live system that
  switches weights could differ in length and in accuracy.
- The long-budget table also lists a per-layer "dial" that sets a separate strength of P for each layer. Its weights are
  not included.
