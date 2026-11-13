# Scripts

The research scripts behind the tables in `../results/`, with machine-specific paths removed. Every path comes from
environment variables (`cutconfig.py`), and device choice is left to `CUDA_VISIBLE_DEVICES`. The logic is the logic that
produced the reported numbers, with five kinds of change: paths and the llama-server launch go through `cutconfig.py`;
`torch.load` calls on the shipped files pass `weights_only=True`; three inputs became optional (the earlier-caps pool in
`h5_router.py`, the list of ids answered in the first long-budget run in `h5_route.py`, and a CPU fall-back for the
embedding scripts); `build_interp.py` computes the raw difference itself instead of reading a saved copy; and comments
that recorded our working sessions were rewritten.

## What is here

| File | What it does | Needs | Checked while packaging |
|---|---|---|---|
| `cutconfig.py` | reads `CUT_ROOT`, `THINKING_DIR`, `INSTRUCT_DIR`, `LLAMA_SERVER`, `CONVERT_SCRIPT` | | imported by everything |
| `step4_eval.py` | the evaluation harness: starts llama-server on a GGUF, runs GSM8K, MATH, BBH, MMLU, IFEval and the long-budget sets, writes `step4/<label>/` | llama-server (a GPU makes it practical), the datasets | run end to end against a stand-in server (all task types, the long-budget options) |
| `h2_pool.py` | answers the 440-question router pool at one setting, keeps right/wrong and length | as above | run against the stand-in server |
| `h4_pool_text.py` | the same, keeping the reply text (needed to train the dial) | as above | run against the stand-in server |
| `h5_pool32.py` | redoes the pool answers that hit a cap at 32,768 tokens | as above | run against the stand-in server, including the redo branch |
| `h3_embed.py` | writes `hyper/embed.pt`: layer-18 mean hidden state of Qwen3-4B-Instruct-2507 for every pool and test question | Instruct folder, the datasets | run on the CPU (`EMBED_DEVICE=cpu`) for 15 of the 880 questions: every vector matched `router/embed.pt` to a cosine of at least 0.99995; the full set would take about 80 minutes on that CPU |
| `h3_train_eval.py` | router v1 (the setting with the best result per question as a class label) and the shared readers `read_pool()` and `test_results()` | `hyper/embed.pt`, pool labels, per-item results | run end to end on the shipped files |
| `h3b_value_router.py` | router v2 ("value router"): one logistic model per setting; writes `hyper/router_v2.pt` and the first test table | same | run end to end on the shipped files |
| `h5_router.py` | router v2 retrained on the 32k pool; writes `hyper/h5/router_v2.pt` | `hyper/embed.pt`, `hyper/pool_text32/` | run end to end on the shipped files |
| `build_interp.py` | plain blends Instruct + alpha x (Thinking - Instruct) as bf16 GGUF | both Qwen folders, the llama.cpp converter | `blend()` reproduces five tensors of the blend GGUF built earlier from the saved difference bit for bit, embedding table included; the GGUF conversion step was not run |
| `h4_compose.py` | the Thinking model with a separate strength of P in each of the 36 layers and the embedding (hooks) | Thinking folder, `P_q80.safetensors`, a GPU for training | built on the CPU with 252 matrix hooks, 46 norm hooks and the embedding hook; with every strength at 0 it matched weights rebuilt by `apply_piece.py` on a 62-token text (mean KL 0.005 nats, same top token at 60 of 62 positions) |
| `h4_train.py` | trains the per-layer dial network (a 64-128-37 network on the question vector) against the model's own correct answers | a GPU, `hyper/pool_text/` or `pool_text32/` with reply text | imports only |
| `h5_route.py` | routes the long-budget questions with the 32k router and the trained dial; writes profiles and routes | `hyper/h5/hypernet.pt` from `h4_train.py` | run on the CPU with a random, untrained dial network plus the shipped router and question vectors: it wrote the profiles, routes and id lists |
| `h5_build.py` | builds a bf16 GGUF for each dial profile from Thinking and P | `hyper/h5/profiles.json` from `h5_route.py`, the converter | imports only |
| `setup_workdir.py` | lays out the files of this release the way the scripts expect | | run |
| `fetch_data.sh`, `setup_ifeval.sh`, `ifeval_patch.diff`, `DATA_SHA256.txt` | fetch the public test sets; build the IFEval checker; hashes of the copies behind the reported numbers | network (fetch only), `patch` | `setup_ifeval.sh` run; `fetch_data.sh` not run (no downloads while packaging) |

"Run against the stand-in server" means a small script that answers `/health` and `/completion` with canned text, used to
exercise the harness without a model; the accuracies it produces are meaningless. None of the scripts here that call
llama-server was run against a real model while packaging this release; the numbers in `../results/` come from the
original runs. The routing, piece and comparison scripts (`../router/`, `../piece/`) were run on the CPU with the real
checkpoints.

## Requirements

Tested with Python 3.12.3, torch 2.6.0 (CUDA 12.4 build; the CPU scripts do not use CUDA), transformers 5.19.0,
safetensors 0.8.0, numpy 2.5.3, gguf 0.19.0, and llama.cpp build 11669 (commit 620d2225c) for every GGUF run. IFEval's
checker also imports nltk 3.10.4, langdetect 1.0.9, absl-py 2.5.1 and immutabledict 4.3.1. `requirements.txt` pins the
Python packages. The reported runs used one RTX 3090 (24 GB): bf16 weights, 8 requests in parallel with a 16-bit KV
cache, and 5 requests in parallel with an 8-bit KV cache for the 32,768-token runs.

## Setting up

```
export CUT_ROOT=$PWD/work
python3 code/setup_workdir.py $CUT_ROOT       # copies the small files, links the piece
bash code/fetch_data.sh                       # public datasets into $CUT_ROOT/data   (needs a network)
bash code/setup_ifeval.sh                     # the IFEval checker with the two patches
# put Qwen3-4B-Thinking-2507 and Qwen3-4B-Instruct-2507 (Hugging Face format) in $CUT_ROOT, or set THINKING_DIR / INSTRUCT_DIR
export LLAMA_SERVER=/path/to/llama-server     # plus CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH as your build needs
```

Layout under `$CUT_ROOT` after `setup_workdir.py`: `hyper/` (embeddings, routers, pool labels, the piece), `step4/` (per-item
results), `data/`, `models/` and `logs/` (empty), and `DATA_SHA256.txt`. Run `cd $CUT_ROOT/data && sha256sum -c
../DATA_SHA256.txt` after fetching; a mismatch means upstream changed and your numbers may differ from ours.

## Recomputing the shipped tables (CPU only)

From the package root, with `CUT_ROOT` set and `setup_workdir.py` run as above:

```
python3 -I router/score_router.py router/router_pool440_h3b/router_v2_pool440.pt     # results/router_H3b.md
python3 -I router/score_long_budget.py                                               # router rows of results/long_budget_H5.md
python3 -I code/h3b_value_router.py                                                  # retrains router v2 (not bit-identical, see router/README.md)
python3 -I code/h5_router.py                                                         # retrains the 32k-pool router
```

## Running the whole chain (needs a GPU)

1. Models. `piece/apply_piece.py --beta 0` rebuilds D (or use the GGUF in `model/`); `--beta 0.75` gives the middle
   setting; the Thinking and Instruct GGUFs come from llama.cpp's `convert_hf_to_gguf.py --outtype bf16` on the Qwen
   folders. Put the GGUFs in `$CUT_ROOT/models/`.
2. Scorecard: `python3 -I code/step4_eval.py LABEL models/FILE.gguf --tasks gsm8k,math,bbh,mmlu,ifeval`. Long budget:
   `... --tasks mathhard,bbhhard,math,gsm8k --cap 32768 --slots 5 --kvq8 --keep-text`. A second seed: `--seed-offset N`.
   Only some items: `--only ids.json`.
3. Router pool: `h4_pool_text.py LABEL GGUF` once per setting (`dial0`, `dial75`, `dial100`; it keeps the reply text
   that `h5_pool32.py` and `h4_train.py` need; `h2_pool.py` writes labels only), then `h5_pool32.py LABEL GGUF` once
   per setting for the 32k labels.
4. Vectors and routers: `h3_embed.py`, then `h3b_value_router.py` and `h5_router.py`.
5. Dial (optional): `H4_OUT=$CUT_ROOT/hyper/h5 H4_POOL_TEXT=$CUT_ROOT/hyper/pool_text32 python3 -I code/h4_train.py
   --epochs 3 --mu 0.02 --maxlen 24576`, then `h5_route.py`, then `h5_build.py`.

## The IFEval checker

The reported IFEval scores use Google's checker with two edits, both to avoid NLTK's downloaded tokenizer data:
sentences are split on `.`, `!` or `?` followed by whitespace, and the all-capitals check splits words with a regular
expression instead of `word_tokenize`. `ifeval_patch.diff` holds exactly those edits against the files
`fetch_data.sh` downloads (hashes in `DATA_SHA256.txt`); `setup_ifeval.sh` applies it. Scores computed with NLTK's
own tokenizers can differ by a point or two.

## Not in this package

- The scripts that selected the piece: the shared-start check, the raw difference, the gradient and direction scoring
  (SVD of the difference, first-order scores), and the builder that turned a selection into weights. They read tens of
  gigabytes of intermediate files that are not shipped, so they could not be tested here. `../piece/README.md` describes
  the method.
- The weights and profiles of the per-layer dial network, and the scoring scripts for the dial rows of the long-budget
  table.
- The Qwen models in Hugging Face format or as GGUF files, and every dataset. They are public; `../README.md` names them
  and gives their licences.
