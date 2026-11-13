# The removed piece, P_q80

`P_q80.safetensors` (2,173,615,960 bytes) is what was subtracted from Qwen/Qwen3-4B-Thinking-2507 to make the released
model D. With it you can rebuild D from the original Qwen checkpoint, or add part of the piece back to get the middle
setting the routers choose between. Full SHA-256 values are in `../SHA256SUMS`.

## Contents

551 tensors, 1,086,699,648 stored values.

- **252 weight matrices as factor pairs.** Every attention projection (q, k, v, o) and MLP projection (gate, up, down) of
  the 36 layers has two bf16 tensors, `<name>.A` (out x r) and `<name>.B` (r x in); the piece for that matrix is `A @ B`.
  The rank r differs by matrix, from 34 to 986 (median 248, mean 293.1). The ranks add up to 73,871 directions.
- **47 tensors stored whole.** The embedding table `model.embed_tokens.weight` (bf16, 151,936 x 2,560; Qwen3-4B ties the
  output head to it) and 46 normalization vectors (float32): `input_layernorm`, `post_attention_layernorm`, `q_norm` and
  `k_norm` vectors of various layers. The other 99 tensors, including the final norm, are not in the file.

A direction is one singular direction of the difference R = Thinking - Instruct for a matrix: R = sum_k s_k u_k v_k^T.
For the chosen k the file keeps A = [s_k u_k] and B = [v_k^T], so a matrix's piece is the sum of the chosen terms.

## How the directions were chosen

Qwen3-4B-Instruct-2507 and Qwen3-4B-Thinking-2507 appear to share a starting checkpoint; we infer it from Qwen's naming of
the pair and from the weights: matching tensors have a median cosine similarity of 0.998 (lowest 0.994, the key
projection of layer 33), and R is about 8% of each matrix by norm (median
8.1%, largest 11.2%). R is close to full rank (median effective rank 1,381 per matrix), so size alone does not pick out a
small part of it.

Each singular direction of each matrix of R got a score: a first-order estimate of how much the next-token loss on
Thinking's own correct solutions would rise if that direction were removed from the Thinking model,
`score = - s_k u_k^T G v_k`, with G the gradient of that loss with respect to the matrix. The solutions were 256 of
Thinking's correct answers to 400 GSM8K training problems (383 were correct). The embedding table and each norm vector
were scored as single units. For the whole difference the estimated rise was 0.691 against 0.618 measured, an error of
12%: good enough to rank directions, not to add them up. Directions with positive scores were taken in order until they
covered 80% of the total positive score ("q80"): 73,871 matrix directions and 47 whole tensors. The positive scores sum
to 3.05 while the net effect of removing all of R is 0.69, so many directions lower the loss when removed; the selection
ignores them.

What we found along the way, all on the tests described in `../results/scorecard.md`:

- Adding the same directions to Instruct does not work. Instruct plus the chosen directions deliberates fluently and
  does not stop: 280 of its first 300 GSM8K replies ran to the 4,096-token cap. Adding directions chosen for the
  decision to close the think block gave models that stop and then give no answer. Scores computed for one direction
  at a time do not add up.
- Plain blends do work. Instruct + alpha x R switches thinking on between alpha 0.25 and 0.75; at 0.75 it thought and
  closed the think block on every reply of the screening run (`scorecard.md`, "Blend 3/4", has the full run).
- Subtracting the chosen directions from Thinking works, within limits: D wrote no think block in the 2,380 replies we scored where it was free to, and keeps most of the
  gain on the six BBH logic tasks, but little of it on the harder puzzles of the long-budget test
  (`../results/long_budget_H5.md`).

## Using it

```
W(beta) = Thinking + (beta - 1) * P
```

`beta = 0` gives D, `beta = 1` gives Thinking, and `beta = 0.75` gives the middle setting. Arithmetic is float32 with a
bf16 result, the recipe the released file was built with.

```
python3 apply_piece.py --thinking-dir /path/to/Qwen3-4B-Thinking-2507 --piece P_q80.safetensors --beta 0 --out ./D
python3 /path/to/llama.cpp/convert_hf_to_gguf.py ./D --outtype bf16 --outfile D-bf16.gguf
```

`--out` writes `model.safetensors` (bf16) plus the config and tokenizer files copied from the Thinking folder.
`apply_piece.py` needs torch and safetensors. Peak memory was 18.8 GB for `--out` (about 20 seconds on 8 CPU threads
with the files cached) and 19.7 GB for `--compare-gguf` (about 85 seconds).

## The released file and the shipped piece differ by bf16 rounding

D was built from float32 factors. `P_q80.safetensors` holds those factors rounded to bf16 (and the embedding difference
rounded to bf16), so Thinking minus the shipped piece reproduces the released weights only to within that rounding. We
measured the gap with `apply_piece.py --beta 0 --compare-gguf` on all 398 tensors of the released file (4,022,468,096
values), on the CPU:

| Tensors | Values | Bit-identical | One bf16 step off | Two or more steps off | Largest absolute difference | Distance to the released tensors | Distance as a share of the piece |
|---|---|---|---|---|---|---|---|
| 252 factored matrices | 3,633,315,840 | 93.88% | 5.23% | 0.89% | 0.00195 | 0.058% | 1.37% |
| 47 whole tensors | 389,032,576 | 96.88% | 2.05% | 1.07% | 0.000244 | 0.0064% | 0.072% |
| 99 untouched tensors | 119,680 | 100% | 0 | 0 | 0 | 0 | |
| All 398 | 4,022,468,096 | **94.17%** | 4.93% | 0.90% | 0.00195 | 0.051% | |

"One bf16 step" is the gap between neighbouring bf16 numbers; 0.00195 is one step for a weight between 0.25 and 0.5.
"Distance" is the Frobenius norm of the difference divided by the Frobenius norm of the released tensors (or of the
piece, in the last column). The 99 untouched tensors, rebuilt from the Qwen checkpoint alone, match the released file
exactly, which confirms the base weights and the name mapping.

**On text.** We scored 12 of D's own replies (6 GSM8K, 3 hard MATH, 3 Dyck puzzles; up to the first 300 tokens of each,
2,514 tokens in all) with three weight sets on the CPU: the released file, the rebuilt model, and Thinking. For the
rebuilt model, the next-token distribution differs from the released file's by a mean KL of 0.0008 nats per token (median
3e-11, largest 0.45), and the most likely next token is the same at 99.64% of positions (9 of 2,514 differ). Thinking
differs from the released file by a mean KL of 0.64 nats and agrees on the top token at 84.6%. The replies cost 0.0682
nats per token under the released file, 0.0689 under the rebuilt weights (+1.0%) and 0.718 under Thinking. By mean KL the
rounding moves the model about 800 times less than the piece itself does. The sample is small and covers only maths and
logic text; `../results/logits_report.json` has the raw numbers.

The released file is the exact model we evaluated, and it is in this package. If you rebuild D with `apply_piece.py` you
are testing a model about 0.05% away from it in weight space; any number you measure on the rebuild is a measurement of
the rebuild.

## Verifying your copy

```
sha256sum P_q80.safetensors     # 64798af8568023ce41edef0856b3f7fc125f088a4c5aee4788dc80c135f0cad3
python3 apply_piece.py --thinking-dir DIR --piece P_q80.safetensors --beta 0 --compare-gguf ../model/Qwen3-4B-Thinking-Concise-bf16.gguf
python3 compare_logits.py --thinking-dir DIR --piece P_q80.safetensors --gguf ../model/Qwen3-4B-Thinking-Concise-bf16.gguf --texts sample_texts.jsonl
```

`compare_logits.py` loads the Thinking model once and overwrites its weights in turn with the rebuilt ones and with the
released ones, so the model needs about 8 GB plus the log-probabilities (150,000 floats per scored token per weight set);
the 12 texts of `results/bf16_check.md` peaked at about 12 GB and took 14 minutes on 8 CPU threads.
`sample_texts.jsonl` holds three short hand-written examples; pass your own `{"prompt", "reply"}` lines for a real test.

## Not provided

The piece is not packaged as a LoRA adapter and no conversion is provided. The factors have the right shape for one (rank
34 to 986 per matrix, scale 1), but we have not tried it.
