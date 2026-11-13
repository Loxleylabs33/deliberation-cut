# bf16 rounding check: Thinking minus the shipped piece against the released file

The released model was built from float32 factors of the piece. `piece/P_q80.safetensors` stores those factors rounded to
bf16, and the embedding difference rounded to bf16. So rebuilding D as Thinking minus the shipped piece (`piece/apply_piece.py
--beta 0`) gives weights that differ slightly from the released file. This page reports how much.

Run on 8 November 2026 on the CPU, with the released GGUF (SHA-256 `492b0605e7a80616...`) and the Qwen3-4B-Thinking-2507
checkpoint. The raw report is `bf16_report_beta0.json`. The comparison is tensor by tensor over all 398 tensors; all 398
were found in both. A second run gave identical counts.

## Weight by weight

| Tensors | Values | Bit-identical | One bf16 step off | Two or more steps off | Largest absolute difference | Distance to the released tensors | Distance as a share of the piece |
|---|---|---|---|---|---|---|---|
| 252 factored matrices | 3,633,315,840 | 93.88% | 5.23% | 0.89% | 0.00195 | 0.058% | 1.37% |
| 47 whole tensors | 389,032,576 | 96.88% | 2.05% | 1.07% | 0.000244 | 0.0064% | 0.072% |
| 99 untouched tensors | 119,680 | 100% | 0 | 0 | 0 | 0 | |
| All 398 | 4,022,468,096 | **94.17%** | 4.93% | 0.90% | 0.00195 | 0.051% | |

Counts: 3,787,950,979 values identical, 198,148,655 one step off, 36,368,462 two or more steps off.

- One bf16 step is the gap between neighbouring bf16 numbers (7 stored mantissa bits). 0.00195 = 2^-9 is one step for a weight
  between 0.25 and 0.5, so the largest differences are single-step flips of the largest weights.
- "Distance" is the Frobenius norm of the difference divided by the Frobenius norm of the released tensors; the last
  column divides by the Frobenius norm of the piece (A @ B, or the stored tensor) instead. For the factored matrices the
  rounding moves the piece by 1.4% of its norm.
- The 99 untouched tensors are rebuilt from the Qwen checkpoint alone and match the released file exactly. That confirms
  the base weights, the tensor-name mapping and the float32-to-bf16 cast.

## Behaviour on text

`piece/compare_logits.py` loads the Thinking model once and overwrites its weights in turn with the rebuilt ones and with
the released ones, then teacher-forces the same text through each. The text here was 12 of D's own replies from the
32,768-token runs: 6 GSM8K, 3 hard MATH and 3 Dyck-language puzzles, up to the first 300 tokens of each (2,514 scored
tokens). CPU, bf16. A is the released file, B the model rebuilt from Thinking and the shipped piece, C Thinking.

| Comparison | Mean KL per token (nats) | Median | Largest | Same most likely next token |
|---|---|---|---|---|
| A against B (released against rebuilt) | 0.0008 | 3e-11 | 0.45 | 99.64% (9 of 2,514 positions differ) |
| A against C (released against Thinking) | 0.64 | 0.00015 | 25.4 | 84.6% |

| Weights | Mean negative log-likelihood of the 12 replies (nats per token) | Total (nats) |
|---|---|---|
| A, released | 0.0682 | 171.4 |
| B, rebuilt | 0.0689 | 173.1 |
| C, Thinking | 0.718 | 1,804.7 |

Rebuilding from the shipped piece moves the next-token distribution about 800 times less (by mean KL) than the piece
itself does, and raises the negative log-likelihood of D's own text by 1.0%. Twelve texts of maths and logic is a small
sample; treat this as an order of magnitude, not a bound. The raw numbers are in `logits_report.json`.

## How to repeat it

```
python3 piece/apply_piece.py --thinking-dir DIR --piece piece/P_q80.safetensors --beta 0 \
    --compare-gguf model/Qwen3-4B-Thinking-Concise-bf16.gguf --report bf16_report_beta0.json
python3 piece/compare_logits.py --thinking-dir DIR --piece piece/P_q80.safetensors \
    --gguf model/Qwen3-4B-Thinking-Concise-bf16.gguf --texts YOUR_TEXTS.jsonl
```
