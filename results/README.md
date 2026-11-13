# Results

| File | What it holds |
|---|---|
| `scorecard.md` | D against Instruct and Thinking on GSM8K, MATH, six BBH tasks, MMLU and IFEval (original token caps) |
| `long_budget_H5.md` | the same models and the routers on harder puzzles with a 32,768-token cap |
| `router_H3b.md` | the first router test (GSM8K, MATH, BBH, original caps) |
| `bf16_check.md`, `bf16_report_beta0.json`, `logits_report.json` | how far the shipped piece rebuilds the released weights from the released file |
| `refusal_check.md`, `refusal_draw<1 or 2>_<model>.counts.json` | refusal rates on AdvBench and XSTest for D and the two models it came from, two draws; the counts files hold counts only |
| `items/` | per-item results behind the first three files |

"H3b" and "H5" are the stage names in our run log: H3b is the first router test, H5 the long-budget retest.

## Folder names under `items/`

Each folder holds `results.json` (the run's summary) and one `items_<task>.jsonl` per task, one line per test item: id, correct or
not, extracted answer, tokens, whether the reply hit its cap, whether it wrote a think block. Long-budget folders carry no
reply text.

| Folder | Model and run |
|---|---|
| `instruct`, `instruct-native` | Instruct with the think-tag prompt, and in its own chat format |
| `thinking`, `thinking-seed1`, `thinking-seed2` | Thinking; the two seed folders repeat the BBH set with other seeds |
| `q80-neg`, `q80-neg-seed1`, `q80-neg-seed2` | D (the released model, evaluated as `q80-neg.gguf`; that file is byte-identical to the shipped `model/Qwen3-4B-Thinking-Concise-bf16.gguf`); seed folders repeat BBH |
| `interp75` | Blend 3/4: Instruct + 0.75 x (Thinking - Instruct) |
| `dial75-full`, `dial75-full-seed1`, `dial75-full-seed2` | the middle setting D + 0.75 P on the first test set |
| `c32-instruct`, `c32-thinking`, `c32-dial0` | long-budget runs (reply cap 32,768) of Instruct, Thinking and D |
| `c32-dial75`, `c32-dial75x` | the middle setting, on the questions the routers could send to it (`x` holds the extra questions the second router needed) |
| `*-s1` | a second run of the same model with a different seed offset |

In the refusal files, `q80neg` is D, `instruct` is Instruct and `thinking` is Thinking. The three evaluated files were
`q80-neg.gguf` (SHA-256 `492b0605e7a80616bb9434fe8954a9d549bb7cd7eccc3ddba8689f3ea061d432`, identical to the file in
`model/`, which carries the release name `Qwen3-4B-Thinking-Concise-bf16.gguf`), and our bf16 conversions of the two
Qwen checkpoints made with llama.cpp's `convert_hf_to_gguf.py`: `instruct-bf16.gguf`
(`19f69c461704c3678a70fef288fe51e617a337004bf957d8aea69b551230828a`) and `thinking-bf16.gguf`
(`3dcb408a1be4393cdcf4f0dc5a612fc1dfb41bfbacf226d66c54378716143fde`). The two conversions are not in this package.
