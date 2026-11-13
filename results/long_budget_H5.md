# Long-budget test: harder puzzles, reply cap 32,768 tokens

This is the second router test ("H5" in our run log). The tables below are the output of our scoring script, unedited
except for one omitted line (which per-layer profiles the dial used for each task) and two sign-test rows added
afterwards (see the note under that table). The notes after them say which rows you can recompute from this package and
which you cannot.

Settings: reply cap 32,768 tokens (the Qwen3-4B-Thinking-2507 card's setting for most queries), KV cache at 8 bits, 5
requests in parallel, sampling as in `scorecard.md`. Cell format: accuracy % (runs per question) / mean tokens per reply.
A question's correctness is averaged over its runs first and then over questions. The hard maths, hard logic and MATH
rows average two runs per question; the GSM8K row is one run.

| Set | Items | Contents |
|---|---|---|
| hard maths | 96 | the level-5 MATH-500 problems that are in neither the 50-problem test set nor the router pool |
| hard logic | 200 | examples 0 to 49 of logical_deduction_seven_objects, tracking_shuffled_objects_seven_objects, multistep_arithmetic_two, dyck_languages |
| MATH test | 50 | the same 50 problems as in `scorecard.md` |
| GSM8K | 300 | the same 300 problems as in `scorecard.md` |

Policies. Instruct, Thinking and D run every question on one model. "router v2, old pool" is the router in
`router/router_pool440_h3b/` and "router v2, 32k pool" is the one in `router/router_pool32k_h5/`; "accuracy first" picks
the setting with the highest predicted probability of a correct answer, "balanced" subtracts 0.01 x mean tokens / 1000
first. "dial" is a small network that sets a separate scale for P in each of the 36 layers and the embedding, for each
question (H4b trained on the old pool, H5b on the 32k pool); its weights are not part of this package.

| policy | hard maths (96) | hard logic (200) | MATH test (50) | GSM8K (300) |
|---|---|---|---|---|
| Instruct (original) | 85.4 (2) / 3,004 | 77.8 (2) / 1,825 | 88.0 (2) / 2,388 | 95.7 (1) / 372 |
| Thinking (original) | 90.1 (2) / 10,830 | 92.5 (2) / 4,705 | 91.0 (2) / 7,839 | 96.0 (1) / 1,471 |
| D (the cut, never thinks) | 79.2 (2) / 2,103 | 80.5 (2) / 939 | 85.0 (2) / 1,439 | 93.3 (1) / 232 |
| router v2, old pool, accuracy first | 79.7 (2) / 3,171 | 85.0 (2) / 2,556 | 88.0 (2) / 2,174 | 95.7 (1) / 764 |
| router v2, 32k pool, accuracy first | 89.6 (2) / 7,421 | 90.0 (2) / 3,263 | 95.0 (2) / 4,769 | 95.3 (1) / 970 |
| router v2, 32k pool, balanced | 86.5 (2) / 5,979 | 90.0 (2) / 2,757 | 94.0 (2) / 4,304 | 94.0 (1) / 619 |
| dial, old pool (H4b) | 80.2 (2) / 2,099 | 88.0 (2) / 2,141 | 88.0 (2) / 1,470 | 94.0 (1) / 436 |
| dial, 32k pool (H5b) | 90.6 (2) / 7,212 | 86.5 (2) / 2,447 | 93.0 (2) / 4,559 | 94.7 (1) / 486 |

Question by question: A better / B better (averaged over runs), exact two-sided sign test p. Questions both policies
treat alike drop out. A question counts for one side when its two-run average is higher (for GSM8K, its single run), so
a question one run got right and one wrong contributes half a point to the accuracy gap: 12 to 3 on 96 questions is a
6.2-point gap (85.4 against 79.2), not a 9.4-point one.

| A vs B | hard maths (96) | hard logic (200) | MATH test (50) | GSM8K (300) |
|---|---|---|---|---|
| dial, 32k pool (H5b) vs dial, old pool (H4b) | 16 / 1 (p 0.000) | 9 / 16 (p 0.230) | 7 / 2 (p 0.180) | 6 / 4 (p 0.754) |
| dial, 32k pool (H5b) vs Thinking (original) | 1 / 1 (p 1.000) | 1 / 20 (p 0.000) | 3 / 3 (p 1.000) | 4 / 8 (p 0.388) |
| dial, 32k pool (H5b) vs D (the cut, never thinks) | 17 / 2 (p 0.001) | 28 / 7 (p 0.001) | 6 / 1 (p 0.125) | 9 / 5 (p 0.424) |
| dial, 32k pool (H5b) vs router v2, 32k pool, accuracy first | 2 / 0 (p 0.500) | 7 / 19 (p 0.029) | 0 / 2 (p 0.500) | 5 / 7 (p 0.774) |
| router v2, 32k pool, accuracy first vs router v2, old pool, accuracy first | 15 / 2 (p 0.002) | 19 / 2 (p 0.000) | 6 / 0 (p 0.031) | 3 / 4 (p 1.000) |
| router v2, 32k pool, accuracy first vs Thinking (original) | 1 / 1 (p 1.000) | 2 / 10 (p 0.039) | 3 / 1 (p 0.625) | 3 / 5 (p 0.727) |
| Thinking (original) vs D (the cut, never thinks) | 16 / 2 (p 0.001) | 39 / 3 (p 0.000) | 7 / 2 (p 0.180) | 12 / 4 (p 0.077) |
| router v2, 32k pool, accuracy first vs D (the cut, never thinks) | 16 / 2 (p 0.001) | 32 / 3 (p 0.000) | 7 / 0 (p 0.016) | 9 / 3 (p 0.146) |
| Instruct (original) vs D (the cut, never thinks) | 12 / 3 (p 0.035) | 15 / 23 (p 0.256) | 4 / 2 (p 0.688) | 11 / 4 (p 0.118) |

The last two rows were not in the table of the original run. Both come from `router/score_long_budget.py` in this
release: Instruct against D because the paper's summary cites it (hard maths, 12 questions to 3), and the 32k-pool router
against D because the script prints it. Every other row and number on this page is unchanged.

Hard logic by puzzle type (accuracy %, averaged over runs, 50 questions each):

- Instruct (original): dyck_languages 17, logical_deduction_seven_objects 97, multistep_arithmetic_two 100, tracking_shuffled_objects_seven_objects 97
- Thinking (original): dyck_languages 70, logical_deduction_seven_objects 100, multistep_arithmetic_two 100, tracking_shuffled_objects_seven_objects 100
- D (the cut, never thinks): dyck_languages 30, logical_deduction_seven_objects 96, multistep_arithmetic_two 98, tracking_shuffled_objects_seven_objects 98
- router v2, old pool, accuracy first: dyck_languages 44, logical_deduction_seven_objects 97, multistep_arithmetic_two 100, tracking_shuffled_objects_seven_objects 99
- router v2, 32k pool, accuracy first: dyck_languages 62, logical_deduction_seven_objects 100, multistep_arithmetic_two 98, tracking_shuffled_objects_seven_objects 100
- router v2, 32k pool, balanced: dyck_languages 63, logical_deduction_seven_objects 100, multistep_arithmetic_two 98, tracking_shuffled_objects_seven_objects 99
- dial, old pool (H4b): dyck_languages 52, logical_deduction_seven_objects 100, multistep_arithmetic_two 100, tracking_shuffled_objects_seven_objects 100
- dial, 32k pool (H5b): dyck_languages 49, logical_deduction_seven_objects 100, multistep_arithmetic_two 97, tracking_shuffled_objects_seven_objects 100

## Reading the tables

Figures and counts in this section are two-run averages and two-run counts as defined above, except GSM8K (one run).

Removing P costs accuracy when the budget is large. At 32,768 tokens Thinking beats D on hard maths (90.1 against 79.2,
16 questions to 2) and on hard logic (92.5 against 80.5, 39 to 3). Most of the logic gap is one puzzle type: Dyck
languages, where Thinking scores 70, D 30 and Instruct 17; that one type accounts for 10 of the 12-point gap. D's mean
reply is 5.0 to 6.3 times shorter than Thinking's on all four sets (2,103 against 10,830 tokens on hard maths).

Instruct is also ahead of D on hard maths (85.4 against 79.2; Instruct better on 12 questions, D on 3, p 0.035). On the
other three sets the differences are inside the noise: hard logic 15 questions to 23 in D's favour (p 0.256), the MATH
test 4 to 2, GSM8K 11 to 4 (p 0.118).

The 32k-pool router at "accuracy first" is level with Thinking on hard maths (89.6 against 90.1, one question each
way), on the MATH test (95.0 against 91.0, 3 to 1) and on GSM8K (95.3 against 96.0), and it uses 7,421 tokens per reply
on hard maths against Thinking's 10,830. It is below Thinking on hard logic (90.0 against 92.5; 2 questions better, 10
worse, p 0.039). The old-pool router was trained on answers that Thinking often could not finish inside the original
caps (34 of its 440 pool answers hit a cap), which taught it that thinking does not pay on maths: it scores 79.7 on hard
maths, level with D.

## What you can recompute from this package

`router/score_long_budget.py` recomputes the Instruct, Thinking, D and router rows and the sign tests among them from
`items/c32-*` and `router/embed_long_budget.pt`. Everything matches the tables above except one cell: for the old-pool
router the recomputation gives 2,549 mean tokens on hard logic, where the table says 2,556. The saved routing decisions
of the original run and the recomputed ones differ on 1 of 646 questions at each of the two settings of lambda (a
different question each time, both close to a tie). The question vectors in this package come from a second embedding
pass over the same questions, which is the likely cause. The 32k-pool router agrees on all 646 questions at both
settings. Adding the Instruct-against-D pair to the script changed no other line of its output.

The two dial rows and the four sign tests that involve the dial are not reproducible here, because the dial's weights
and its per-question profiles are not included.
