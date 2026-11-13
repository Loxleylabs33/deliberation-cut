# First router test: GSM8K 300, MATH 50, BBH 90, original token caps

Router: `router/router_pool440_h3b/router_v2_pool440.pt`, trained on a pool of 440 questions that share nothing with the
test items (240 BIG-Bench Hard examples 15 to 54 of the same six tasks, 100 MATH-500 problems outside the test set, 100
GSM8K training problems), each answered once at each of three settings under the original caps (GSM8K 4,096 tokens,
MATH and BBH 8,192). Settings: 0 = D (the released model), 0.75 = D + 0.75 P (Thinking minus 0.25 P), 1 = Thinking.

The table is the output of `router/score_router.py router_pool440_h3b/router_v2_pool440.pt` run on the per-item results
in `items/` (the three settings, BBH averaged over three runs each). Every number matches the table in our run log. The
header line reports the pool's mean reply length by setting.

```
# router router_v2_pool440.pt: pool mean tokens by setting {'dial0': 523, 'dial75': 2139, 'dial100': 3158}

| policy | gsm8k | math | bbh | mean tokens | thinks on |
|---|---|---|---|---|---|
| always D (0) | 93.7 | 86.0 | 90.4 | 405 | 0% |
| always Thinking (1) | 93.0 | 64.0 | 94.8 | 2219 | 100% |
| router v2, lambda 0.0 | 95.0 | 84.0 | 96.3 | 1195 | 67% |
| router v2, lambda 0.005 | 95.0 | 84.0 | 95.2 | 877 | 39% |
| router v2, lambda 0.01 | 94.0 | 82.0 | 94.8 | 778 | 28% |
| router v2, lambda 0.02 | 93.3 | 82.0 | 94.8 | 681 | 18% |
| router v2, lambda 0.05 | 93.7 | 82.0 | 93.0 | 560 | 8% |

Router v2 (lambda 0.01) setting chosen by question type (share choosing 0 / 0.75 / 1): bbh-boolean_expressions 33/0/67; bbh-date_understanding 27/20/53; bbh-formal_fallacies 100/0/0; bbh-logical_deduction_five_objects 100/0/0; bbh-tracking_shuffled_objects_three_objects 80/20/0; bbh-web_of_lies 0/100/0; gsm8k 76/18/6; math 78/16/6
```

How to read it. "always D" and "always Thinking" use one model for every question. In the router rows, lambda is the
price of 1,000 tokens in probability points: the router picks the setting with the highest `p_correct - lambda x
pool_mean_tokens / 1000`, and "thinks on" is the share of questions sent to 0.75 or to Thinking. BBH is the average of
the six tasks' items (15 each, 90 in all). The BBH "always D" figure of 90.4 is the three-run average; the single run in
`scorecard.md` scored 90.0.

Each test question is scored with what the model at the chosen setting actually did on it in the per-item results; the
router is not run in a loop with the models.

By question type the router (lambda 0.01) sent every formal_fallacies and logical_deduction_five_objects question to D,
every web_of_lies question to setting 0.75, and two thirds of the boolean_expressions questions to Thinking. On GSM8K
and MATH it chose D for about three quarters of the questions (76% and 78%). Over the 440 test questions it scores 94.0
on GSM8K, 82.0 on MATH and 94.8 on BBH at a mean of 778 tokens per reply. Always-Thinking scores 93.0, 64.0 and 94.8 at
2,219 tokens; always-D scores 93.7, 86.0 and 90.4 at 405.

The pool has 440 questions and the test sets have 50 to 300, so a difference of one or two points between router rows is
inside the noise (standard errors of about 1.4 points on GSM8K, 4.9 on MATH, 3.2 on BBH).

