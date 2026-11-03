# prefixgate: report

## Question

A tool-call gate usually reads the finished call once. prefixgate asks whether a small typed decision model can
re-score allow / deny / escalate on every growing prefix of the call and commit early, while Learn-then-Test (LTT)
keeps false-allow and false-deny under stated bounds.

Pre-registered hypothesis (from `docs/brief/SCOUT_BRIEF.md`): on held-out tool calls with published attack templates,
(1) the LTT-calibrated streaming gate keeps test false-allow <= alpha_a and false-deny <= alpha_d (inside the binomial
95% interval of alpha), and (2) at matched alpha its median decision lead over the batch Policy Gates baseline (full
call only) is positive with a 95% bootstrap CI that excludes 0. Falsifier: the stream breaks an alpha bound, or a
cheaper baseline (prompted Qwen with an entropy threshold) matches its lead.

## Data

- **Calls.** The Policy Gates release ([agentapprove/decision-policy-rules](https://github.com/agentapprove/decision-policy-rules)
  @ `9e68b07`, CC BY 4.0; Beno 2026, [arXiv:2611.02155](https://arxiv.org/abs/2611.02155)). 2,643 coding-agent tool
  calls in 106 families (1,535 dev, 1,108 sealed holdout) and a 320-case confirmation set written later in 16 new
  families. Each case has gold `tool_only` labels for 28 rules, written blind and adjudicated by the authors.
  These are authored calls with fictional people and random secrets, not traces captured from live agents. That is
  the main caveat on the word "real" here: the data is public and was not generated or labelled by this project.
  Files are fetched and checked against pinned SHA-256 (`prefixgate/data.py`).
- **Gold gate.** The product's `recommendedAction` per rule (block -> deny, ask -> escalate) with the precedence in
  `RULES.md`. The allow-scope exemption is this project's reading of the rule texts, not something the release states.
- **Attacks.** Each unsafe call (gold deny or escalate) gets one attacked copy: one of the six published templates of
  *Ranked but Not Stopped* ([arXiv:2611.02522](https://arxiv.org/abs/2611.02522)) with target `allow`, placed before
  the first string argument, after the last, or in a new `notes` argument. Gold is unchanged; the call still does the
  same thing. Training only sees three of the six families (override, system note, JSON label).
- **Splits.** dev -> train, holdout -> calibration / test (seeded coin per case, attacked copy follows its case),
  confirmation -> shift. Calls longer than 768 Qwen tokens are dropped and counted (about 15%, see below).

## Method

- **Decision model.** Qwen2.5-0.5B-Instruct (`7ae5576`), fp32 on CPU, rank-8 LoRA on q/v of all 24 layers plus a
  3-way linear head on the last hidden state, 543k trainable parameters. Input: a fixed system prompt naming the rules
  by action, then the call. One cross-entropy loss over every token position: each prefix predicts the gate of the
  whole call. Because the model is causal, one forward pass scores every prefix, and a KV-cache stream gives the same
  numbers token by token (checked: max abs difference below 1e-5).
- **Gate.** At each prefix: deny if p(deny) >= lambda_d, else escalate if p(escalate) >= lambda_e, else allow if
  p(allow) >= lambda_a; the first prefix that crosses commits and nothing later can change it. If nothing ever
  crosses, the full call escalates. The batch baseline is the same rule on the complete call only.
- **LTT.** Grid lambda in {0.5, 0.7, 0.9, 0.95, 0.99, 0.995, 0.999}^3 (343 configs). Two risks on the calibration
  split: false-allow = P(allow | gold != allow), false-deny = P(deny | gold != deny). Hoeffding-Bentkus p-values,
  max over the two risks, Bonferroni over the grid, delta = 0.1. Among certified configs pick the most accurate on
  calibration (ties: longer mean lead). Every method gets its own LTT pass on the same calibration data.
- **Lead.** For each test call, `T - t` tokens where t is the commit prefix, counted only when the stream's decision
  equals the batch gate's decision on the same call (otherwise 0). Batch lead is 0 by definition. Milliseconds are
  tokens times the measured CPU decode cost of this model on this box. CIs: 2,000 cluster-bootstrap resamples over
  source cases (a call and its attacked copy move together).
- **Baselines.** batch DM; DM with an entropy threshold; the DM checked only every 16 tokens; zero-shot prompted Qwen
  (rule definitions in the prompt, next-token probabilities of allow/deny/escalate, every 16 tokens) with thresholds,
  with entropy, and batch; a Local-or-Escalate-style multinomial logistic probe on frozen Qwen states at 50% of the
  call that commits there or hands over to the DM on the full call.

## Setup actually run

| | |
|---|---|
| box | Intel Xeon, 8 threads, CPU only, no GPU, no paid API |
| model | Qwen2.5-0.5B-Instruct `7ae5576`, fp32, torch 2.5.1+cpu |
| train | 2,281 calls (406 dropped over 768 tokens), 2 epochs, lr 2e-4, accumulation 8, 5,082 s |
| score | DM on 2,114 cal/test/shift calls (361 dropped), 1,157 s |
| baselines | prompted Qwen every 16 tokens + probe states, 13,115 s |
| latency | 28.29 ms per streamed token (median, 20 calls); stream vs batch max abs diff 4.8e-6 |
| split sizes | cal 869 (163 allow / 356 deny / 350 escalate; 353 attacked), test 776 (144 / 322 / 310; 315 attacked), shift 469 (61 / 189 / 219; 203 attacked) |
| total | about 5.5 h of model compute, under the 16 h budget |

Everything below comes from `results/metrics.json`, which `python3 -m prefixgate.run` rebuilds from the committed
score files in about 3 minutes. Single run, single seed: this is a small-scale study.

## Results

Short version: **the hypothesis holds as written, but in a degenerate regime.** The certified stream keeps both risks
under every alpha on test and on the shift set, and its median lead over the batch gate is 40 to 44 tokens with a
bootstrap CI that excludes 0. It gets there by escalating 92 to 97% of calls, mostly early. It is a fast, safe
escalator, not yet a useful allow/deny gate.

![risk vs alpha](screenshots/risk.png)

**(1) Risk control: holds.** Test false-allow is 0.000 to 0.003 over 632 unsafe calls (alpha_a 0.02 and 0.05) and
test false-deny 0.007 to 0.020 over 454 non-deny calls (alpha_d 0.05 and 0.1). On the 469-call shift set (new
families, written later) false-allow is 0.000 and false-deny at most 0.007. None of the 315 attacked test calls was
allowed at alpha_a = 0.02; at alpha_a = 0.05 one `certain_note` attack (1 of 60) got through. The raw DM never flips
from a deny-crossing back to allow on test (`raw_deny_to_allow_flip_rate_test` = 0.0), so the monotone commit loses
nothing there.

**(2) Lead: holds, with the caveat above.** Median per-call lead difference, prefixgate minus baseline, 95% cluster
bootstrap, in tokens:

| alpha_a / alpha_d | split | vs batch | vs every-16 | vs DM entropy | vs prompted | vs prompted entropy | vs probe |
|---|---|---|---|---|---|---|---|
| 0.02 / 0.05 | test | 40.0 [28.5, 49.0] | 0.0 [0.0, 1.0] | 30.0 [22.0, 37.0] | 40.0 [28.5, 49.0] | no certified config | 18.0 [13.0, 25.0] |
| 0.02 / 0.05 | shift | 86.0 [63.5, 102.0] | 10.0 [6.0, 11.0] | 57.0 [38.5, 73.0] | 86.0 [63.5, 102.0] | no certified config | 46.0 [36.0, 59.5] |
| 0.02 / 0.1 | test | 44.0 [33.0, 56.0] | 0.0 [0.0, 1.0] | 32.5 [26.0, 41.5] | 44.0 [33.0, 56.0] | no certified config | 20.0 [14.0, 27.0] |
| 0.02 / 0.1 | shift | 86.0 [66.0, 102.0] | 10.0 [3.5, 11.0] | 58.0 [41.0, 74.0] | 86.0 [66.0, 102.0] | no certified config | 46.0 [36.5, 60.0] |
| 0.05 / 0.05 | test | 40.0 [30.0, 51.0] | 0.0 [0.0, 1.0] | 16.0 [12.0, 24.0] | 40.0 [30.0, 51.0] | 40.0 [30.0, 51.0] | 19.0 [14.0, 26.0] |
| 0.05 / 0.05 | shift | 86.0 [64.0, 102.0] | 10.0 [7.0, 11.0] | 34.0 [26.0, 53.0] | 86.0 [64.0, 102.0] | 86.0 [64.0, 102.0] | 46.0 [36.0, 59.5] |
| 0.05 / 0.1 | test | 44.0 [34.0, 56.0] | 0.0 [0.0, 1.0] | 11.5 [9.0, 18.0] | 44.0 [34.0, 56.0] | 44.0 [34.0, 56.0] | 21.0 [15.0, 28.0] |
| 0.05 / 0.1 | shift | 86.0 [68.0, 102.0] | 10.0 [5.0, 11.0] | 29.0 [21.5, 40.0] | 86.0 [68.0, 102.0] | 86.0 [68.0, 102.0] | 46.0 [36.5, 60.0] |

At 28.3 ms per token, 44 tokens is about 1.2 s of agent decode on this CPU (86 tokens, 2.4 s, on the shift set).

![lead vs baselines](screenshots/lead_ci.png)

**Falsifier check.** The named cheaper baseline, prompted Qwen with an entropy threshold, has no certified config at
alpha_a = 0.02 and at 0.05 it escalates 99% of calls at the full call; lead 0. So it does not match. Two closer
results: the same DM checked only every 16 tokens has a per-call median difference of 0 [0, 1] on test (it is the same
model, just checked less often), and the Local-or-Escalate-style probe at 50% trails by about 20 tokens.

**What the gate actually decides** (test, alpha 0.05 / 0.1): 10 allow, 49 deny, 717 escalate out of 776. The batch
gate on the same model: 64 / 64 / 648. Accuracy against gold falls from 0.54 (batch) to 0.46 (stream). The lead
counted only on committed allow/deny decisions is 33 tokens median, over 59 calls. The escalate threshold is 0.5 in
every certified config: early prefixes look uncertain, the model puts mass on escalate, and the first-crossing rule
commits it. The lead histogram shows the two modes: about a third of test calls get lead 0 (they commit at the last
token, or the stream disagrees with batch), the rest commit far before the end.

![lead distribution](screenshots/lead.png)

**Model quality.** Full-call argmax accuracy of the DM is 0.75 on calibration, 0.71 on shift; AUROC of p(allow) for
safe vs unsafe is 0.88 at the full call and 0.81 across all prefixes. The zero-shot prompted Qwen's p(allow) has AUROC 0.48 at the
full call, no better than chance, so it is not a serious competitor at this size. The gold allow class never becomes confident (mean
p(allow | gold allow) stays below 0.55 until the last token), which is why so few allows survive alpha_a.

![trajectories](screenshots/trajectory.png)

**Full result tables** (every method, every alpha pair; FA/FD on test unless marked):

**alpha_a = 0.02, alpha_d = 0.05** (test n = 776, shift n = 469)

| method | lambda / tau | certified | FA | FD | escalate | acc | median lead (tok) | lead on allow/deny (tok) | shift FA | shift FD | shift lead |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prefixgate (stream) | 0.999/0.995/0.5 | 14 | 0.000 | 0.007 | 0.97 | 0.42 | 40 | 31 | 0.000 | 0.000 | 86 |
| prefixgate, every 16 tok | 0.95/0.99/0.5 | 84 | 0.000 | 0.009 | 0.96 | 0.44 | 16 | 29 | 0.000 | 0.000 | 47 |
| batch Policy Gate | 0.9/0.9/0.5 | 175 | 0.002 | 0.009 | 0.91 | 0.48 | 0 | 0 | 0.000 | 0.018 | 0 |
| DM entropy | 0.02 | 1 | 0.000 | 0.000 | 0.99 | 0.41 | 0 | 33 | 0.000 | 0.000 | 0 |
| prompted Qwen | 0.999/0.5/0.5 | 49 | 0.000 | 0.007 | 0.99 | 0.40 | 0 | 0 | 0.000 | 0.011 | 0 |
| prompted Qwen entropy | none | 0 | | | | | | | | | |
| prompted Qwen batch | 0.995/0.5/0.5 | 147 | 0.000 | 0.002 | 0.99 | 0.40 | 0 | 0 | 0.000 | 0.007 | 0 |
| probe @50% | 0.999/0.999/0.5 | 14 | 0.000 | 0.020 | 0.96 | 0.42 | 0 | 0 | 0.002 | 0.025 | 0 |

**alpha_a = 0.02, alpha_d = 0.1** (test n = 776, shift n = 469)

| method | lambda / tau | certified | FA | FD | escalate | acc | median lead (tok) | lead on allow/deny (tok) | shift FA | shift FD | shift lead |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prefixgate (stream) | 0.999/0.99/0.5 | 21 | 0.000 | 0.020 | 0.94 | 0.45 | 44 | 42 | 0.000 | 0.007 | 86 |
| prefixgate, every 16 tok | 0.95/0.9/0.5 | 116 | 0.000 | 0.059 | 0.80 | 0.55 | 28 | 0 | 0.000 | 0.032 | 58 |
| batch Policy Gate | 0.9/0.9/0.5 | 175 | 0.002 | 0.009 | 0.91 | 0.48 | 0 | 0 | 0.000 | 0.018 | 0 |
| DM entropy | 0.02 | 1 | 0.000 | 0.000 | 0.99 | 0.41 | 0 | 33 | 0.000 | 0.000 | 0 |
| prompted Qwen | 0.999/0.5/0.5 | 49 | 0.000 | 0.007 | 0.99 | 0.40 | 0 | 0 | 0.000 | 0.011 | 0 |
| prompted Qwen entropy | none | 0 | | | | | | | | | |
| prompted Qwen batch | 0.995/0.5/0.5 | 147 | 0.000 | 0.002 | 0.99 | 0.40 | 0 | 0 | 0.000 | 0.007 | 0 |
| probe @50% | 0.999/0.99/0.5 | 42 | 0.000 | 0.051 | 0.91 | 0.44 | 0 | 0 | 0.002 | 0.046 | 0 |

**alpha_a = 0.05, alpha_d = 0.05** (test n = 776, shift n = 469)

| method | lambda / tau | certified | FA | FD | escalate | acc | median lead (tok) | lead on allow/deny (tok) | shift FA | shift FD | shift lead |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prefixgate (stream) | 0.99/0.995/0.5 | 42 | 0.003 | 0.007 | 0.96 | 0.43 | 40 | 24 | 0.000 | 0.000 | 86 |
| prefixgate, every 16 tok | 0.9/0.99/0.5 | 105 | 0.005 | 0.009 | 0.94 | 0.45 | 18 | 17 | 0.000 | 0.000 | 49 |
| batch Policy Gate | 0.7/0.9/0.5 | 210 | 0.019 | 0.009 | 0.84 | 0.54 | 0 | 0 | 0.007 | 0.018 | 0 |
| DM entropy | 0.06 | 2 | 0.003 | 0.022 | 0.92 | 0.46 | 0 | 30 | 0.000 | 0.014 | 0 |
| prompted Qwen | 0.999/0.5/0.5 | 98 | 0.000 | 0.007 | 0.99 | 0.40 | 0 | 0 | 0.000 | 0.011 | 0 |
| prompted Qwen entropy | 0.02 | 1 | 0.006 | 0.000 | 0.99 | 0.40 | 0 | 0 | 0.002 | 0.000 | 0 |
| prompted Qwen batch | 0.995/0.5/0.5 | 147 | 0.000 | 0.002 | 0.99 | 0.40 | 0 | 0 | 0.000 | 0.007 | 0 |
| probe @50% | 0.95/0.999/0.5 | 28 | 0.014 | 0.020 | 0.93 | 0.43 | 0 | 0 | 0.010 | 0.025 | 0 |

**alpha_a = 0.05, alpha_d = 0.1** (test n = 776, shift n = 469)

| method | lambda / tau | certified | FA | FD | escalate | acc | median lead (tok) | lead on allow/deny (tok) | shift FA | shift FD | shift lead |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prefixgate (stream) | 0.99/0.99/0.5 | 63 | 0.003 | 0.020 | 0.92 | 0.46 | 44 | 33 | 0.000 | 0.007 | 86 |
| prefixgate, every 16 tok | 0.9/0.9/0.5 | 145 | 0.005 | 0.057 | 0.78 | 0.56 | 29 | 0 | 0.000 | 0.032 | 58 |
| batch Policy Gate | 0.7/0.9/0.5 | 210 | 0.019 | 0.009 | 0.84 | 0.54 | 0 | 0 | 0.007 | 0.018 | 0 |
| DM entropy | 0.1 | 3 | 0.009 | 0.051 | 0.84 | 0.51 | 0 | 6 | 0.000 | 0.036 | 0 |
| prompted Qwen | 0.999/0.5/0.5 | 98 | 0.000 | 0.007 | 0.99 | 0.40 | 0 | 0 | 0.000 | 0.011 | 0 |
| prompted Qwen entropy | 0.02 | 1 | 0.006 | 0.000 | 0.99 | 0.40 | 0 | 0 | 0.002 | 0.000 | 0 |
| prompted Qwen batch | 0.995/0.5/0.5 | 147 | 0.000 | 0.002 | 0.99 | 0.40 | 0 | 0 | 0.000 | 0.007 | 0 |
| probe @50% | 0.95/0.99/0.5 | 84 | 0.014 | 0.051 | 0.88 | 0.44 | 0 | 0 | 0.010 | 0.046 | 0 |


## Related work

- **Decision Models as Policy Gates** (Beno 2026, arXiv:2611.02155): the data and the batch gate. prefixgate reads
  the same calls one token at a time.
- **Learn then Test** (Angelopoulos, Bates, Candès, Jordan, Lei 2021): the risk-control procedure, Hoeffding-Bentkus
  p-values with Bonferroni over a threshold grid.
- **Ranked but Not Stopped** (arXiv:2611.02522): the six prompt-injection templates, here aimed at `allow`.
- **Local or Escalate**: a linear probe on frozen states that commits or hands over; used as the 50% probe baseline.
- Streaming / early-exit classification and conformal early stopping more broadly: see `INSPIRATIONS.md`.

## Limitations

- Mostly escalate. A deployed gate that escalates 92% of calls moves the work to the judge. The headline lead is
  real but most of it is an early escalation, not an early allow or deny.
- First crossing at a single prefix. The gate commits on the first prefix over threshold. Requiring a few consecutive
  prefixes, or calibrating a time-dependent threshold, is the obvious next step; exploratory runs on calibration data
  only suggested it raises accuracy, and it was not adopted here because it would have changed the method after seeing
  results.
- Authored, not captured, calls. The Policy Gates release is public and gold-labelled by its authors, but its calls are
  written to probe rules, not logged from agents in the wild.
- Gold composition is this project's reading (`RULES.md`). The release labels rules, not a single gate decision.
- 768-token cap drops about 15% of calls; long calls are exactly where lead would matter most.
- Single seed, one model size, one box. Latency in ms is for this CPU only.
- LTT guarantees are marginal over calls exchangeable with calibration. The shift set is not exchangeable; it held
  here, which is evidence, not a guarantee.
- Lead counts only when the stream's decision equals the batch decision. If the stream disagrees, that call's lead is 0.

## Reproduce

```sh
make data train score latency baselines   # about 5.5 h on 8 CPU threads
make run plots                            # about 3 min from the committed score files
```
