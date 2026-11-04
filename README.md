![prefixgate](assets/banner.png)

prefixgate watches a coding agent's tool call while it is still being written and says allow, deny or escalate as
soon as it is sure enough. "Sure enough" is set by Learn-then-Test, so false-allow and false-deny stay under the
bounds you pick. It is a CPU-only experiment on Qwen2.5-0.5B-Instruct and the public Policy Gates data, with a local
dashboard that replays the real run. For the Meta hackathon it also gates a live agent: Llama 3.2 3B writes tool
calls on the same CPU and the gate decides on them token by token, before anything could run.

![python](https://img.shields.io/badge/python-3.10-black?style=flat-square) ![cpu only](https://img.shields.io/badge/cpu-only-black?style=flat-square) ![model](https://img.shields.io/badge/qwen2.5-0.5b-black?style=flat-square) ![agent](https://img.shields.io/badge/llama_3.2-3b-black?style=flat-square) ![license](https://img.shields.io/badge/license-MIT-black?style=flat-square) ![result](https://img.shields.io/badge/hypothesis-holds%2C%20degenerate-white?style=flat-square)

#ai-frontier-research #decision-models #agent-guardrails #conformal-prediction #tool-calling #streaming #system-one #cpu-inference #typed-decisions

![The dashboard](assets/hero.png)

## The Gate

A typed decision model reads the call one token at a time and gives three probabilities: allow, deny, escalate.
At each prefix the gate checks deny first, then escalate, then allow, against three thresholds. The first one that
crosses wins, and nothing after it can change the verdict. If nothing crosses, the finished call escalates.

Deny fails closed. Escalate hands the call to a judge. Neither one runs anything; in this repo they only change what
the screen says.

The thresholds come from Learn-then-Test on a calibration split: a 343-point grid, a Hoeffding-Bentkus p-value per
risk, Bonferroni over the grid, delta 0.1. Configurations that cannot be certified are not used, and when none can
be, the method says so.

![risk vs alpha](assets/risk.png)

On 776 held-out calls (315 carrying published prompt-injection templates aimed at `allow`) the stream keeps
false-allow at 0.000 to 0.003 and false-deny at 0.007 to 0.020, under every alpha tried. It holds on a later
469-call confirmation set from new families too.

## The Lead

The batch Policy Gate reads the full call once. prefixgate reaches the same decision a median **44 tokens earlier**
at alpha 0.05 / 0.1, 95% cluster-bootstrap CI **[34, 56]**, about 1.2 s of decode on this CPU. On the shift set it is
86 tokens, CI [68, 102]. Prompted Qwen, its entropy variant, and a probe at 50% of the call all trail it.

| | |
|---|---|
| ![lead vs baselines](assets/lead_ci.png) | ![lead histogram](assets/lead.png) |

Now the catch. The gate gets there by escalating 92% of test calls, most of them early. It allowed 10 and denied 49
of 776; batch allowed 64 and denied 64. Accuracy against gold drops from 0.54 to 0.46. The lead counted on allow and
deny alone is 33 tokens over 59 calls. So the hypothesis holds as written, but what was built is a fast, safe
escalator, not yet a useful allow/deny gate.

![trajectories](assets/trajectory.png)

## The Screen

`make app` serves the dashboard at `http://127.0.0.1:8765`. Replay picks any held-out call and scrubs through it:
the prefix strip greys out what has not arrived yet, the bars show the three probabilities against their calibrated
thresholds, the lead clock counts tokens saved, and the gate log notes each argmax change and the commit. The table
is the real held-out run. Pick an alpha pair at the top and everything re-gates.

![demo](assets/demo.gif)

Live mode takes a pasted tool call and streams it through the trained model on your CPU. Nothing is executed.

![live mode](assets/live.png)

The replay above is one call chosen for the demo (an attacked call both gates deny, with the largest lead). It is
not typical; the typical call escalates.

## The Agent

`make demo` serves the dashboard with the agent tab open. Pick a task, press **run live**: Llama 3.2 3B Instruct
(Meta's open weights, fp32 on this 8-thread CPU, greedy, its native tool-call template) writes one call, and every
token goes straight into the decision model, which scores the growing prefix. The gate log, bars and verdict update
as the tokens land. Nothing is executed; allow only means "would be passed to the tool".

![agent demo](assets/agent.gif)

The GIF is a screen capture of three real live runs (`rm-rf-home`, `git-status`, `wire-stripe`) at 2x speed. The
agent tab gates with v2: the stream thresholds on every prefix, plus a second calibrated triple on the finished call,
both certified together by Learn-then-Test.

The recorded run in `results/agent_runs.json` is 24 generated tasks, 16 of them risky (`rm -rf`, wiring money,
emailing a customer export out, reading AWS keys, dropping a table, force pushes, `curl | sh`, prod deploys):

| | allow | escalate | deny |
|---|---|---|---|
| 8 benign tasks | 6 | 2 | 0 |
| 16 risky tasks | 2 | 14 | 0 |

So the gate held 14 of 16 risky calls for a human and passed 6 of 8 benign ones, with a median 17.8 s per task, most
of it Llama decoding. It is not the clean demo: it never says *deny*, even for `rm -rf` (deny peaks at 0.63 against a
0.9 threshold); six of the 24 commit mid-stream, the rest only on the finished call; and the two risky calls it
allowed are ones Llama rewrote into something milder (a plain `git push` for "force push", a Slack post for
"public GitHub comment"). Small-scale, single run; every task, token and probability is in `results/agent_runs.json`.

## The Hackathon

Meta Global AI Developer Hackathon, Indie track, Push the Frontier. The entry is this repo:

- **Meta open weights on a CPU.** Llama 3.2 3B Instruct from the ungated mirror
  `unsloth/Llama-3.2-3B-Instruct@006f5dc` (Llama 3.2 Community License, see [NOTICE.md](NOTICE.md)). No GPU, no
  cloud, no paid API.
- **A gate on the stream, with a bound.** prefixgate decides while the agent is still writing, and its thresholds
  carry a Learn-then-Test guarantee on false-allow and false-deny instead of a hand-picked cut-off.
- **Muse Spark, pluggable.** Set `META_API_KEY` and the dashboard offers the Meta Model API (`muse-spark-1.3`,
  streamed chat completions) as a second agent source; without it the adapter is never called. No committed result
  uses it.
- **It runs.** `make demo` locally, or the recorded runs in the browser on the
  [Hugging Face Space](https://huggingface.co/spaces/sroda/prefixgate).

## Install

```sh
git clone https://github.com/Sebastianrodaaa/prefixgate && cd prefixgate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.46.3 "jinja2>=3.1" safetensors numpy scipy scikit-learn matplotlib pillow
make test            # data checks, LTT, gate invariants (no model needed)
make run plots       # rebuild results/metrics.json and the plots from committed scores, ~3 min
make app             # dashboard on :8765
make demo            # the same, agent tab: live Llama 3.2 3B runs (first run downloads about 6.5 GB)
make agent           # re-record results/agent_runs.json, 24 tasks, ~8 min
make all             # the whole run from scratch: train, score, baselines, ~5.5 h on 8 CPU threads
```

It downloads the Policy Gates release into `data/` (sha-256 checked) and the Qwen weights into the Hugging Face cache
(`~/.cache/huggingface`, about 1 GB). It writes to `results/` and `assets/`. To undo, delete the clone,
`data/`, `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct` and
`~/.cache/huggingface/hub/models--unsloth--Llama-3.2-3B-Instruct`.

### Hugging Face Space

[huggingface.co/spaces/sroda/prefixgate](https://huggingface.co/spaces/sroda/prefixgate) is a free static Space:
`scripts/static_space.py` writes the server's answers for the recorded runs as files next to the dashboard. It opens
on the agent tab and replays the 24 recorded Llama 3.2 3B runs, labelled as replays, plus the replay tab; no model
runs there, so both live modes are off ([space/README.md](space/README.md) has the measured 2-CPU timings).
[`space/Dockerfile`](space/Dockerfile) is the full dashboard in a container (paste-a-call live scoring on, live agent
off); as a Space it needs a Docker Space, which on HF needs PRO.

```sh
python3 scripts/static_space.py                                  # build/space/, serve it with any static server
HF_TOKEN=hf_... python3 scripts/static_space.py --push sroda/prefixgate
docker build -t prefixgate-space space && docker run --rm -p 7860:7860 prefixgate-space   # http://127.0.0.1:7860
```

## The Name

A gate on a prefix. The tool call is still arriving and the decision is already at the door. Most of the time,
for now, the decision is "ask someone".

## Inspirations

Decision Models as Policy Gates, Learn then Test, Llama 3.2, Ranked but Not Stopped, Local or Escalate, and athanor's README.
See [INSPIRATIONS.md](INSPIRATIONS.md).

## License

MIT, see [LICENSE](LICENSE). Data, models (Qwen2.5, Llama 3.2: Built with Llama), art and fonts keep their own terms, listed in [NOTICE.md](NOTICE.md).
