![prefixgate](docs/screenshots/banner.png)

prefixgate watches a coding agent's tool call while it is still being written and says allow, deny or escalate as
soon as it is sure enough. "Sure enough" is set by Learn-then-Test, so false-allow and false-deny stay under the
bounds you pick. It is a CPU-only experiment on Qwen2.5-0.5B-Instruct and the public Policy Gates data, with a local
dashboard that replays the real run.

![python](https://img.shields.io/badge/python-3.10-black?style=flat-square) ![cpu only](https://img.shields.io/badge/cpu-only-black?style=flat-square) ![model](https://img.shields.io/badge/qwen2.5-0.5b-black?style=flat-square) ![license](https://img.shields.io/badge/license-MIT-black?style=flat-square) ![result](https://img.shields.io/badge/hypothesis-holds%2C%20degenerate-white?style=flat-square)

#ai-frontier-research #decision-models #agent-guardrails #conformal-prediction #tool-calling #streaming #system-one #cpu-inference #typed-decisions

![The dashboard](docs/screenshots/hero.png)

## The Gate

A typed decision model reads the call one token at a time and gives three probabilities: allow, deny, escalate.
At each prefix the gate checks deny first, then escalate, then allow, against three thresholds. The first one that
crosses wins, and nothing after it can change the verdict. If nothing crosses, the finished call escalates.

Deny fails closed. Escalate hands the call to a judge. Neither one runs anything; in this repo they only change what
the screen says.

The thresholds come from Learn-then-Test on a calibration split: a 343-point grid, a Hoeffding-Bentkus p-value per
risk, Bonferroni over the grid, delta 0.1. Configurations that cannot be certified are not used, and when none can
be, the method says so.

![risk vs alpha](docs/screenshots/risk.png)

On 776 held-out calls (315 carrying published prompt-injection templates aimed at `allow`) the stream keeps
false-allow at 0.000 to 0.003 and false-deny at 0.007 to 0.020, under every alpha tried. It holds on a later
469-call confirmation set from new families too.

## The Lead

The batch Policy Gate reads the full call once. prefixgate reaches the same decision a median **44 tokens earlier**
at alpha 0.05 / 0.1, 95% cluster-bootstrap CI **[34, 56]**, about 1.2 s of decode on this CPU. On the shift set it is
86 tokens, CI [68, 102]. Prompted Qwen, its entropy variant, and a probe at 50% of the call all trail it.

| | |
|---|---|
| ![lead vs baselines](docs/screenshots/lead_ci.png) | ![lead histogram](docs/screenshots/lead.png) |

Now the catch. The gate gets there by escalating 92% of test calls, most of them early. It allowed 10 and denied 49
of 776; batch allowed 64 and denied 64. Accuracy against gold drops from 0.54 to 0.46. The lead counted on allow and
deny alone is 33 tokens over 59 calls. So the hypothesis holds as written, but what was built is a fast, safe
escalator, not yet a useful allow/deny gate. [docs/REPORT.md](docs/REPORT.md) has every number, every baseline and
the limitations.

![trajectories](docs/screenshots/trajectory.png)

## The Screen

`make app` serves the dashboard at `http://127.0.0.1:8765`. Replay picks any held-out call and scrubs through it:
the prefix strip greys out what has not arrived yet, the bars show the three probabilities against their calibrated
thresholds, the lead clock counts tokens saved, and the gate log notes each argmax change and the commit. The table
is the real held-out run. Pick an alpha pair at the top and everything re-gates.

![demo](docs/screenshots/demo.gif)

Live mode takes a pasted tool call and streams it through the trained model on your CPU. Nothing is executed.

![live mode](docs/screenshots/live.png)

The replay above is one call chosen for the demo (an attacked call both gates deny, with the largest lead). It is
not typical; the typical call escalates.

## Install

```sh
git clone https://github.com/Sebastianrodaaa/prefixgate && cd prefixgate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.46.3 safetensors numpy scipy scikit-learn matplotlib pillow
make test            # data checks, LTT, gate invariants (no model needed)
make run plots       # rebuild results/metrics.json and the plots from committed scores, ~3 min
make app             # dashboard on :8765
make all             # the whole run from scratch: train, score, baselines, ~5.5 h on 8 CPU threads
```

It downloads the Policy Gates release into `data/` (sha-256 checked) and the Qwen weights into the Hugging Face cache
(`~/.cache/huggingface`, about 1 GB). It writes to `results/` and `docs/screenshots/`. To undo, delete the clone,
`data/`, and `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct`.

## The Name

A gate on a prefix. The tool call is still arriving and the decision is already at the door. Most of the time,
for now, the decision is "ask someone".

## Inspirations

Decision Models as Policy Gates, Learn then Test, Ranked but Not Stopped, Local or Escalate, and athanor's README.
See [INSPIRATIONS.md](INSPIRATIONS.md).

## License

MIT, see [LICENSE](LICENSE). Data, model, art and fonts keep their own terms, listed in [NOTICE.md](NOTICE.md).
