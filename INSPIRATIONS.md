# INSPIRATIONS

prefixgate is a small experiment that stands on these.

- **Decision Models as Policy Gates for Coding Agents** (Beno, 2026, [arXiv:2611.02155](https://arxiv.org/abs/2611.02155)).
  Typed decision models scoring a finished tool call against explicit rules. The data, the 28 rules and the
  batch baseline all come from its release. prefixgate asks only *when* the same decision can be made.
- **Learn then Test** (Angelopoulos, Bates, Candès, Jordan, Lei, 2021, [arXiv:2110.01052](https://arxiv.org/abs/2110.01052)).
  Risk control by multiple testing over a threshold grid. The Hoeffding-Bentkus p-value and the Bonferroni
  correction in `prefixgate/ltt.py` follow it.
- **Ranked but Not Stopped** ([arXiv:2611.02522](https://arxiv.org/abs/2611.02522)). Published prompt-injection
  templates against typed decision models; used here verbatim as the attack set.
- **Local or Escalate** ([arXiv:2611.01639](https://arxiv.org/abs/2611.01639)). Routing at one frame; the 50%-prefix
  logistic probe is the one-frame version of that idea, run as an ablation.
- **Flide** ([arXiv:2610.42148](https://arxiv.org/abs/2610.42148)) on cheap decision-model inference, and
  **LeanGuard** ([arXiv:2611.02332](https://arxiv.org/abs/2611.02332)), the hard check an escalation could hand to.
