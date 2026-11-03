"""Black-and-white plots of the real run, from results/metrics.json and the cached scores."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from prefixgate.run import jload  # noqa: E402

OUT = ROOT / "docs" / "screenshots"
plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
                     "text.color": "#f2f2f2", "axes.labelcolor": "#a3a3a3", "axes.edgecolor": "#3a3a3a",
                     "xtick.color": "#a3a3a3", "ytick.color": "#a3a3a3", "font.family": "monospace", "font.size": 9,
                     "axes.grid": True, "grid.color": "#1c1c1c", "legend.frameon": False})
NAMES = {"prefixgate": "prefixgate", "prefixgate_stride16": "prefixgate /16", "batch_policy_gate": "batch gate",
         "dm_entropy": "dm entropy", "prompted_threshold": "prompted", "prompted_entropy": "prompted entropy",
         "prompted_batch": "prompted batch", "probe_50pct": "probe @50%"}


def main():
    M = jload("metrics.json")
    meta = jload("records.json")
    head = next(r for r in M["runs"] if (r["alpha_allow"], r["alpha_deny"]) == (0.05, 0.1))

    # 1. risk vs alpha: every method, every alpha pair, test split
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    marks = dict(zip(NAMES, "o s ^ v D P X *".split()))
    for j, (risk, key, akey) in enumerate((("false-allow", "false_allow", "alpha_allow"), ("false-deny", "false_deny", "alpha_deny"))):
        for i, run in enumerate(M["runs"]):
            for k, m in run["methods"].items():
                if m.get("test"):
                    ax[j].scatter(i + (list(NAMES).index(k) - 3.5) * 0.07, m["test"][key], marker=marks[k], s=26,
                                  facecolor="#f2f2f2" if k == "prefixgate" else "none", edgecolor="#f2f2f2" if k == "prefixgate" else "#8a8a8a",
                                  label=NAMES[k] if i == 0 else None)
            ax[j].hlines(run[akey], i - 0.35, i + 0.35, color="#f2f2f2", lw=1.2, linestyles="--")
        ax[j].set_xticks(range(len(M["runs"])), [f"αa={r['alpha_allow']}\nαd={r['alpha_deny']}" for r in M["runs"]])
        ax[j].set_title(f"test {risk} vs α (dashed)", loc="left", fontsize=9)
    ax[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout(); fig.savefig(OUT / "risk.png", dpi=140); plt.close(fig)

    # 2. lead distribution at the headline alpha: prefixgate vs batch (batch is zero by construction)
    pc, bc = head["per_call_test"], head["batch_per_call_test"]
    T = np.array([meta[k]["T"] for k in pc])
    lead = np.array([meta[k]["T"] - t if d == bc[k][0] else 0 for k, (d, t) in pc.items()])
    frac = lead / T
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].hist(lead, bins=40, color="#f2f2f2", edgecolor="black")
    ax[0].axvline(np.median(lead), color="#f2f2f2", ls="--")
    ax[0].set_title(f"prefixgate lead before full call (tokens), test, αa=0.05 αd=0.1, median {np.median(lead):.0f}", loc="left", fontsize=9)
    ax[1].hist(frac, bins=40, color="#8a8a8a", edgecolor="black")
    ax[1].set_title("lead as a fraction of the call", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "lead.png", dpi=140); plt.close(fig)

    # 3. median lead per method with the bootstrap CI of prefixgate minus that method
    pg = head["methods"]["prefixgate"]["test"]
    ks = [k for k in NAMES if k != "prefixgate" and pg and f"ci95_vs_{k}" in pg]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, k in enumerate(ks):
        lo, hi = pg[f"ci95_vs_{k}"]
        ax.plot([lo, hi], [i, i], color="#f2f2f2", lw=2)
        ax.scatter(pg[f"median_lead_diff_vs_{k}"], i, color="#f2f2f2", zorder=3)
    ax.axvline(0, color="#8a8a8a", ls="--")
    ax.set_yticks(range(len(ks)), [NAMES[k] for k in ks])
    ax.set_title("median lead, prefixgate minus baseline (tokens), test, αa=0.05 αd=0.1, 95% cluster bootstrap", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "lead_ci.png", dpi=140); plt.close(fig)

    # 4. mean trajectories of p(gold class) by gold label, as a function of the fraction of the call seen
    dm = jload("scores_dm.json.gz")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    grid = np.linspace(0, 1, 51)
    for lab, ls in (("allow", "-"), ("deny", "--"), ("escalate", ":")):
        ks_ = [k for k in pc if meta[k]["gold"] == lab]
        col = ("allow", "deny", "escalate").index(lab)
        Y = np.array([np.interp(grid, np.linspace(0, 1, meta[k]["T"]), np.array(dm[k])[:, col]) for k in ks_])
        ax.plot(grid, Y.mean(0), color="#f2f2f2", ls=ls, label=f"gold {lab} (n={len(ks_)})")
    ax.set_xlabel("fraction of call seen"); ax.set_ylabel("mean p(gold class)"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.set_title("mean p(gold class) as the call grows, test", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "trajectory.png", dpi=140); plt.close(fig)
    print("wrote", *(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
