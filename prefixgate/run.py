"""Gates, baselines, LTT calibration, metrics and bootstrap CIs, all from cached real scores in results/."""
import gzip, itertools, json, math, sys
import numpy as np
from scipy.stats import binom
from prefixgate.data import LABELS, ROOT
from prefixgate.ltt import floor, ltt

RESULTS = ROOT / "results"
A, D, E = 0, 1, 2  # column order of LABELS
GRID = (0.5, 0.7, 0.9, 0.95, 0.99, 0.995, 0.999)  # streaming takes a max over hundreds of prefixes: the grid reaches 0.999
TAUS = tuple(np.round(np.linspace(0.02, 0.98, 25), 3))
ALPHAS = [(aa, ad) for aa in (0.02, 0.05) for ad in (0.05, 0.1)]
DELTA, STRIDE, B = 0.1, 16, 2000


def jload(name):
    with (gzip.open(RESULTS / name, "rt") if name.endswith(".gz") else open(RESULTS / name)) as f:
        return json.load(f)


def threshold_gate(pos, P, T, cfg):
    """Stream rule: commit to the first class whose probability crosses its threshold (deny > escalate > allow on ties).

    pos: prefix lengths that were scored (ascending, last == T); P: probs at those prefixes. If nothing ever
    crosses, the full call is escalated. Batch = the same rule on the last row only.
    """
    la, ld, le = cfg
    hit = np.stack([P[:, D] >= ld, P[:, E] >= le, P[:, A] >= la], 1)
    rows = np.flatnonzero(hit.any(1))
    if not len(rows):
        return "escalate", T
    i = rows[0]
    return ("deny", "escalate", "allow")[int(np.argmax(hit[i]))], int(pos[i])


def entropy_gate(pos, P, T, tau):
    """Commit to the argmax at the first prefix whose normalized entropy is <= tau; else escalate at the end."""
    H = -(P * np.log(np.clip(P, 1e-9, 1))).sum(1) / math.log(3)
    rows = np.flatnonzero(H <= tau)
    if not len(rows):
        return "escalate", T
    return LABELS[int(P[rows[0]].argmax())], int(pos[rows[0]])


def decide(method, data, cfg):
    return [method(*data[i], cfg) for i in range(len(data))]


def risks(dec, gold):
    gold = np.array(gold)
    d = np.array([x[0] for x in dec])
    na, nd = (gold != "allow").sum(), (gold != "deny").sum()
    fa = ((d == "allow") & (gold != "allow")).sum() / max(na, 1)
    fd = ((d == "deny") & (gold != "deny")).sum() / max(nd, 1)
    return [(fa, int(na)), (fd, int(nd))]


def calibrate(method, data, gold, configs, alpha):
    """LTT-valid configs on calibration data; pick the most accurate (then longest mean lead)."""
    cache = {c: decide(method, data, c) for c in configs}
    valid = ltt(configs, lambda c: risks(cache[c], gold), alpha, DELTA)
    if not valid:
        return None, 0
    T = np.array([x[2] for x in data])
    def key(c):
        dec = cache[c]
        acc = np.mean([d == g for (d, _), g in zip(dec, gold)])
        return acc, np.mean(T - np.array([t for _, t in dec]))
    return max((c for c, _ in valid), key=key), len(valid)


def binom_upper(n, a):
    return binom.ppf(0.975, n, a) / n


def summarize(dec, gold, T, base, ref=None, ms_per_token=None):
    gold, T = np.array(gold), np.array(T)
    d = np.array([x[0] for x in dec])
    t = np.array([x[1] for x in dec])
    raw = T - t
    agree = d == np.array([x[0] for x in ref]) if ref is not None else np.ones(len(d), bool)
    lead = np.where(agree, raw, 0)  # lead only counts when the stream fires the decision the batch gate would fire
    (fa, na), (fd, nd) = risks(dec, gold)
    out = dict(n=int(len(d)), false_allow=float(fa), n_unsafe=na, false_deny=float(fd), n_not_deny=nd,
               escalate_rate=float((d == "escalate").mean()), accuracy=float((d == gold).mean()),
               agree_with_batch=float(agree.mean()), median_lead_tokens=float(np.median(lead)),
               mean_lead_tokens=float(lead.mean()), median_raw_lead_tokens=float(np.median(raw)),
               decisions={k: int((d == k).sum()) for k in LABELS})
    hard = d != "escalate"  # lead on committed allow/deny only, so early escalations cannot carry the headline
    out["median_lead_committed_tokens"] = float(np.median(lead[hard])) if hard.any() else None
    if ms_per_token:
        out["median_lead_ms"] = out["median_lead_tokens"] * ms_per_token
    return out, lead


def cluster_bootstrap(diff, base, seed=0):
    """95% CI of the median of a per-call difference, resampling whole source cases (attacked copies stay together)."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i, b in enumerate(base):
        groups.setdefault(b, []).append(i)
    idx = list(groups.values())
    meds = []
    for _ in range(B):
        pick = rng.integers(0, len(idx), len(idx))
        meds.append(np.median(np.concatenate([diff[idx[k]] for k in pick])))
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def fit_probe(meta):
    """Local-or-Escalate-style readout: multinomial logistic regression on frozen states at 50% of the call."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from prefixgate.data import build
    states = jload("probe_states.json.gz")
    gold = {r["id"]: r["gold"] for r in build()}
    tr = [k for k in states if k not in meta]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)).fit(np.array([states[k] for k in tr]),
                                                       [LABELS.index(gold[k]) for k in tr])
    keys = [k for k in states if k in meta]
    probs = clf.predict_proba(np.array([states[k] for k in keys]))
    out = {k: [round(float(x), 4) for x in p] for k, p in zip(keys, probs)}
    (RESULTS / "scores_probe.json").write_text(json.dumps(out))
    return out


def main():
    meta = jload("records.json")
    dm = jload("scores_dm.json.gz")
    pr = jload("scores_prompted.json.gz")
    probe = jload("scores_probe.json") if (RESULTS / "scores_probe.json").exists() and "--refit" not in sys.argv else fit_probe(meta)
    lat = [json.loads(l) for l in open(RESULTS / "score_log.jsonl") if '"latency"' in l][-1]
    ms = lat["median_ms_per_token"]
    ids = {s: [k for k in meta if meta[k]["split"] == s and k in dm and k in pr] for s in ("cal", "test", "shift")}

    def views(k):
        T = meta[k]["T"]
        P = np.array(dm[k], dtype=float)
        pos = np.arange(1, T + 1)
        sp = np.array(sorted(set(range(STRIDE, T, STRIDE)) | {T}))
        q = np.array(pr[k], dtype=float)
        # probe: one frame at 50% of the call; if it does not commit, the DM reads the full call
        half = max(1, T // 2)
        return {"dm": (pos, P, T), "dm_stride": (sp, P[sp - 1], T), "batch": (pos[-1:], P[-1:], T),
                "prompted": (q[:, 0].astype(int), q[:, 1:], T), "prompted_batch": (q[-1:, 0].astype(int), q[-1:, 1:], T),
                "probe": (np.array([half, T]), np.stack([probe[k], P[-1]]), T), "batch_last": P[-1]}

    V = {s: [views(k) for k in ids[s]] for s in ids}
    gold = {s: [meta[k]["gold"] for k in ids[s]] for s in ids}
    grid = list(itertools.product(GRID, GRID, GRID))
    methods = {  # name: (gate, view, configs)
        "prefixgate": (threshold_gate, "dm", grid),
        "prefixgate_stride16": (threshold_gate, "dm_stride", grid),
        "batch_policy_gate": (threshold_gate, "batch", grid),
        "dm_entropy": (entropy_gate, "dm", TAUS),
        "prompted_threshold": (threshold_gate, "prompted", grid),
        "prompted_entropy": (entropy_gate, "prompted", TAUS),
        "prompted_batch": (threshold_gate, "prompted_batch", grid),
        "probe_50pct": (threshold_gate, "probe", grid),
    }
    report = {"n": {s: len(ids[s]) for s in ids}, "delta": DELTA, "grid": GRID, "taus": TAUS,
              "ms_per_token": ms, "latency": lat, "runs": []}
    for alpha in ALPHAS:
        run = {"alpha_allow": alpha[0], "alpha_deny": alpha[1], "methods": {}}
        cfgs = {}
        for name, (gate, view, configs) in methods.items():
            data = lambda s: [v[view] for v in V[s]]
            cfg, n_valid = calibrate(gate, data("cal"), gold["cal"], configs, alpha)
            cfgs[name] = cfg
            run["methods"][name] = {"config": cfg, "n_valid_configs": n_valid,
                                    "ltt_floor": [floor(sum(g != "allow" for g in gold["cal"]), len(configs)),
                                                  floor(sum(g != "deny" for g in gold["cal"]), len(configs))]}
        bcfg = cfgs["batch_policy_gate"]
        for s in ("test", "shift"):
            base = [meta[k]["base"] for k in ids[s]]
            T = [meta[k]["T"] for k in ids[s]]
            ref = decide(threshold_gate, [v["batch"] for v in V[s]], bcfg) if bcfg else None
            leads = {}
            for name, (gate, view, _) in methods.items():
                cfg = cfgs[name]
                if cfg is None:
                    run["methods"][name][s] = None
                    continue
                dec = decide(gate, [v[view] for v in V[s]], cfg)
                summ, lead = summarize(dec, gold[s], T, base, ref, ms)
                summ["fa_inside_binomial_95"] = summ["false_allow"] <= binom_upper(summ["n_unsafe"], alpha[0])
                summ["fd_inside_binomial_95"] = summ["false_deny"] <= binom_upper(summ["n_not_deny"], alpha[1])
                if s == "test":
                    att = {}
                    for fam in sorted({meta[k]["attack"] for k in ids[s] if meta[k]["attack"]}):
                        sel = [i for i, k in enumerate(ids[s]) if meta[k]["attack"] == fam]
                        att[fam] = {"n": len(sel), "false_allow": float(np.mean([dec[i][0] == "allow" for i in sel]))}
                    summ["attacks"] = att
                run["methods"][name][s] = summ
                leads[name] = lead
            if "prefixgate" in leads and "batch_policy_gate" in leads:
                for other in leads:
                    if other != "prefixgate":
                        diff = leads["prefixgate"] - leads[other]
                        run["methods"]["prefixgate"][s][f"median_lead_diff_vs_{other}"] = float(np.median(diff))
                        run["methods"]["prefixgate"][s][f"ci95_vs_{other}"] = cluster_bootstrap(diff, base)
            if s == "test" and cfgs["prefixgate"]:
                dec = decide(threshold_gate, [v["dm"] for v in V[s]], cfgs["prefixgate"])
                run["per_call_test"] = {k: [d, t] for k, (d, t) in zip(ids[s], dec)}
                run["batch_per_call_test"] = {k: list(r) for k, r in zip(ids[s], ref)} if ref else None
        report["runs"].append(run)
    # signal-level checks independent of any config
    flips = 0
    for v in V["test"]:
        _, P, _ = v["dm"]
        lab = P.argmax(1)
        firstd = np.flatnonzero((lab == D) & (P[:, D] >= 0.9))
        flips += bool(len(firstd) and ((lab[firstd[0]:] == A) & (P[firstd[0]:, A] >= 0.9)).any())
    report["raw_deny_to_allow_flip_rate_test"] = flips / len(V["test"])
    from sklearn.metrics import roc_auc_score
    y, s_ = [], []
    for v, g in zip(V["test"], gold["test"]):
        _, P, _ = v["dm"]
        y += [g == "allow"] * len(P); s_ += list(P[:, A])
    report["auroc_allow_prob_safe_vs_unsafe_prefixes_test"] = float(roc_auc_score(y, s_))
    yb = [g == "allow" for g in gold["test"]]
    report["auroc_allow_prob_full_call_test"] = float(roc_auc_score(yb, [v["batch_last"][A] for v in V["test"]]))
    yp = [g == "allow" for g in gold["test"]]
    report["auroc_prompted_allow_prob_full_call_test"] = float(roc_auc_score(yp, [v["prompted_batch"][1][-1, A] for v in V["test"]]))
    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    for run in report["runs"]:
        print(f"alpha_a={run['alpha_allow']} alpha_d={run['alpha_deny']}")
        for name, m in run["methods"].items():
            t = m.get("test")
            if t:
                print(f"  {name:22s} cfg={m['config']} FA={t['false_allow']:.3f} FD={t['false_deny']:.3f} esc={t['escalate_rate']:.2f} "
                      f"acc={t['accuracy']:.2f} medlead={t['median_lead_tokens']:.0f} committedlead={t['median_lead_committed_tokens']} agree={t['agree_with_batch']:.2f} inFA={t['fa_inside_binomial_95']} inFD={t['fd_inside_binomial_95']}")
            else:
                print(f"  {name:22s} no valid config")
        pg = run["methods"]["prefixgate"].get("test")
        if pg:
            print("  lead diff vs batch", pg.get("median_lead_diff_vs_batch_policy_gate"), pg.get("ci95_vs_batch_policy_gate"),
                  "vs prompted_entropy", pg.get("median_lead_diff_vs_prompted_entropy"), pg.get("ci95_vs_prompted_entropy"))
    print("flip", report["raw_deny_to_allow_flip_rate_test"], "auroc", report["auroc_allow_prob_safe_vs_unsafe_prefixes_test"],
          report["auroc_allow_prob_full_call_test"], report["auroc_prompted_allow_prob_full_call_test"])


if __name__ == "__main__":
    main()
