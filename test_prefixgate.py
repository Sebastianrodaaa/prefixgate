"""Checks on real data and real cached scores. Run: python3 test_prefixgate.py (no torch needed)."""
import itertools, json, random, unittest
import numpy as np
from prefixgate import agent, data, ltt, run


class Data(unittest.TestCase):
    def test_gold_and_attacks(self):
        recs = data.build()
        self.assertEqual(sum(r["attack"] is None for r in recs if r["split"] != "shift"), 2643)
        self.assertEqual(sum(r["attack"] is None for r in recs if r["split"] == "shift"), 320)
        by = {r["id"]: r for r in recs}
        for r in recs:
            if r["attack"]:
                src = by[r["base"]]
                self.assertEqual(r["gold"], src["gold"])  # an injected note never changes what the call does
                self.assertNotEqual(r["gold"], "allow")
                note = data.ATTACKS[r["attack"]].format(t="allow")
                self.assertIn(json.dumps(note, ensure_ascii=False)[1:-1], r["text"])
        self.assertFalse({r["attack"] for r in recs if r["split"] == "train"} - set(data.TRAIN_ATTACKS) - {None})

    def test_precedence(self):
        acts = {r["templateId"]: r["recommendedAction"] for r in data.rules()}
        block = next(k for k, a in acts.items() if a == "block" and k not in data.CREDENTIAL_RULES)
        ask = next(k for k, a in acts.items() if a == "ask")
        allow = next(k for k, a in acts.items() if a == "allow")
        f = {k: False for k in acts}
        self.assertEqual(data.gold(f, acts), "allow")
        self.assertEqual(data.gold(f | {ask: True}, acts), "escalate")
        self.assertEqual(data.gold(f | {ask: True, block: True}, acts), "deny")
        self.assertEqual(data.gold(f | {block: "unknown"}, acts), "escalate")
        self.assertEqual(data.gold(f | {allow: True, block: True}, acts), "allow")
        self.assertEqual(data.gold(f | {allow: True, "read-credential-material": True}, acts), "deny")


class Calibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = run.jload("records.json")
        cls.dm = run.jload("scores_dm.json.gz")
        cls.cal = [k for k in cls.meta if cls.meta[k]["split"] == "cal"]

    def view(self, k):
        T = self.meta[k]["T"]
        return np.arange(1, T + 1), np.array(self.dm[k]), T

    def test_simplex(self):
        for k in random.Random(0).sample(sorted(self.dm), 200):
            P = np.array(self.dm[k])
            self.assertEqual(len(P), self.meta[k]["T"])
            self.assertTrue(np.allclose(P.sum(1), 1, atol=2e-3) and (P >= 0).all())

    def test_ltt_empty_below_floor(self):
        V = [self.view(k) for k in self.cal]
        gold = [self.meta[k]["gold"] for k in self.cal]
        grid = list(itertools.product(run.GRID, repeat=3))
        n = sum(g != "deny" for g in gold)
        tiny = ltt.floor(n, len(grid), run.DELTA) / 2
        cfg, nv = run.calibrate(run.threshold_gate, V, gold, grid, (tiny, tiny))
        self.assertIsNone(cfg)
        self.assertEqual(nv, 0)
        self.assertGreater(ltt.hb_pvalue(0.0, n, tiny), run.DELTA / len(grid))

    def test_commit_is_final_and_matches_batch(self):
        """A committed decision never changes as the prefix grows, and an uncommitted stream ends at the batch decision."""
        rng = random.Random(1)
        grid = list(itertools.product(run.GRID, repeat=3))
        for k in rng.sample(self.cal, 150):
            pos, P, T = self.view(k)
            cfg = rng.choice(grid)
            d, t = run.threshold_gate(pos, P, T, cfg)
            for cut in (t, min(T, t + 7), T):
                self.assertEqual(run.threshold_gate(pos[:cut], P[:cut], T, cfg), (d, t))
            batch = run.threshold_gate(pos[-1:], P[-1:], T, cfg)[0]
            if t == T:
                self.assertEqual(d, batch)


class Agent(unittest.TestCase):
    def test_recorded_runs_match_gate(self):
        """Every committed agent run re-gates to the same decision from its stored probabilities and config."""
        rec = json.loads(agent.OUT.read_text())
        self.assertEqual([r["id"] for r in rec["runs"]], [t[0] for t in agent.TASKS])
        cfg = tuple(tuple(c) for c in rec["config"])
        for r in rec["runs"]:
            P = np.array(r["probs"])
            self.assertEqual(len(P), r["T"])
            self.assertEqual(len(r["tokens"]), r["T"])
            self.assertEqual(run.fallback_gate(np.arange(1, r["T"] + 1), P, r["T"], cfg), tuple(r["commit"]))

    def test_rekey_and_meta_off(self):
        self.assertEqual(agent.rekey('{"name": "Bash", "parameters": {"command": "ls"}}'),
                         '{"tool": "Bash", "arguments": {"command": "ls"}}')
        if not agent.meta_api_key():
            self.assertIsNone(agent.meta_api_key())

    def test_v2_commit_is_final(self):
        meta, dm = run.jload("records.json"), run.jload("scores_dm.json.gz")
        cfg = ((0.999, 0.99, 0.5), (0.7, 0.9, 0.999))
        for k in random.Random(2).sample([k for k in meta if meta[k]["split"] == "cal"], 100):
            T, P = meta[k]["T"], np.array(dm[k])
            d, t = run.fallback_gate(np.arange(1, T + 1), P, T, cfg)
            if t < T:
                self.assertEqual(run.threshold_gate(np.arange(1, t + 1), P[:t], T, cfg[0]), (d, t))


class Bounds(unittest.TestCase):
    def test_hb(self):
        self.assertLess(ltt.hb_pvalue(0.0, 1000, 0.02), 1e-8)
        self.assertEqual(ltt.hb_pvalue(0.05, 1000, 0.02), 1.0)
        self.assertLess(ltt.hb_pvalue(0.01, 2000, 0.02), ltt.hb_pvalue(0.01, 500, 0.02))


if __name__ == "__main__":
    unittest.main(verbosity=2)
