"""Local dashboard: replays real scored calls from results/ and scores pasted calls live with the trained DM.

Nothing here executes a tool. A deny or escalate only changes what the page shows.
ponytail: one process, one lock around the model; fine for one local user, not for a shared deployment.
"""
import gzip, json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from prefixgate.data import build, rules  # noqa: E402

RES = ROOT / "results"
MAX_PASTE = 20000
state, lock = {}, threading.Lock()


def jl(name):
    with (gzip.open(RES / name, "rt") if name.endswith(".gz") else open(RES / name)) as f:
        return json.load(f)


def boot():
    from transformers import AutoTokenizer
    from prefixgate.dm import MODEL, REVISION
    metrics = jl("metrics.json")
    meta, dm, pr = jl("records.json"), jl("scores_dm.json.gz"), jl("scores_prompted.json.gz")
    recs = {r["id"]: r for r in build() if r["id"] in meta and r["id"] in dm}
    state.update(metrics=metrics, meta=meta, dm=dm, pr=pr, recs=recs,
                 tok=AutoTokenizer.from_pretrained(MODEL, revision=REVISION), rules=rules(),
                 logs=[json.loads(l) for f in ("train_log.jsonl", "score_log.jsonl") for l in open(RES / f)])


def run_for(alpha):
    for r in state["metrics"]["runs"]:
        if f"{r['alpha_allow']},{r['alpha_deny']}" == alpha:
            return r
    return state["metrics"]["runs"][-1]


def summary():
    m = state["metrics"]
    runs = [{k: v for k, v in r.items() if not k.endswith("per_call_test")} for r in m["runs"]]
    return dict({k: v for k, v in m.items() if k != "runs"}, runs=runs, logs=state["logs"],
                rules=[{k: r[k] for k in ("templateId", "name", "recommendedAction", "definition")} for r in state["rules"]])


def calls(alpha):
    run = run_for(alpha)
    pc, bc = run.get("per_call_test") or {}, run.get("batch_per_call_test") or {}
    out = []
    for k, r in state["recs"].items():
        if r["split"] != "test":
            continue
        mm = state["meta"][k]
        out.append(dict(id=k, gold=mm["gold"], attack=mm["attack"], position=mm["position"], T=mm["T"],
                        tool=json.loads(r["text"])["tool"],
                        stream=pc.get(k), batch=bc.get(k)))
    return out


def call(k):
    r, mm = state["recs"][k], state["meta"][k]
    tok = state["tok"]
    ids = tok(r["text"], add_special_tokens=False)["input_ids"]
    return dict(id=k, text=r["text"], tokens=[tok.decode([i]) for i in ids], probs=state["dm"][k],
                prompted=state["pr"].get(k), truth={a: b for a, b in r.get("truth", {}).items() if b not in (False, "excluded")},
                **mm)


def live(text):
    """Score a pasted call token by token with the trained DM (one batch pass == the KV-cache stream, see latency())."""
    import torch
    import torch.nn.functional as F
    from prefixgate.dm import MAX_TOKENS, dm_prompt, load
    with lock:
        if "model" not in state:
            state["model"] = load()[1]
        tok, m = state["tok"], state["model"]
        P = tok(dm_prompt(), add_special_tokens=False)["input_ids"]
        ids = tok(text, add_special_tokens=False)["input_ids"][:MAX_TOKENS]
        with torch.no_grad():
            h = m.model(torch.tensor([P + ids])).last_hidden_state[0, len(P):]
            probs = F.softmax(m.head(h), -1).tolist()
    return dict(tokens=[tok.decode([i]) for i in ids], probs=[[round(x, 4) for x in p] for p in probs], T=len(ids))


class H(BaseHTTPRequestHandler):
    def send(self, body, ctype="application/json", code=200):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/":
            return self.send((ROOT / "app" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/summary":
            return self.send(summary())
        if u.path == "/api/calls":
            return self.send(calls(q.get("alpha", "")))
        if u.path == "/api/call" and q.get("id") in state["recs"]:
            return self.send(call(q["id"]))
        self.send({"error": "not found"}, code=404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/live":
            return self.send({"error": "not found"}, code=404)
        n = int(self.headers.get("Content-Length", 0))
        if not 0 < n <= MAX_PASTE:
            return self.send({"error": f"paste 1..{MAX_PASTE} bytes"}, code=400)
        text = self.rfile.read(n).decode("utf-8", "replace")
        self.send(live(text))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    boot()
    print(f"prefixgate dashboard on http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
