"""Static export of the dashboard for a free static Hugging Face Space: the server's answers for the recorded runs,
written as files next to app/index.html. Replays only; nothing is scored or generated in the browser.

python3 scripts/static_space.py            -> build/space/
python3 scripts/static_space.py --push R   -> also upload to the Space R (needs HF_TOKEN)
"""
import json, os, re, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "app")]
os.environ["PREFIXGATE_AGENT_LIVE"] = "0"
import server  # noqa: E402

OUT = ROOT / "build" / "space"


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "data" / "call").mkdir(parents=True)
    w = lambda name, obj: (OUT / "data" / name).write_text(json.dumps(obj, separators=(",", ":")))
    server.boot()
    s = server.summary()
    w("summary.json", s)
    w("agent.json", server.agent_runs())
    ids = set()
    for r in s["runs"]:
        a = f"{r['alpha_allow']},{r['alpha_deny']}"
        cs = server.calls(a)
        w(f"calls_{a}.json", cs)
        ids |= {c["id"] for c in cs}
    names = {k: re.sub(r"[^\w-]", "_", k) for k in ids}
    assert len(set(names.values())) == len(names), "file name collision"
    for k, n in names.items():
        w(f"call/{n}.json", server.call(k))
    html = (ROOT / "app" / "index.html").read_text().replace("<script>", "<script>window.STATIC=1;</script>\n<script>", 1)
    (OUT / "index.html").write_text(html)
    shutil.copy(ROOT / "space" / "README.md", OUT / "README.md")
    print(f"{OUT}: {len(names)} calls, {len(s['runs'])} risk levels, {len(server.agent_runs()['runs'])} agent runs")
    if "--push" in sys.argv:
        from huggingface_hub import create_repo, upload_folder
        repo = sys.argv[sys.argv.index("--push") + 1]
        create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
        print(upload_folder(repo_id=repo, repo_type="space", folder_path=OUT, commit_message="static replay of the recorded runs"))


if __name__ == "__main__":
    main()
