"""Tao thu muc ket qua cho mot lan chay: results/<dataset>/<tag>/ (+ logs/, run.json).

Quy uoc: moi STT/thi nghiem co thu muc rieng, KHONG ghi thang vao results/.
  R=$(python newrun.py adgen stt60_lora_head)      # in ra duong dan
  python eve.py run ... --out $R/q05_REAL_eve.jsonl > $R/logs/eve.log 2>&1
  python newrun.py --list                          # liet ke cac run da co
run.json ghi commit cua thu muc code thuc su chay (--code, mac dinh thu muc chua newrun.py) va so file chua commit.
"""
import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

RESULTS = Path(__file__).with_name("results")
DATASETS = ("adgen", "attackdata", "optc")
TAG_RE = re.compile(r"^(stt\d+[a-z]?|task\d+[a-z]?|00)_[a-z0-9_]+$")


def git_state(code_dir):
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=code_dir, timeout=10).stdout
    try:
        head = git("rev-parse", "--short", "HEAD").strip() or None
        dirty = [l for l in git("status", "--porcelain", "--", ".").splitlines() if l.strip()]
        return head, len(dirty)
    except Exception:
        return None, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("tag", nargs="?", help="stt<so>|task<so>[a-z]_<mo_ta>, vd task04_cpu_15b")
    ap.add_argument("--note", default="")
    ap.add_argument("--code", default=str(Path(__file__).parent), help="thu muc code se chay (vd worktree)")
    ap.add_argument("--reuse", action="store_true", help="cho phep dung lai thu muc da ton tai")
    ap.add_argument("--new_dataset", action="store_true", help="cho phep dataset ngoai " + "/".join(DATASETS))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        for d in sorted(p for p in RESULTS.rglob("*") if p.is_dir() and p.name != "logs"):
            n = sum(1 for f in d.iterdir() if f.is_file())
            nl = sum(1 for f in (d / "logs").glob("*") if f.is_file()) if (d / "logs").is_dir() else 0
            if n or nl:
                print(f"{d.relative_to(RESULTS).as_posix():40s} {n:4d} file {nl:3d} log")
        return
    if not a.dataset or not a.tag:
        ap.error("can <dataset> <tag>")
    if a.dataset not in DATASETS and not a.new_dataset:
        ap.error(f"dataset '{a.dataset}' la moi; them --new_dataset neu dung")
    if not TAG_RE.match(a.tag):
        ap.error(f"tag '{a.tag}' sai quy uoc stt<so>|task<so>[a-z]_<mo_ta> (chu thuong, so, _)")
    run = RESULTS / a.dataset / a.tag
    if run.exists() and not a.reuse:
        sys.exit(f"{run.as_posix()} da ton tai; dung --reuse hoac tag khac (tranh ghi de)")
    (run / "logs").mkdir(parents=True, exist_ok=True)
    meta = run / "run.json"
    if not meta.exists():
        head, dirty = git_state(a.code)
        if dirty:
            print(f"WARN: {dirty} file chua commit trong {a.code}; ket qua khong truy duoc ve commit {head}",
                  file=sys.stderr)
        meta.write_text(json.dumps({"dataset": a.dataset, "tag": a.tag, "note": a.note, "git": head,
                                    "git_dirty": dirty, "code": Path(a.code).resolve().as_posix(),
                                    "created": datetime.datetime.now().isoformat(timespec="seconds")},
                                   indent=1), encoding="utf-8")
    print(run.as_posix())


if __name__ == "__main__":
    main()
