"""Tao thu muc ket qua cho mot lan chay: results/<dataset>/<tag>/ (+ logs/, run.json).

Quy uoc: moi STT/thi nghiem co thu muc rieng, KHONG ghi thang vao results/.
  R=$(python newrun.py adgen stt60_lora_head)      # in ra duong dan
  python eve.py run ... --out $R/q05_REAL_eve.jsonl > $R/logs/eve.log 2>&1
  python newrun.py --list                          # liet ke cac run da co
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
TAG_RE = re.compile(r"^(stt\d+[a-z]?|00)_[a-z0-9_]+$")


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, timeout=5).stdout.strip() or None
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("tag", nargs="?", help="stt<so>[a-z]_<mo_ta>, vd stt60_lora_head")
    ap.add_argument("--note", default="")
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
        ap.error(f"tag '{a.tag}' sai quy uoc stt<so>[a-z]_<mo_ta> (chu thuong, so, _)")
    run = RESULTS / a.dataset / a.tag
    if run.exists() and not a.reuse:
        sys.exit(f"{run.as_posix()} da ton tai; dung --reuse hoac tag khac (tranh ghi de)")
    (run / "logs").mkdir(parents=True, exist_ok=True)
    meta = run / "run.json"
    if not meta.exists():
        meta.write_text(json.dumps({"dataset": a.dataset, "tag": a.tag, "note": a.note, "git": git_head(),
                                    "created": datetime.datetime.now().isoformat(timespec="seconds")},
                                   indent=1), encoding="utf-8")
    print(run.as_posix())


if __name__ == "__main__":
    main()
