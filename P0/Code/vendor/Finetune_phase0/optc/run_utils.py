"""run_utils.py — Helper thuan thu vien chuan (KHONG import torch/transformers): doc
JSONL va quan ly thu muc artifact moi lan chay. Tach rieng khoi utils.py de
baseline_tfidf.py (baseline re nhat, khong can GPU) khong bi keo theo torch/transformers
chi vi goi data.py (ma data.py can doc JSONL).

Dung boi baseline_tfidf.py + data.py truc tiep, va boi utils.py (re-export) cho cac
script can torch.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List

RUNS_ROOT = Path(__file__).parent / "runs"


def file_sha256(path: str) -> str:
    """Bam SHA256 noi dung file dataset — luu vao metrics.json de sau nay doi chieu
    duoc 2 lan chay co thuc su dung CUNG mot du lieu hay khong (ten file giong nhau
    KHONG dam bao noi dung giong nhau neu file bi ghi de/tai tao giua cac lan chay)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        return True  # khong co psutil -> gia dinh con song, thoi khong dong gi (an toan)


def cleanup_empty_runs() -> int:
    """Xoa cac run_dir CU (khong phai lan chay nay) ma RONG (chua co metrics.json,
    tuc script bi dung/lỗi giua chung truoc khi ghi ket qua) VA process tao ra no
    KHONG con song (doc PID tu _pid.txt) — de khong bao gio xoa nham 1 job dang chay
    du no chay bao lau. Goi tu dong moi lan make_run_dir(). Tra ve so run_dir da xoa."""
    if not RUNS_ROOT.exists():
        return 0
    n_removed = 0
    for d in RUNS_ROOT.iterdir():
        if not d.is_dir() or (d / "metrics.json").exists():
            continue
        pid_file = d / "_pid.txt"
        if not pid_file.exists():
            continue  # run cu tu ban truoc khi co co che nay -> khong dong, de an toan
        try:
            pid = int(pid_file.read_text().strip())
        except (ValueError, OSError):
            continue
        if _pid_alive(pid):
            continue  # dang chay that -> khong dong vao
        import shutil
        shutil.rmtree(d, ignore_errors=True)
        n_removed += 1
    return n_removed


def read_jsonl(path: str) -> List[dict]:
    """Doc toan bo JSONL {text,label,meta:{host,ts,...}} vao list."""
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_run_dir(script: str, **tags) -> Path:
    """Tao optc/runs/<script>__<tag1>_<tag2>..._<YYYYMMDD-HHMMSS>/ va tra ve Path.
    Dung de moi lan chay (train.py/baseline_encoder.py/baseline_tfidf.py/measure_cpu.py)
    co 1 thu muc rieng chua checkpoint/scores/metrics — thay vi ghi de len cung
    ./slm-ckpt, ./encoder-ckpt, scores.jsonl, results.jsonl giua cac lan chay khac nhau.

    vd: make_run_dir("baseline_encoder", model="ModernBERT-base", split="host_holdout")
        -> optc/runs/baseline_encoder__ModernBERT-base_host_holdout_20260808-014500/
    """
    n_cleaned = cleanup_empty_runs()
    if n_cleaned:
        print(f"[run] don {n_cleaned} run_dir rong cua lan chay truoc bi dung giua chung", flush=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag_str = "_".join(str(v) for v in tags.values() if v)
    run_id = f"{script}__{tag_str}_{ts}" if tag_str else f"{script}__{ts}"
    run_id = re.sub(r"[^\w.-]", "-", run_id)  # an toan lam ten thu muc tren moi OS
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "_pid.txt").write_text(str(os.getpid()))
    return run_dir


def save_run_metrics(run_dir: Path, **fields) -> None:
    """Ghi metrics.json vao run_dir — 1 cho duy nhat de biet lan chay nay dung
    model/split/quant gi va ra ket qua gi, khong phai doi chieu qua log."""
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(fields, f, indent=2, ensure_ascii=False)
