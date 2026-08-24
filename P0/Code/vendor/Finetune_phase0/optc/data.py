"""data.py — Doc JSONL {text,label,meta:{host,ts,size,...}}, chia CAUSAL (hoac
host-holdout cho RQ5).

Khac ban goc: bo chat-template/instruction-output. Input la chuoi token telemetry + nhan 0/1.
Chia theo THOI GIAN (khong shuffle) de khong nhin tuong lai. Chi `text`/`label` dua vao
model; `meta` (host/ts/pid/ppid/size/...) chi dung de split/truy vet, khong phai feature.
"""
from __future__ import annotations

import os
import random
from typing import List, Tuple

from config import DataConfig
from run_utils import read_jsonl

# Lat cat final-test dong bang boi make_final_test.py — KHONG BAO GIO duoc lot vao
# train/dev, du dataset_path la file nao. Neu file nay ton tai, load_split() tu dong
# loai cac dong trung khoa (host,pid,ppid,ts) khoi ket qua tra ve.
FINAL_TEST_PATH = r"E:\dataset\processed\final_test.jsonl"


def _load_rows(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Khong thay file: '{path}'. Chay export_dataset.py truoc de tao JSONL "
            f"{{text,label,meta:{{host,ts,size}}}}.")
    rows = read_jsonl(path)
    for k in ("text", "label", "meta"):
        if k not in rows[0]:
            raise ValueError(f"Record thieu key '{k}'. Co: {list(rows[0].keys())}")
    if "ts" not in rows[0]["meta"]:
        raise ValueError(f"meta thieu key 'ts'. Co: {list(rows[0]['meta'].keys())}")
    return rows


def _row_key(r: dict):
    m = r["meta"]
    return (m.get("host"), m.get("pid"), m.get("ppid"), m.get("ts"))


def _exclude_final_test(rows: List[dict]) -> List[dict]:
    if not os.path.exists(FINAL_TEST_PATH):
        return rows
    final_keys = {_row_key(r) for r in read_jsonl(FINAL_TEST_PATH)}
    kept = [r for r in rows if _row_key(r) not in final_keys]
    n_removed = len(rows) - len(kept)
    if n_removed:
        print(f"[data] loai {n_removed:,} dong trung final_test.jsonl (bao ve tinh "
              f"toan ven danh gia) khoi tap dang load")
    return kept


def load_split(cfg: DataConfig) -> Tuple[List[dict], List[dict]]:
    rows = _exclude_final_test(_load_rows(cfg.dataset_path))
    n_pos = sum(r["label"] for r in rows)
    print(f"[data] {len(rows):,} mau, doc={n_pos:,} ({100*n_pos/max(len(rows),1):.2f}%)")

    if cfg.split_mode == "host_holdout":
        hold = set(cfg.holdout_hosts or [])
        if not hold:
            raise ValueError("split_mode=host_holdout nhung holdout_hosts rong.")
        train = [r for r in rows if r["meta"].get("host") not in hold]
        test = [r for r in rows if r["meta"].get("host") in hold]
        print(f"[data] host-holdout: train={len(train):,} test={len(test):,} "
              f"(test hosts={sorted(hold)})")

        # Nhieu process he thong/agent giam sat sinh ra command line GIONG HET NHAU
        # tren MOI host (vd lwabeat.exe voi cung tham so) -> neu dong text nay vua o
        # train vua o test, test khong con do KHA NANG TONG QUAT HOA sang host la nua,
        # ma do KHA NANG GHI NHO chuoi da thay luc train. Loai het cac dong test trung
        # text voi train de test phan anh dung host chua tung thay.
        train_texts = set(r["text"] for r in train)
        test_clean = [r for r in test if r["text"] not in train_texts]
        n_dup = len(test) - len(test_clean)
        if n_dup:
            print(f"[data] loai {n_dup:,} dong test trung text voi train (chong "
                  f"memorization, chi giu dong THAT SU chua tung thay) "
                  f"-> test={len(test_clean):,}")
        test = test_clean
    else:
        # CAUSAL: sort theo thoi gian, train=qua khu, test=tuong lai
        rows.sort(key=lambda r: r["meta"]["ts"])
        cut = int(len(rows) * cfg.split_frac)
        train, test = rows[:cut], rows[cut:]
        print(f"[data] causal split @ {cfg.split_frac}: train={len(train):,} test={len(test):,}")

    if cfg.neg_pos_ratio is not None:
        pos = [r for r in train if r["label"] == 1]
        neg = [r for r in train if r["label"] == 0]
        n_neg_before = len(neg)
        keep_n = int(len(pos) * cfg.neg_pos_ratio)
        if keep_n < n_neg_before:
            neg = random.Random(cfg.seed).sample(neg, keep_n)
            train = pos + neg
            train.sort(key=lambda r: r["meta"]["ts"])  # giu thu tu thoi gian, gon debug
            print(f"[data] undersample benign trong train: {n_neg_before:,} -> {keep_n:,} "
                  f"(ty le {cfg.neg_pos_ratio}:1) -> train={len(train):,}")

    ptr = sum(r["label"] for r in train)
    pte = sum(r["label"] for r in test)
    print(f"[data] train doc={ptr}  test doc={pte}")
    if ptr == 0 or pte == 0:
        raise ValueError("Train hoac test thieu lop doc. Kiem tra nhan/thoi gian/split.")
    return train, test
