# -*- coding: utf-8 -*-
"""
load_split — BẢN ĐÚNG THEO CODE GỐC máy bên kia (đã nhận source, 2026-08-17).

Quan trọng (đã phân tích):
  1. split_frac CHỈ dùng trong nhánh causal; với host_holdout nó KHÔNG ảnh hưởng
     (train = toàn bộ non-holdout hosts). Sự khác 0.6/0.7 trong metrics.json chỉ là
     field config còn sót, KHÔNG cần đồng bộ.
  2. _exclude_final_test loại (host,pid,ppid,ts) trùng final_test → test thật = 18 mal/506 ben.
  3. Undersample dùng random.Random(42) CỐ ĐỊNH — muốn nhiều seed thật phải đổi dòng này
     thành random.Random(cfg.seed) và thêm shuffle train theo seed.
"""
import json
import random
from typing import List, Tuple


class DataConfig:
    """Tối giản — thay bằng dataclass thật của pipeline."""
    dataset_path: str
    split_mode: str = "host_holdout"          # "host_holdout" | "causal"
    holdout_hosts: List[str] = None
    split_frac: float = 0.6
    neg_pos_ratio: int = 20
    seed: int = 42                          # NOTE: code gốc chưa dùng cho undersample
    final_test_path: str = None


def _load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _exclude_final_test(rows, final_test_path=None):
    """Loại dòng trùng (host,pid,ppid,ts) với final_test.jsonl."""
    if not final_test_path:
        return rows
    ft_ids = set()
    with open(final_test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)["meta"]
            ft_ids.add((m["host"], m["pid"], m["ppid"], round(m["ts"], 3)))
    return [r for r in rows
            if (r["meta"]["host"], r["meta"]["pid"], r["meta"]["ppid"],
                round(r["meta"]["ts"], 3)) not in ft_ids]


def load_split(cfg: DataConfig) -> Tuple[List[dict], List[dict]]:
    rows = _exclude_final_test(_load_rows(cfg.dataset_path), cfg.final_test_path)
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

        # Loai dong test trung text voi train (chong memorization)
        train_texts = set(r["text"] for r in train)
        test_clean = [r for r in test if r["text"] not in train_texts]
        n_dup = len(test) - len(test_clean)
        if n_dup:
            print(f"[data] loai {n_dup:,} dong test trung text voi train "
                  f"-> test={len(test_clean):,}")
        test = test_clean
    else:
        # CAUSAL: split_frac CHI ap dung o day
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
            # GOC: seed CỐ ĐỊNH 42. Muon nhieu seed that: random.Random(cfg.seed)
            rng = random.Random(42)
            neg = rng.sample(neg, keep_n)
            train = pos + neg
            train.sort(key=lambda r: r["meta"]["ts"])
            print(f"[data] undersample benign: {n_neg_before:,} -> {keep_n:,} "
                  f"({cfg.neg_pos_ratio}:1) -> train={len(train):,}")

    ptr = sum(r["label"] for r in train)
    pte = sum(r["label"] for r in test)
    print(f"[data] train doc={ptr}  test doc={pte}")
    if ptr == 0 or pte == 0:
        raise ValueError("Train hoac test thieu lop doc.")
    return train, test


if __name__ == "__main__":
    import os
    cfg = DataConfig()
    cfg.dataset_path = r"C:\Users\BDTG\OpTC-data\phase1\results\phase0_other_machine\train_data_frozen_v2.jsonl"
    cfg.holdout_hosts = ["SysClient0501.systemia.com"]
    cfg.final_test_path = r"C:\Users\BDTG\OpTC-data\phase1\results\phase0_other_machine\final_test.jsonl"
    tr, te = load_split(cfg)
    print(f"FINAL: train={len(tr)} doc={sum(r['label'] for r in tr)} "
          f"| test={len(te)} doc={sum(r['label'] for r in te)}")
