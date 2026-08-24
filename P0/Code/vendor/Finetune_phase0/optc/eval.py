"""eval.py — Batched inference tren test set: tra ve scores/labels/latency trung binh.

Dung chung boi train.py va baseline_encoder.py thay vi moi noi tu goi model tung dong
mot (cham hon nhieu lan tren test set 1 vai chuc nghin dong, khong tan dung duoc song
song cua GPU/CPU batch).
"""
from __future__ import annotations

import time
from typing import List, Tuple

import torch


def score_rows(
    model,
    tokenizer,
    rows: List[dict],
    max_seq_length: int,
    batch_size: int = 32,
) -> Tuple[List[float], List[int], float]:
    """Chay inference theo batch tren `rows` (moi row can key 'text','label').
    Tra ve (scores, labels, latency_ms_trung_binh_moi_mau)."""
    device = next(model.parameters()).device
    labels = [int(r["label"]) for r in rows]
    texts = [r["text"] for r in rows]
    scores: List[float] = []

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, truncation=True, max_length=max_seq_length,
                             padding=True, return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1]
            scores.extend(probs.tolist())
    infer_ms = 1000 * (time.time() - t0) / max(len(rows), 1)
    return scores, labels, infer_ms
