"""metrics.py — Thuoc do phat hien dung chung.

AUC_PR la metric CHINH (dat dau tien trong dict/output) — voi malicious chi ~2.64% du
lieu, AUC (ROC) de gay ao tuong vi phan lon duong cong nam o vung FPR cao it y nghia van
hanh; AUC_PR phan anh dung hon kha nang giu precision khi co gang bat nhieu recall tren
lop hiem. recall@FPR/MCC/F1 (o nguong da calibrate 0.5) la cac metric phu de doi chieu."""
from __future__ import annotations

import math
from typing import Dict, List


def recall_at_fpr(y: List[int], s: List[float], alpha: float) -> float:
    order = sorted(zip(s, y), reverse=True)
    pos = sum(y); neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    fp = tp = 0; best = 0.0
    for _sc, lab in order:
        if lab: tp += 1
        else: fp += 1
        if fp / neg <= alpha:
            best = max(best, tp / pos)
    return best


def mcc(y: List[int], s: List[float], thr: float = 0.5) -> float:
    tp = fp = tn = fn = 0
    for yi, si in zip(y, s):
        p = 1 if si >= thr else 0
        if yi and p: tp += 1
        elif not yi and p: fp += 1
        elif not yi and not p: tn += 1
        else: fn += 1
    num = tp * tn - fp * fn
    den = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return num / den if den > 0 else 0.0


def f1_at(y: List[int], s: List[float], thr: float = 0.5) -> float:
    tp = fp = fn = 0
    for yi, si in zip(y, s):
        p = 1 if si >= thr else 0
        if yi and p: tp += 1
        elif not yi and p: fp += 1
        elif yi and not p: fn += 1
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def all_metrics(y: List[int], s: List[float]) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score
    return {
        "AUC_PR": float(average_precision_score(y, s)),
        "MCC": mcc(y, s),
        "F1": f1_at(y, s),
        "recall@FPR1%": recall_at_fpr(y, s, 0.01),
        "recall@FPR0.1%": recall_at_fpr(y, s, 0.001),
        "AUC": float(roc_auc_score(y, s)),
    }


def print_metrics(m: Dict[str, float]) -> None:
    for k, v in m.items():
        print(f"  {k:<14}: {v:.3f}")
