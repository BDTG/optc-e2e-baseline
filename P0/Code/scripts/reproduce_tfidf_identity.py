# -*- coding: utf-8 -*-
"""Reproduce TF-IDF — dung identity (host,pid,ppid,ts) de tach dev-test 18/506
dung nhu make_final_test.py ben kia (tru 131 dong final_test cu the)."""
import json
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    auc, average_precision_score, matthews_corrcoef, f1_score, roc_curve,
)

TRAIN = "train_data_frozen_v2.jsonl"
TEST = "final_test.jsonl"
HOLDOUT = "SysClient0501.systemia.com"

def load(path, exclude_host=None, only_host=None):
    texts, labels, metas = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if exclude_host and r["meta"]["host"] == exclude_host:
                continue
            if only_host and r["meta"]["host"] != only_host:
                continue
            texts.append(r["text"])
            labels.append(r["label"])
            metas.append(r["meta"])
    return texts, np.array(labels), metas

X_tr, y_tr, _ = load(TRAIN, exclude_host=HOLDOUT)
X_ft, y_ft, m_ft = load(TEST)

# identity set cua final_test
ft_ids = set()
for m in m_ft:
    ft_ids.add((m["host"], m["pid"], m["ppid"], round(m["ts"], 3)))

X_dev, y_dev_raw, m_dev = load(TRAIN, only_host=HOLDOUT)
mask = np.array([(m["host"], m["pid"], m["ppid"], round(m["ts"], 3)) not in ft_ids for m in m_dev])
X_te = [t for t, m in zip(X_dev, mask) if m]
y_te = y_dev_raw[mask]
print(f"train={len(X_tr)} (mal={y_tr.sum()}) devtest={len(X_te)} (mal={y_te.sum()}) finaltest={len(X_ft)} (mal={y_ft.sum()})")

pos_idx = np.where(y_tr == 1)[0]
neg_idx = np.where(y_tr == 0)[0]
rng = np.random.RandomState(42)
neg_sampled = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 20), replace=False)
keep = np.concatenate([pos_idx, neg_sampled])
X_tr, y_tr = [X_tr[i] for i in keep], y_tr[keep]

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=200000)
Xv_tr = vec.fit_transform(X_tr).toarray()
Xv_te = vec.transform(X_te).toarray()
print(f"vocab={Xv_tr.shape[1]} dense={Xv_tr.shape}")

clf = HistGradientBoostingClassifier(max_iter=300, early_stopping=False, random_state=42)
t1 = time.time()
clf.fit(Xv_tr, y_tr)
train_sec = time.time() - t1
proba = clf.predict_proba(Xv_te)[:, 1]

fpr, tpr, _ = roc_curve(y_te, proba)
auc_val = auc(fpr, tpr)
ap = average_precision_score(y_te, proba)

def recall_at_fpr(proba, y, fpr_target):
    fpr, tpr, th = roc_curve(y, proba)
    idx = np.argmin(np.abs(fpr - fpr_target))
    return tpr[idx]

pred = (proba >= 0.5).astype(int)
mcc = matthews_corrcoef(y_te, pred)
f1 = f1_score(y_te, pred)
rec1, rec01 = recall_at_fpr(proba, y_te, 0.01), recall_at_fpr(proba, y_te, 0.001)

print(f"=== TF-IDF + HistGBM reproduce dev-test (identity split) ===")
print(f"AUC={auc_val:.4f} AUC_PR={ap:.4f} recall@FPR1%={rec1:.4f} recall@FPR0.1%={rec01:.4f} MCC={mcc:.4f} F1={f1:.4f} train={train_sec:.1f}s")
print(f"(ben kia: AUC 0.9433 APR 0.9167 rec1% 0.8333 rec0.1% 0.8333 MCC 0.8526 F1 0.8571 train 23s)")
