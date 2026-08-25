"""
Quick TF-IDF baseline run on enriched alerts (no torch needed).
Input: alerts_enriched_partial.jsonl + gt_and_scores.json
Output: FP reduction metrics
"""
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
alerts = [json.loads(l) for l in open("C:/Users/BDTG/OpTC-data/phase1/gh_repo/P1/Output/alerts_enriched_partial.jsonl")]
gt_data = json.load(open("C:/Users/BDTG/OpTC-data/phase1/gh_repo/P1/Output/gt_and_scores.json"))
gt_nids = set(gt_data["gt_nids"])
print(f"Alerts: {len(alerts)}, GT nodes: {len(gt_nids)}")

# Features: self_label + parent_chain ops + event_seq ops
texts, labels = [], []
for a in alerts:
    parts = [a.get("self_label", "")]
    for p in a.get("parent_chain", []):
        parts.append(f"{p.get('op','')} {p.get('node','')}")
    for e in a.get("event_seq", [])[:10]:
        parts.append(f"{e.get('src','')} {e.get('op','')} {e.get('dst','')}")
    texts.append(" ".join(parts))
    labels.append(1 if str(a.get("nid")) in gt_nids else 0)

n_pos = sum(labels)
n_neg = len(labels) - n_pos
print(f"Positive: {n_pos}, Negative: {n_neg}")

# TF-IDF char n-gram
tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=50000, sublinear_tf=True)
X = tfidf.fit_transform(texts)
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X, labels)
scores = clf.predict_proba(X)[:, 1]

# Results at various k
print("\n=== TF-IDF + LogReg baseline (NO SLM) ===")
print(f"{'k':>6} | {'TP_base':>7} | {'FP_base':>7} | {'prec_base':>9} | {'TP_slm':>6} | {'FP_slm':>6} | {'prec_slm':>8} | {'FP_red':>7}")
for k in [500, 1000, 2000]:
    order = np.argsort(-scores)[:k]
    tp_b = int(sum(labels[i] for i in order))
    fp_b = k - tp_b
    # threshold 0.5
    tp_a = int(sum(1 for i in order if scores[i] > 0.5 and labels[i] == 1))
    fp_a = int(sum(1 for i in order if scores[i] > 0.5 and labels[i] == 0))
    filt = int(sum(1 for i in order if scores[i] <= 0.5))
    prec_b = tp_b / max(k, 1)
    prec_a = tp_a / max(tp_a + fp_a, 1) if (tp_a + fp_a) > 0 else 0
    fp_red = 1 - fp_a / max(fp_b, 1)
    print(f"{k:>6} | {tp_b:>7} | {fp_b:>7} | {prec_b:>9.4f} | {tp_a:>6} | {fp_a:>6} | {prec_a:>8.4f} | {fp_red:>7.4f}")

print(f"\nFiltered out by threshold 0.5 (at k=2000): {filt}/2000")
print("Note: This is TF-IDF baseline (H0). SLM zero-shot should beat this.")
