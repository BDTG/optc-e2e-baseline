"""TF-IDF S1-only re-run — xac minh lai con so 0.89 con thieu file goc.
Tap: S1 474 alerts (has_cmd + chain>=2 + event_seq>=3, pos 9) tu alerts-enriched-v2.
Model: TfidfVectorizer char2-5 50k + LogisticRegression balanced, 5-fold CV OOF AP.
CPU, khong can GPU."""
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score

DATA = "P1/Output/data/alerts-enriched-v2.jsonl"
GT = "P1/Output/data/gt_and_scores.json"
OUT = "P1/Output/results_phase2/tfidf-s1-474-result.json"
SEED = 42

gt = set(map(str, json.load(open(GT, encoding="utf-8"))["gt_nids"]))
alerts = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]


def has_cmd(r):
    for c in r.get("parent_chain", []) or []:
        m = c.get("msg") or ""
        if "| cmd:" in m:
            v = m.split("| cmd:", 1)[1].strip()
            if v and v.lower() != "none":
                return True
    return False


def build_text(a):
    chain = a.get("parent_chain", []) or []
    return " | ".join([(c.get("msg") or "") for c in chain[-5:]])[:800]


S1 = [a for a in alerts if has_cmd(a)
      and len(a.get("parent_chain", []) or []) >= 2
      and len(a.get("event_seq", []) or []) >= 3]
X = [build_text(a) for a in S1]
y = np.array([1 if str(a["nid"]) in gt else 0 for a in S1])
print(f"S1 n={len(S1)} pos={int(y.sum())}", flush=True)

skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof = np.zeros(len(y))
fold_aps = []
for tr, te in skf.split(X, y):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=50000)
    Xtr = vec.fit_transform([X[i] for i in tr])
    Xte = vec.transform([X[i] for i in te])
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(Xtr, y[tr])
    prob = clf.predict_proba(Xte)[:, 1]
    oof[te] = prob
    fold_aps.append(round(float(average_precision_score(y[te], prob)), 4))

res = {"n": len(S1), "pos": int(y.sum()), "k": 5, "seed": SEED,
       "model": "TF-IDF char2-5 50k + LogReg balanced (raw msg concat, no norm)",
       "ap_oof": round(float(average_precision_score(y, oof)), 4),
       "auc_oof": round(float(roc_auc_score(y, oof)), 4),
       "fold_aps": fold_aps}
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"SAVED {OUT}", flush=True)
