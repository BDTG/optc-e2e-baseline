"""TF-IDF head 12 TTP cung split LAB->REAL (cong bang voi SLM head) — khong random.
Baseline cho SLM LAB->REAL 0.3243.
"""
import json, re, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
sys.stdout.reconfigure(line_buffering=True)

SRC = "P1/Output/data/adgen-chains.jsonl"
OUT = "P1/Output/results_phase2/adgen-ttp-head-tfidf-LABREAL.json"
KEEP = ["T1059","T1547","T1055","T1543","T1003","T1562","T1490","T1553","T1036","T1070","T1218","T1071"]

def text_of(o):
    ch = o.get("parent_chain", [])
    if isinstance(ch, str):
        try: ch = json.loads(ch.replace("'", '"'))
        except Exception: ch = []
    return " | ".join([(c.get("msg") or "") for c in (ch or [])][-10:])

rows = []
for l in open(SRC, encoding="utf-8"):
    try: o = json.loads(l)
    except Exception: continue
    if o.get("label") != 1: continue
    tids = set(re.match(r"(T\d{4})", t).group(1) for t in re.findall(r"T\d{4}(?:\.\d{3})?", str(o.get("technique_id", ""))))
    if not tids: continue
    src = "LAB" if "LAB" in str(o.get("src", "")) else "REAL"
    rows.append({"text": text_of(o), "src": src,
                 "labels": [1 if k in tids else 0 for k in KEEP]})

tr = [r for r in rows if r["src"] == "LAB"]; te = [r for r in rows if r["src"] == "REAL"]
print(f"LAB {len(tr)} / REAL {len(te)}")
Ytr = np.array([r["labels"] for r in tr]); Yte = np.array([r["labels"] for r in te])
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=100_000, min_df=2)
Xtr = vec.fit_transform([r["text"] for r in tr]); Xte = vec.transform([r["text"] for r in te])
res = []
for j, k in enumerate(KEEP):
    ytr, yte = Ytr[:, j], Yte[:, j]
    if ytr.sum() < 20 or yte.sum() < 20:
        res.append({"ttp": k, "skip": True, "n_train_pos": int(ytr.sum()), "n_test_pos": int(yte.sum())}); continue
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    ap = average_precision_score(yte, p); auc = roc_auc_score(yte, p)
    res.append({"ttp": k, "AP": round(float(ap), 4), "AUC": round(float(auc), 4),
                "n_train_pos": int(ytr.sum()), "n_test_pos": int(yte.sum())})
    print(f"{k}: AP={ap:.3f} AUC={auc:.3f}")
aps = [r["AP"] for r in res if "AP" in r]
json.dump({"split": "LAB->REAL", "per_ttp": res, "mean_AP": round(float(np.mean(aps)), 4)},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("SAVED", OUT, "mean AP", round(float(np.mean(aps)), 4))
