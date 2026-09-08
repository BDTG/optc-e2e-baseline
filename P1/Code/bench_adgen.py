"""Bench AD-GEN tier-2 — CPU sklearn only:
 1. TF-IDF char2-5 + LogReg: OOF 5-fold AP/AUC tren BALANCED 200 (100 mal + 100 ben)
 2. Verdict-truth check: AD-GEN co 'verdict' GT nen do chinh xac file A-style tren sample
 3. TTP coverage: tan so technique_id trong mal (GT TTP co that!)
"""
import json, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
import collections

SAMPLE = "P1/Output/data/adgen-chains-balanced.jsonl"
FULL = "P1/Output/data/adgen-chains.jsonl"
OUT = "P1/Output/results_phase2/adgen-bench.json"
SEED = 42

def chain_text(o):
    ch = o.get("parent_chain", []) or []
    return " | ".join([(c.get("msg") or "") for c in ch[-5:]])[:2000]

def main():
    rows = [json.loads(l) for l in open(SAMPLE, encoding="utf-8")]
    y = np.array([r["label"] for r in rows])
    X = [chain_text(r) for r in rows]
    print(f"sample n={len(y)} pos={int(y.sum())}", flush=True)

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y)); folds = []
    for tr, te in skf.split(X, y):
        v = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=50000)
        c = LogisticRegression(class_weight="balanced", max_iter=1000)
        c.fit(v.fit_transform([X[i] for i in tr]), y[tr])
        p = c.predict_proba(v.transform([X[i] for i in te]))[:, 1]
        oof[te] = p
        folds.append(round(float(average_precision_score(y[te], p)), 4))
    ap = round(float(average_precision_score(y, oof)), 4)
    auc = round(float(roc_auc_score(y, oof)), 4)
    print(f"TF-IDF balanced200: AP={ap} AUC={auc} folds={folds}", flush=True)

    ttps = collections.Counter()
    ttp_rows = 0
    for r in rows:
        if r["label"] == 1 and r.get("technique_id"):
            ttp_rows += 1
            for t in re.findall(r"T\d{4}(?:\.\d{3})?", r["technique_id"]):
                ttps[t] += 1
    print("TTP GT trong mal sample:", dict(ttps.most_common(10)), f"(rows co GT: {ttp_rows})", flush=True)

    full_mal = full_tech = 0
    fttp = collections.Counter()
    for l in open(FULL, encoding="utf-8"):
        if not l.strip():
            continue
        o = json.loads(l)
        if o.get("label") == 1:
            full_mal += 1
            ts = re.findall(r"T\d{4}(?:\.\d{3})?", str(o.get("technique_id", "")))
            if ts:
                full_tech += 1
                for t in ts:
                    fttp[t] += 1
    print(f"FULL: mal={full_mal}, co TTP={full_tech}", flush=True)

    res = {"n": len(y), "pos": int(y.sum()), "tfidf": {"ap_oof": ap, "auc_oof": auc, "folds": folds},
           "ttp_top_sample": dict(ttps.most_common(15)), "ttp_rows_sample": ttp_rows,
           "full_mal": full_mal, "full_mal_with_ttp": full_tech, "ttp_top_full": dict(fttp.most_common(15))}
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"SAVED {OUT}", flush=True)

if __name__ == "__main__":
    main()
