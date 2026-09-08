"""Train TTP head (TF-IDF char 2-5 + LogReg) tren AD-GEN 6.5K chain co GT TTP.
Multi-label One-vs-Rest per TTP — output per-TTP AP/AUC + demo inference.
Chay CPU — khong can GPU. Do lech vs zero-shot SLM T1218-collapse.
"""
import json, re, collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import random, sys
sys.stdout.reconfigure(line_buffering=True)

SEED = 42
SRC = "P1/Output/data/adgen-chains.jsonl"
OUT = "P1/Output/results_phase2/adgen-ttp-head.json"

def text_of(o):
    ch = o.get("parent_chain", [])
    if isinstance(ch, str):
        try: ch = json.loads(ch.replace("'", '"'))
        except Exception: ch = []
    return " | ".join([(c.get("msg") or "") for c in (ch or [])][-10:])

def main():
    rows = []
    for l in open(SRC, encoding="utf-8"):
        try: o = json.loads(l)
        except Exception: continue
        if o.get("label") != 1: continue
        tids = set(re.match(r"(T\d{4})", t).group(1)
                   for t in re.findall(r"T\d{4}(?:\.\d{3})?", str(o.get("technique_id", ""))))
        if not tids: continue
        rows.append((text_of(o), tids))
    print(f"rows co GT TTP: {len(rows)}")

    # 12 TTP co GT >= 60 mau trong bench (dieu kien N cua thay)
    keep = ["T1059", "T1547", "T1055", "T1543", "T1003", "T1562",
            "T1490", "T1553", "T1036", "T1070", "T1218", "T1071"]
    sel = [(t, s) for (t, s) in rows if s & set(keep)]
    print(f"rows trong {len(keep)} TTP: {len(sel)}")

    texts = [t for (t, s) in sel]
    Y = np.zeros((len(sel), len(keep)), dtype=int)
    for i, (_, s) in enumerate(sel):
        for j, k in enumerate(keep):
            if k in s: Y[i, j] = 1

    Xtr, Xte, Ytr, Yte = train_test_split(texts, Y, test_size=0.2, random_state=SEED)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=100_000, min_df=2)
    Xtrv = vec.fit_transform(Xtr); Xtev = vec.transform(Xte)
    print("train/test:", Xtrv.shape, Xtev.shape)

    res = []
    for j, k in enumerate(keep):
        ytr, yte = Ytr[:, j], Yte[:, j]
        if ytr.sum() < 20 or yte.sum() < 5:
            res.append({"ttp": k, "skip": True, "n_train_pos": int(ytr.sum()), "n_test_pos": int(yte.sum())})
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
        clf.fit(Xtrv, ytr)
        p = clf.predict_proba(Xtev)[:, 1]
        ap = average_precision_score(yte, p)
        auc = roc_auc_score(yte, p) if yte.sum() > 0 and (1 - yte).sum() > 0 else None
        res.append({"ttp": k, "AP": round(ap, 4), "AUC": (round(auc, 4) if auc else None),
                    "n_train_pos": int(ytr.sum()), "n_test_pos": int(yte.sum())})
        print(f"{k}: AP={ap:.3f} AUC={auc:.3f} train_pos={int(ytr.sum())} test_pos={int(yte.sum())}")

    aps = [r["AP"] for r in res if "AP" in r]
    json.dump({"n_rows": len(sel), "ttps": keep, "per_ttp": res,
               "mean_AP": round(float(np.mean(aps)), 4)},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"SAVED {OUT} — mean AP {np.mean(aps):.3f}")

if __name__ == "__main__":
    main()
