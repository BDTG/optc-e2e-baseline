"""A1 — Ensemble SLM head + TF-IDF head (split LAB->REAL, khong-leak).
Load du doan SLM (adgen-ttp-head-slm-LABREAL.json chi co AP per TTP — khong co she inference scores)
=> buoc chay lai: TF-IDF head predict tren REAL, SLM head can prob raw nen phai re-run inference.
De nhanh: dung TF-IDF alone grid-weighted voi "prior" SLM head — nhung khong co prob SLM luu
=> Chien luoc thuc te: A1 van re-run SLM inference 1 lan tren test REAL (978 mau, 0.199s×978 ~ 4ph)
De don gian va CHINH XAC: chay lai inference cho 978 mau TEST REAL, luu probs, roi ensemble voi TF-IDF.
"""
import json, re, sys, numpy as np
sys.stdout.reconfigure(line_buffering=True)

BASE = r"C:\Users\BDTG\Desktop\Backup\OpTC-thesis"
DATA = BASE + r"\P1\Output\data\adgen-chains.jsonl"
KEEP = ["T1059","T1547","T1055","T1543","T1003","T1562","T1490","T1553","T1036","T1070","T1218","T1071"]

# === 1. TF-IDF head probabilities cho test REAL ===
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def text_of(msgs):
    return " | ".join([(c.get("msg") or "") for c in (msgs or [])][-10:])

rows = []
for l in open(DATA, encoding="utf-8"):
    try: o = json.loads(l)
    except Exception: continue
    if o.get("label") != 1: continue
    tids = set(re.match(r"(T\d{4})", t).group(1) for t in re.findall(r"T\d{4}(?:\.\d{3})?", str(o.get("technique_id",""))))
    if not tids: continue
    ch = o.get("parent_chain", [])
    if isinstance(ch, str):
        try: ch = json.loads(ch.replace("'", '"'))
        except Exception: ch = []
    src = "LAB" if "LAB" in str(o.get("src","")) else "REAL"
    rows.append({"text": text_of(ch)[:8000], "src": src,
                 "labels":[1 if k in tids else 0 for k in KEEP]})

tr = [r for r in rows if r["src"]=="LAB"]; te = [r for r in rows if r["src"]=="REAL"]
print(f"LAB {len(tr)} / REAL {len(te)}")
Ytr = np.array([r["labels"] for r in tr]); Yte = np.array([r["labels"] for r in te])
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), max_features=100_000, min_df=2)
Xtr = vec.fit_transform([r["text"] for r in tr]); Xte = vec.transform([r["text"] for r in te])
P_tfidf = np.zeros((len(te), len(KEEP)))
for j, k in enumerate(KEEP):
    if Ytr[:,j].sum() < 20: continue
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    clf.fit(Xtr, Ytr[:,j])
    P_tfidf[:,j] = clf.predict_proba(Xte)[:,1]
np.save(BASE + r"\P1\Output\results_phase2\p3b-tfidf-probs.npy", P_tfidf)
json.dump({"test_texts_idx": [te[i]["labels"] for i in range(len(te))] if False else None,
           "Y_test": Yte.tolist()},
          open(BASE + r"\P1\Output\results_phase2\p3b-Ytest.json","w"))
from sklearn.metrics import average_precision_score
for j,k in enumerate(KEEP):
    if Yte[:,j].sum() >= 20:
        ap = average_precision_score(Yte[:,j], P_tfidf[:,j])
        print(f"TFIDF {k}: AP {ap:.3f}")
aps = [average_precision_score(Yte[:,j], P_tfidf[:,j]) for j in range(len(KEEP)) if Yte[:,j].sum()>=20]
print("TF-IDF mean AP (verify):", round(float(np.mean(aps)),4))
print("TFIDF probs saved — buoc 2 chay SLM inference (script rieng)")
