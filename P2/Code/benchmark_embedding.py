import json, torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

DATA="P1/Output/data/alerts-enriched-v2.jsonl"
GT="P1/Output/data/gt_and_scores.json"
MODELS=[
    "Qwen/Qwen3-Embedding-0.6B",
    "ibm-granite/granite-embedding-small-english-r2",
    "sentence-transformers/all-MiniLM-L6-v2",
]

def build_text(a):
    chain=a.get("parent_chain",[]) or []
    return " | ".join([(c.get("msg") or "") for c in chain[-5:]])[:800]
def has_cmd(r):
    for c in r.get("parent_chain",[]) or []:
        msg=c.get("msg") or ""
        if "| cmd:" in msg:
            v=msg.split("| cmd:",1)[1].strip()
            if v and v.lower()!="none": return True
    return False
def has_chain(r): return len(r.get("parent_chain",[]) or [])>=2
def has_events(r): return len(r.get("event_seq",[]) or [])>=3

gt=set(json.load(open(GT))["gt_nids"])
alerts=[json.loads(l) for l in open(DATA,encoding='utf-8') if l.strip()]
s1=[a for a in alerts if all([has_cmd(a), has_chain(a), has_events(a)])]
print(f"S1 n={len(s1)} pos={sum(str(a['nid']) in gt for a in s1)}", flush=True)
texts=[build_text(a) for a in s1]
labels=np.array([0 if str(a["nid"]) in gt else 1 for a in s1])
y_true=1-labels  # 1=mal

import random
random.seed(42)
idx=list(range(len(s1)))
random.shuffle(idx)
train_idx=idx[:380]
test_idx=idx[380:]
print(f"train {len(train_idx)} test {len(test_idx)}", flush=True)

results=[]
for mid in MODELS:
    print(f"\n=== {mid} ===", flush=True)
    try:
        try:
            model=SentenceTransformer(mid, trust_remote_code=True, device="cuda")
        except:
            model=SentenceTransformer(mid, device="cuda")
    except Exception as e:
        print(f"  skip {mid}: {e}", flush=True)
        continue
    import time
    t0=time.time()
    emb_train=model.encode([texts[i] for i in train_idx], batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    emb_test=model.encode([texts[i] for i in test_idx], batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    enc_time=time.time()-t0
    print(f"  encode {enc_time:.2f}s dim={emb_train.shape[1]}", flush=True)
    clf=LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(emb_train, labels[train_idx])
    probs=clf.predict_proba(emb_test)[:,1]  # prob benign
    probs_mal=1-probs
    y_test_mal=y_true[test_idx]
    ap=average_precision_score(y_test_mal, probs_mal)
    try: auc=roc_auc_score(y_test_mal, probs_mal)
    except: auc=0
    print(f"  AP={ap:.4f} AUC={auc:.4f}", flush=True)
    results.append({"model":mid, "dim": int(emb_train.shape[1]), "ap": float(ap), "auc": float(auc), "encode_time": float(enc_time)})
    del model
    torch.cuda.empty_cache()

import os, json
os.makedirs("P1/Output/results_phase2", exist_ok=True)
with open("P1/Output/results_phase2/embedding-benchmark.json","w") as f:
    json.dump({"results":results}, f, indent=2)
print(f"\nSAVED embedding-benchmark.json", flush=True)
for r in results:
    print(f"{r['model']:45s} dim={r['dim']:4d} AP={r['ap']:.4f} AUC={r['auc']:.4f}", flush=True)
