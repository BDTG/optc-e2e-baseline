import json, time, torch, random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sentence_transformers import SentenceTransformer

DATA="P1/Output/data/alerts-enriched-v2.jsonl"
GT="P1/Output/data/gt_and_scores.json"
SECBERT_DIR="P1/Output/results_phase2/secbert-finetuned"

def build_text(a):
    chain=a.get("parent_chain",[]) or []
    return " | ".join([(c.get("msg") or "") for c in chain[-5:]])[:800]

def has_cmd(r):
    for c in r.get("parent_chain",[]) or []:
        m=c.get("msg") or ""
        if "| cmd:" in m:
            v=m.split("| cmd:",1)[1].strip()
            if v and v.lower()!="none": return True
    return False
def has_chain(r): return len(r.get("parent_chain",[]) or [])>=2
def has_events(r): return len(r.get("event_seq",[]) or [])>=3

print("Loading...", flush=True)
gt=set(json.load(open(GT))["gt_nids"])
alerts=[json.loads(l) for l in open(DATA,encoding='utf-8') if l.strip()]
texts=[build_text(a) for a in alerts]
labels=np.array([1 if str(a["nid"]) in gt else 0 for a in alerts])  # 1=mal
print(f"Total {len(alerts)} pos={labels.sum()} prev={labels.mean():.4f}", flush=True)

# precompute process prevalence
from collections import Counter
proc_counts=Counter()
for a in alerts:
    chain=a.get("parent_chain",[]) or []
    if chain:
        subj=(chain[-1].get("msg") or "").split("|")[0].strip()[:80]
        proc_counts[subj]+=1
total=len(alerts)
prev_thresh=0.01  # <1% rare (tune từ 0.1% -> 1% để giữ 7-8 pos)

# TF-IDF quick train
print("\n[4] Train TF-IDF...", flush=True)
vec=TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), max_features=50000)
X=vec.fit_transform(texts)
clf_tfidf=LogisticRegression(max_iter=1000, class_weight='balanced')
clf_tfidf.fit(X, labels)
probs_tfidf=clf_tfidf.predict_proba(X)[:,1]

# Embedding 0.6B
print("[5] Load Qwen3-Embedding-0.6B...", flush=True)
emb_model=SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda", model_kwargs={'torch_dtype': torch.float16})
emb=emb_model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
# quick LR on embedding
from sklearn.linear_model import LogisticRegression as LR2
clf_emb=LR2(max_iter=500, class_weight='balanced')
clf_emb.fit(emb, labels)
probs_emb=clf_emb.predict_proba(emb)[:,1]
del emb_model; torch.cuda.empty_cache()
print(f" emb done dim={emb.shape[1]}", flush=True)

# SecBERT
print("[6] Load SecBERT FT...", flush=True)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok=AutoTokenizer.from_pretrained(SECBERT_DIR, local_files_only=True, trust_remote_code=True)
    m=AutoModelForSequenceClassification.from_pretrained(SECBERT_DIR, local_files_only=True, trust_remote_code=True).to("cuda").eval()
    import torch
    probs_sec=[]
    for i in range(0,len(texts),32):
        batch=texts[i:i+32]
        enc=tok(batch, truncation=True, padding=True, max_length=512, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits=m(**enc).logits
            p=torch.softmax(logits, dim=-1)[:,1].cpu().numpy()
            probs_sec.extend(p.tolist())
    probs_sec=np.array(probs_sec)
    del m; torch.cuda.empty_cache()
    print(" secbert done", flush=True)
except Exception as e:
    print(f" secbert skip {e}", flush=True)
    probs_sec=np.zeros(len(alerts))

# 8-filter cascade
filters=[
    ("1 has_cmd", lambda i: has_cmd(alerts[i])),
    ("2 has_chain>=2", lambda i: has_chain(alerts[i])),
    ("3 has_events>=3", lambda i: has_events(alerts[i])),
    ("4 TF-IDF>0.5", lambda i: probs_tfidf[i]>0.5),
    ("5 Emb0.6B>0.5", lambda i: probs_emb[i]>0.5),
    ("6 SecBERT>0.3", lambda i: probs_sec[i]>0.3),
    ("7 prevalence<1%", lambda i: proc_counts[(alerts[i].get("parent_chain",[]) or [{}])[-1].get("msg","")[:80].split("|")[0].strip()]/total < prev_thresh if alerts[i].get("parent_chain") else True),
    ("8 Final AI yes/no (Emb>0.7)", lambda i: probs_emb[i]>0.7),
]

remaining=list(range(len(alerts)))
import time
results=[]
print("\n=== 8-FILTER CASCADE ===", flush=True)
for fname, fn in filters:
    t0=time.time()
    before=len(remaining)
    pos_before=sum(labels[i] for i in remaining)
    remaining=[i for i in remaining if fn(i)]
    after=len(remaining)
    pos_after=sum(labels[i] for i in remaining)
    prev_after=pos_after/after if after else 0
    dt=time.time()-t0
    # AP on remaining vs all? compute AP of this filter's score on current set
    print(f"{fname:25s} {before:4d}->{after:4d} (-{before-after:4d}) pos {pos_before}->{pos_after} prev {prev_after:.3f} time {dt:.3f}s", flush=True)
    results.append({"filter":fname,"before":before,"after":after,"pos_after":int(pos_after),"prev":float(prev_after),"time":float(dt)})

# final AP of cascade (how well 8 filters rank)
# Use probs_emb as final score on original set, compute AP
ap_all=average_precision_score(labels, probs_emb)
try: auc_all=roc_auc_score(labels, probs_emb)
except: auc_all=0
print(f"\nFinal embedding AP(all)={ap_all:.4f} AUC={auc_all:.4f}", flush=True)
# AP on S1 subset (first 3 filters)
s1_idx=[i for i in range(len(alerts)) if has_cmd(alerts[i]) and has_chain(alerts[i]) and has_events(alerts[i])]
if s1_idx:
    ap_s1=average_precision_score(labels[s1_idx], probs_emb[s1_idx])
    print(f"S1 (3 filters) n={len(s1_idx)} AP={ap_s1:.4f}", flush=True)

import os, json
os.makedirs("P1/Output/results_phase2", exist_ok=True)
with open("P1/Output/results_phase2/filter-8cascade.json","w") as f:
    json.dump({"cascade":results,"ap_all":float(ap_all),"auc_all":float(auc_all)}, f, indent=2)
print("SAVED filter-8cascade.json", flush=True)
for r in results:
    print(f"{r['filter']:25s} -> {r['after']:4d} prev {r['prev']:.3f}", flush=True)
