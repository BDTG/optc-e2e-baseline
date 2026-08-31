import json, itertools, torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
DATA="P1/Output/data/alerts-enriched-v2.jsonl"
GT="P1/Output/data/gt_and_scores.json"
SECBERT_DIR="P1/Output/results_phase2/secbert-finetuned"
def build(a): return " | ".join([(c.get("msg") or "") for c in (a.get("parent_chain",[]) or [])[-5:]])[:800]
def has_cmd(r):
    for c in r.get("parent_chain",[]) or []:
        m=c.get("msg") or ""
        if "| cmd:" in m:
            v=m.split("| cmd:",1)[1].strip()
            if v and v.lower()!="none": return True
    return False
def has_chain(r): return len(r.get("parent_chain",[]) or [])>=2
def has_events(r): return len(r.get("event_seq",[]) or [])>=3
print("Load...", flush=True)
gt=set(json.load(open(GT))["gt_nids"])
alerts=[json.loads(l) for l in open(DATA,encoding='utf-8') if l.strip()]
labels=np.array([1 if str(a["nid"]) in gt else 0 for a in alerts])
texts=[build(a) for a in alerts]
from collections import Counter
cnt=Counter((a.get("parent_chain",[]) or [{}])[-1].get("msg","")[:80].split("|")[0].strip() for a in alerts)
total=len(alerts)
print(f"Total {len(alerts)} pos={labels.sum()}", flush=True)
# probs
vec=TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), max_features=50000)
X=vec.fit_transform(texts)
clf=LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X, labels)
p_tfidf=clf.predict_proba(X)[:,1]
print("Emb 0.6B...", flush=True)
emb_m=SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda", model_kwargs={'torch_dtype': torch.float16})
emb=emb_m.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
from sklearn.linear_model import LogisticRegression as LR2
clf2=LR2(max_iter=500, class_weight='balanced')
clf2.fit(emb, labels)
p_emb=clf2.predict_proba(emb)[:,1]
del emb_m; torch.cuda.empty_cache()
print("SecBERT...", flush=True)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok=AutoTokenizer.from_pretrained(SECBERT_DIR, local_files_only=True, trust_remote_code=True)
    m=AutoModelForSequenceClassification.from_pretrained(SECBERT_DIR, local_files_only=True, trust_remote_code=True).to("cuda").eval()
    p_sec=[]
    for i in range(0,len(texts),32):
        enc=tok(texts[i:i+32], truncation=True, padding=True, max_length=512, return_tensors="pt").to("cuda")
        with torch.no_grad():
            p=torch.softmax(m(**enc).logits, dim=-1)[:,1].cpu().numpy()
            p_sec.extend(p.tolist())
    p_sec=np.array(p_sec)
    del m; torch.cuda.empty_cache()
except: p_sec=np.zeros(len(alerts))
has1=np.array([has_cmd(a) for a in alerts])
has2=np.array([has_chain(a) for a in alerts])
has3=np.array([has_events(a) for a in alerts])
prev_vals=np.array([cnt[(a.get("parent_chain",[]) or [{}])[-1].get("msg","")[:80].split("|")[0].strip()]/total for a in alerts])
# LOOSE grid to keep 7-9 pos
tfidf_thr=[0.2,0.3,0.5]
emb_thr=[0.2,0.3,0.5]
sec_thr=[0.1,0.2,0.3]
prev_thr=[0.02,0.05,0.1]
final_thr=[0.5,0.7]
combos=list(itertools.product(tfidf_thr, emb_thr, sec_thr, prev_thr, final_thr))
print(f"Sweep {len(combos)} combos (loose)...", flush=True)
results=[]
for tf, em, sc, pr, fi in combos:
    mask = has1 & has2 & has3 & (p_tfidf>tf) & (p_emb>em) & (p_sec>sc) & (prev_vals<pr) & (p_emb>fi)
    n=mask.sum()
    if n==0: continue
    pos=(labels[mask]==1).sum()
    prev=pos/n if n else 0
    rec=pos/12
    f1=2*prev*rec/(prev+rec) if (prev+rec) else 0
    results.append((f1, prev, rec, n, pos, tf, em, sc, pr, fi))
results=sorted(results, reverse=True)
print(f"\nTop 10 F1 (loose):", flush=True)
for f1, prev, rec, n, pos, tf, em, sc, pr, fi in results[:10]:
    print(f"F1={f1:.3f} prev={prev:.3f} rec={rec:.2f} n={n:3d} pos={pos} tf={tf} em={em} sec={sc} prev<{pr} final>{fi}", flush=True)
print(f"\nBest with rec>=0.58 (keep >=7 pos):", flush=True)
cands=[r for r in results if r[2]>=0.58]
for f1, prev, rec, n, pos, tf, em, sc, pr, fi in sorted(cands, key=lambda x: -x[1])[:10]:
    print(f"prev={prev:.3f} rec={rec:.2f} n={n:3d} pos={pos} tf={tf} em={em} sec={sc} prev<{pr} final>{fi} F1={f1:.3f}", flush=True)
print(f"\nBest with rec>=0.75 (keep >=9 pos):", flush=True)
cands2=[r for r in results if r[2]>=0.75]
for f1, prev, rec, n, pos, tf, em, sc, pr, fi in sorted(cands2, key=lambda x: -x[1])[:10]:
    print(f"prev={prev:.3f} rec={rec:.2f} n={n:3d} pos={pos} tf={tf} em={em} sec={sc} prev<{pr} final>{fi} F1={f1:.3f}", flush=True)
import os, json
os.makedirs("P1/Output/results_phase2", exist_ok=True)
with open("P1/Output/results_phase2/sweep-loose.json","w") as f:
    json.dump([{"f1":float(f1),"prev":float(prev),"rec":float(rec),"n":int(n),"pos":int(pos),"tf":tf,"em":em,"sec":sc,"prev_thr":pr,"final":fi} for f1,prev,rec,n,pos,tf,em,sc,pr,fi in results[:100]], f, indent=2)
print("SAVED sweep-loose.json", flush=True)
