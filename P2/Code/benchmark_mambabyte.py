import json, time, torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

DATA="P1/Output/data/alerts-enriched-v2.jsonl"
GT="P1/Output/data/gt_and_scores.json"
# S1 rich 474 pos9
MODELS=[
  ("google/byt5-small", "byt5-byte"),          # byte-level T5, 300M, no BPE
  ("state-spaces/mamba-130m-hf", "mamba-linear"), # linear SSM O(n), 130M
  ("state-spaces/mamba-370m-hf", "mamba-linear"),
]
print("S1 load...", flush=True)
gt=set(json.load(open(GT))["gt_nids"])
alerts=[json.loads(l) for l in open(DATA,encoding='utf-8') if l.strip()]
def build_text(a):
    chain=a.get("parent_chain",[]) or []
    return " | ".join([(c.get("msg") or "") for c in chain[-5:]])[:800] or (a.get("msg") or "")[:800]
def has_cmd(r):
    for c in r.get("parent_chain",[]) or []:
        m=c.get("msg") or ""
        if "| cmd:" in m:
            v=m.split("| cmd:",1)[1].strip()
            if v and v.lower()!="none": return True
    return False
def has_chain(r): return len(r.get("parent_chain",[]) or [])>=2
def has_events(r): return len(r.get("event_seq",[]) or [])>=3
S1_idx=[i for i in range(len(alerts)) if has_cmd(alerts[i]) and has_chain(alerts[i]) and has_events(alerts[i])]
S1=[alerts[i] for i in S1_idx]
print(f"S1 n={len(S1)} pos={sum(1 for i in S1_idx if str(alerts[i]['nid']) in gt)}", flush=True)
texts=[build_text(a) for a in S1]
labels=[1 if str(S1[i]["nid"]) in gt else 0 for i in range(len(S1))]
# train/test 380/94
import random; random.seed(42)
idx=list(range(len(S1))); random.shuffle(idx)
train_idx=idx[:380]; test_idx=idx[380:]
for mname, typ in MODELS:
    print(f"\n=== {mname} ({typ}) ===", flush=True)
    t0=time.time()
    try:
        if typ=="byt5-byte":
            tok=AutoTokenizer.from_pretrained(mname)
            model=AutoModel.from_pretrained(mname, dtype=torch.float16, device_map="auto")
            model.eval()
            # encode via encoder mean pool
            def encode(batch):
                enc=tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out=model.encoder(**enc).last_hidden_state  # byt5 encoder
                    mask=enc.attention_mask.unsqueeze(-1)
                    emb=(out*mask).sum(1)/mask.sum(1).clamp(min=1)
                return emb.float().cpu().numpy()
        else:
            tok=AutoTokenizer.from_pretrained(mname, trust_remote_code=True)
            if tok.pad_token is None: tok.pad_token=tok.eos_token
            model=AutoModelForCausalLM.from_pretrained(mname, dtype=torch.float16, device_map="auto", trust_remote_code=True)
            model.eval()
            def encode(batch):
                enc=tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out=model(**enc, output_hidden_states=True)
                    # use last hidden mean pool
                    hs=out.hidden_states[-1]
                    mask=enc.attention_mask.unsqueeze(-1)
                    emb=(hs*mask).sum(1)/mask.sum(1).clamp(min=1)
                return emb.float().cpu().numpy()
        # batch encode
        import math
        B=16
        embs=[]
        for i in range(0,len(texts),B):
            embs.append(encode(texts[i:i+B]))
        import numpy as np
        X=np.concatenate(embs,axis=0)
        print(f" encode {time.time()-t0:.2f}s dim={X.shape[1]}", flush=True)
        # LR 5-fold quick
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import average_precision_score
        skf=StratifiedKFold(2, shuffle=True, random_state=42)
        aps=[]
        for tr, te in skf.split(X, labels):
            clf=LogisticRegression(max_iter=500, class_weight="balanced")
            clf.fit(X[tr], [labels[i] for i in tr])
            prob=clf.predict_proba(X[te])[:,1]
            aps.append(average_precision_score([labels[i] for i in te], prob))
        print(f" 2-fold AP={sum(aps)/len(aps):.4f} {aps}", flush=True)
        # latency per alert
        print(f" latency {(time.time()-t0)/len(texts)*1000:.1f} ms/alert", flush=True)
        del model; torch.cuda.empty_cache()
    except Exception as e:
        print(f" FAIL {e}", flush=True)
        import traceback; traceback.print_exc()
        try: torch.cuda.empty_cache()
        except: pass
print("DONE", flush=True)
