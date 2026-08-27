"""Test BERT 150M (train OpTC V2) tren TTP holdout that su.
Expect: AP thap vi TTP unseen. Test generalisation gap."""
import json, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import average_precision_score, roc_auc_score

MODEL="answerdotai/modernbert-base"
CKPT="P1/Output/models/bert-150m"
TTP="P1/Output/ttp_holdout.jsonl"

tok=AutoTokenizer.from_pretrained(CKPT)
model=AutoModelForSequenceClassification.from_pretrained(CKPT, dtype=torch.bfloat16)
model=model.to("cuda")
model.eval()

alerts=[json.loads(l) for l in open(TTP) if l.strip()]
labels=[a["label"] for a in alerts]
y_true=np.array([1-l for l in labels])  # pos=mal

def build_text(a):
    chain=a.get("parent_chain",[]) or []
    seq=a.get("event_seq",[]) or []
    parts=[c.get("msg","") for c in chain[-5:]]
    parts+=[f"{e.get('src_msg','')} -> {e.get('dst_msg','')}" for e in seq[-5:]]
    return " | ".join([p for p in parts if p])

prompts=[build_text(a) for a in alerts]
print(f"TTP n={len(alerts)} mal={int(y_true.sum())} ben={int((1-y_true).sum())}")

scores=[]
with torch.no_grad():
    for p in prompts:
        ids=tok(p, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        out=model(**ids)
        probs=torch.softmax(out.logits[0].float(), -1).cpu().numpy()
        scores.append(probs[0])  # class 0 = malicious

scores=np.array(scores)
ap=average_precision_score(y_true, scores)
try:
    auc=roc_auc_score(y_true, scores)
except Exception as e:
    auc=-1.0
    print("AUC err:", e)

print(f"\n=== BERT 150M on TTP holdout ===")
print(f"AP  : {ap:.4f}")
print(f"AUC : {auc:.4f}")

# per-TTP breakdown
by_ttp={}
for i,a in enumerate(alerts):
    tid=a.get("ttp_id","?")
    by_ttp.setdefault(tid, {"y":[], "s":[]})
    by_ttp[tid]["y"].append(1-a["label"])
    by_ttp[tid]["s"].append(scores[i])

print(f"\n=== Per-TTP breakdown ===")
for tid,d in sorted(by_ttp.items()):
    yt=np.array(d["y"]); st=np.array(d["s"])
    n_mal=int(yt.sum())
    if n_mal==0: continue
    order=np.argsort(st)[::-1]
    ys=yt[order]
    print(f"{tid:12s}: n={len(yt)} mal={n_mal} top-1_hit={int(ys[0])} "
          f"top-2_hit={int(ys[:2].sum())} mean_score={st.mean():.3f}")

# save
out={"ap":float(ap),"auc":float(auc),"n":len(alerts),
     "mal":int(y_true.sum()),
     "by_ttp":{k:{"n":len(v["y"]),"mal":int(sum(v["y"])),
                   "top1":int(np.array(v["y"])[np.argmax(v["s"])])}
               for k,v in by_ttp.items()}}
json.dump(out, open("P1/Output/bert-ttp-result.json","w"), indent=2)
print("\n=== SAVED bert-ttp-result.json ===")