"""Test LoRA Qwen 0.5B (train OpTC V2) tren TTP holdout.
Same 319 alerts, cung BERT. So sanh truc tiep."""
import json, torch
import numpy as np
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import average_precision_score, roc_auc_score

BASE="Qwen/Qwen2.5-0.5B-Instruct"
CKPT="P1/Output/models/lora-05b/checkpoint-113"
TTP="P1/Output/ttp_holdout.jsonl"

tok=AutoTokenizer.from_pretrained(BASE)
base=AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16,
    device_map="auto")
model=PeftModel.from_pretrained(base, CKPT)
model.eval()

alerts=[json.loads(l) for l in open(TTP) if l.strip()]
labels=[a["label"] for a in alerts]
y_true=np.array([1-l for l in labels])

def build_text(a):
    chain=a.get("parent_chain",[]) or []
    seq=a.get("event_seq",[]) or []
    parts=[c.get("msg","") for c in chain[-5:]]
    parts+=[f"{e.get('src_msg','')} -> {e.get('dst_msg','')}" for e in seq[-5:]]
    return " | ".join([p for p in parts if p])

prompts=[build_text(a) + "\nVerdict:" for a in alerts]
print(f"TTP n={len(alerts)} mal={int(y_true.sum())}")

mal_ids=tok(" MAL", add_special_tokens=False).input_ids[0]
ben_ids=tok(" BEN", add_special_tokens=False).input_ids[0]

scores=[]
with torch.no_grad():
    for i,p in enumerate(prompts):
        ids=tok(p, return_tensors="pt", truncation=True, max_length=500).to(model.device)
        out=model(**ids)
        probs=torch.softmax(out.logits[0,-1].float(), -1)
        mal_p=float(probs[mal_ids].item())
        ben_p=float(probs[ben_ids].item())
        scores.append(mal_p / (mal_p + ben_p + 1e-9))
        if (i+1) % 50 == 0: print(f"  {i+1}/{len(prompts)}")

scores=np.array(scores)
ap=average_precision_score(y_true, scores)
try: auc=roc_auc_score(y_true, scores)
except: auc=-1.0

print(f"\n=== LoRA 0.5B on TTP holdout ===")
print(f"AP  : {ap:.4f}")
print(f"AUC : {auc:.4f}")

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

out={"ap":float(ap),"auc":float(auc),"n":len(alerts),
     "mal":int(y_true.sum()),
     "by_ttp":{k:{"n":len(v["y"]),"mal":int(sum(v["y"])),
                   "top1":int(np.array(v["y"])[np.argmax(v["s"])])}
               for k,v in by_ttp.items()}}
json.dump(out, open("P1/Output/lora-ttp-result.json","w"), indent=2)
print("\n=== SAVED lora-ttp-result.json ===")