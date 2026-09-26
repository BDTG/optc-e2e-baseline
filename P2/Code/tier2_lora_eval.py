"""Eval LoRA 0.5B on 450 holdout. Labels are in holdout.json directly."""
import json, torch
import numpy as np
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import average_precision_score, roc_auc_score

BASE="Qwen/Qwen2.5-0.5B-Instruct"
CKPT="P1/Output/models/lora-05b/checkpoint-113"
HOLD="P1/Output/models/lora-05b/holdout.json"

tok=AutoTokenizer.from_pretrained(BASE)
base=AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16,
    device_map="auto")
model=PeftModel.from_pretrained(base, CKPT)
model.eval()

hold=json.load(open(HOLD))
labels=[r["label"] for r in hold]   # 0=mal, 1=benign
y_true = np.array([1-l for l in labels])  # pos=mal

prompts=[r["text"].replace("\nVerdict: MALICIOUS","").replace("\nVerdict: BENIGN","")
         + "\nVerdict:" for r in hold]
print(f"holdout n={len(labels)} mal={int((1-np.array(labels)).sum())} ben={int(np.sum(labels))}")

scores=[]
with torch.no_grad():
    for i,p in enumerate(prompts):
        ids=tok(p, return_tensors="pt", truncation=True, max_length=500).to(model.device)
        out=model(**ids)
        last=out.logits[0,-1]
        probs=torch.softmax(last, -1)
        # token IDs for " MAL" and " BEN"
        mal_ids=tok(" MAL", add_special_tokens=False).input_ids
        ben_ids=tok(" BEN", add_special_tokens=False).input_ids
        mal_p = float(probs[mal_ids[0]].item())
        ben_p = float(probs[ben_ids[0]].item())
        scores.append(mal_p / (mal_p + ben_p + 1e-9))
        if (i+1) % 50 == 0: print(f"  {i+1}/{len(prompts)}")

scores=np.array(scores)
ap = average_precision_score(y_true, scores)
auc = roc_auc_score(y_true, scores)
print(f"\n=== LoRA 0.5B EPOCH 1 RESULT ===")
print(f"AP  (malicious): {ap:.4f}")
print(f"AUC            : {auc:.4f}")

order=np.argsort(scores)[::-1]
y_sorted=y_true[order]
for k in [500, 1000, 2000]:
    tp=int(y_sorted[:k].sum())
    print(f"top-{k}: TP={tp}/{k} prec={tp/k*100:.2f}%")

out={"ap":float(ap),"auc":float(auc),"n":len(labels),
     "pos":int(y_true.sum()),
     "p_at":{"P@500":int(y_sorted[:500].sum()),
             "P@1000":int(y_sorted[:1000].sum()),
             "P@2000":int(y_sorted[:2000].sum())}}
json.dump(out, open("P1/Output/lora-05b-epoch1-result.json","w"), indent=2)
print("=== SAVED ===")