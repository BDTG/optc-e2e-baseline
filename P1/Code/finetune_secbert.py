import json, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np

DATA="P1/Output/data/alerts-enriched-v2.jsonl"
GT="P1/Output/data/gt_and_scores.json"
MODEL="jackaduma/SecBERT"
OUT="P1/Output/results_phase2/secbert-finetuned"

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
recs=[{"text":build_text(a),"label":0 if str(a["nid"]) in gt else 1} for a in s1]

import random
random.seed(42); random.shuffle(recs)
train=recs[:380]; test=recs[380:]
print(f"train {len(train)} test {len(test)}", flush=True)

tok=AutoTokenizer.from_pretrained(MODEL)
def tok_fn(e): return tok(e["text"], truncation=True, max_length=512)
train_ds=Dataset.from_list(train).map(lambda e: {**tok_fn(e), "labels": e["label"]}, remove_columns=["text","label"])
test_ds=Dataset.from_list(test).map(lambda e: {**tok_fn(e), "labels": e["label"]}, remove_columns=["text","label"])

model=AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2, dtype=torch.bfloat16, ignore_mismatched_sizes=True).to("cuda")
model.gradient_checkpointing_enable()

args=TrainingArguments(output_dir=OUT, per_device_train_batch_size=8, gradient_accumulation_steps=2, num_train_epochs=3, learning_rate=2e-5, logging_steps=10, save_strategy="no", report_to="none", fp16=False, bf16=True, seed=42)
def compute(p):
    logits, labels = p
    probs=torch.softmax(torch.tensor(logits).float(), -1)[:,1].numpy()
    probs=np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    return {"ap": average_precision_score(1-np.array(labels), probs), "auc": roc_auc_score(1-np.array(labels), probs)}

trainer=Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds, data_collator=DataCollatorWithPadding(tok), compute_metrics=compute)
trainer.train()
res=trainer.evaluate()
print(f"RESULT ap={res['eval_ap']:.4f} auc={res['eval_auc']:.4f}", flush=True)
import json, os
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(f"{OUT}/result.json","w"), indent=2)
model.save_pretrained(OUT); tok.save_pretrained(OUT)
print(f"SAVED {OUT}/result.json", flush=True)
