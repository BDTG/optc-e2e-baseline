"""
Tier-2 H0 check (RQ1a): fine-tune ModernBERT ~150M as binary classifier.
If BERT_AP >= SLM_LoRA_AP, H0 (encoder beats SLM) holds -> Note.md:89 rewrite.
CPU-only. Same 1800/450 split + labels as LoRA script.
"""
import json, random, os
import numpy as np, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import average_precision_score, roc_auc_score

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
MODEL="answerdotai/modernbert-base"
ALERTS="P1/Output/alerts-enriched-v2.jsonl"; GT="P1/Output/gt_and_scores.json"
OUT="P1/Output/models/bert-150m"; os.makedirs(OUT, exist_ok=True)

gt=set(json.load(open(GT))["gt_nids"])
alerts=[json.loads(l) for l in open(ALERTS) if l.strip()]
def build_text(a):
    chain=a.get("parent_chain",[]) or []; seq=a.get("event_seq",[]) or []
    parts=[c.get("msg","") for c in chain[-5:]]
    parts+=[f"{e.get('src_msg','')} -> {e.get('dst_msg','')}" for e in seq[-5:]]
    return " | ".join([p for p in parts if p])
recs=[{"text":build_text(a),"label":0 if str(a["nid"]) in gt else 1} for a in alerts]
random.shuffle(recs); train=recs[:1800]; hold=recs[1800:]
json.dump(hold, open(os.path.join(OUT,"holdout.json"),"w"))
print(f"train={len(train)} hold={len(hold)} pos={sum(r['label']==0 for r in train)}")

tok=AutoTokenizer.from_pretrained(MODEL)
def fn(e):
    enc=tok(e["text"], truncation=True, max_length=512)
    enc["labels"]=e["label"]
    return enc
trds=Dataset.from_list(train).map(fn, remove_columns=["text","label"])
hods=Dataset.from_list(hold).map(fn, remove_columns=["text","label"])

model=AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2,
        dtype=torch.bfloat16)
model=model.to("cuda")
model.gradient_checkpointing_enable()
args=TrainingArguments(output_dir=OUT, per_device_train_batch_size=4,
    gradient_accumulation_steps=4, num_train_epochs=3, learning_rate=2e-5,
    logging_steps=20, save_strategy="no", report_to="none", optim="adamw_torch",
    fp16=False, bf16=True, seed=SEED, dataloader_num_workers=0)
def compute(eval_pred):
    logits,labels=eval_pred
    probs=torch.softmax(torch.tensor(logits),-1)[:,1].numpy()
    return {"val_ap":average_precision_score(labels,probs),
            "val_auc":roc_auc_score(labels,probs)}
trainer=Trainer(model=model,args=args,train_dataset=trds,eval_dataset=hods,
    data_collator=DataCollatorWithPadding(tok),compute_metrics=compute)
trainer.train()
# final eval
res=trainer.evaluate()
print("=== BERT RESULT ===", json.dumps(res, indent=2))
json.dump(res, open(os.path.join(OUT,"result.json"),"w"), indent=2)
model.save_pretrained(OUT); tok.save_pretrained(OUT)
print("=== BERT DONE ===")
