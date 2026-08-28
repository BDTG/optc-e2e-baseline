"""
Chay lai S3 (toan bo 2250) 5-fold CV, BERT 150M, khong save checkpoint.
Python -u de log flush dong, khong bi stuck khi treo.
"""
import json, random, numpy as np, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import sys

# FORCE unbuffered output (in case run as `python` not `python -u`)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DATA="P1/Output/data/alerts-enriched-v2.jsonl"
GT="P1/Output/data/gt_and_scores.json"
MODEL="answerdotai/modernbert-base"

gt=set(json.load(open(GT))["gt_nids"])
alerts=[json.loads(l) for l in open(DATA, encoding='utf-8') if l.strip()]
def build_text(a):
    chain=a.get("parent_chain",[]) or []; seq=a.get("event_seq",[]) or []
    parts=[c.get("msg","") for c in chain[-5:]]
    parts+=[f"{e.get('src_msg','')} -> {e.get('dst_msg','')}" for e in seq[-5:]]
    return " | ".join([p for p in parts if p])

recs=[{"text":build_text(a),"label":0 if str(a["nid"]) in gt else 1} for a in alerts]
labels=[r["label"] for r in recs]
print(f"All: {len(recs)}, pos={labels.count(0)}", flush=True)

tok=AutoTokenizer.from_pretrained(MODEL)
skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results=[]

for fold,(ti,tei) in enumerate(skf.split(recs,labels)):
    train=Dataset.from_list([recs[i] for i in ti]).map(
        lambda e: {**tok(e["text"], truncation=True, max_length=512),
                   "labels": e["label"]}, remove_columns=["text","label"])
    test=Dataset.from_list([recs[i] for i in tei]).map(
        lambda e: {**tok(e["text"], truncation=True, max_length=512),
                   "labels": e["label"]}, remove_columns=["text","label"])
    pos_te=sum(1 for i in tei if recs[i]["label"]==0)
    print(f"\n=== fold {fold+1}/5: train={len(ti)} test={len(tei)} pos_test={pos_te} ===", flush=True)

    model=AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2,
            dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable()

    args=TrainingArguments(output_dir=f"P1/Output/models/bert-150m/S3v2-fold-{fold}",
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        num_train_epochs=3, learning_rate=2e-5,
        logging_steps=50, save_strategy="no",
        report_to="none", optim="adamw_torch",
        fp16=False, bf16=True, seed=SEED, dataloader_num_workers=0,
        log_level="info", disable_tqdm=False)
    def compute(eval_pred):
        logits,lab=eval_pred
        probs=torch.softmax(torch.tensor(logits).float(),-1)[:,1].numpy()
        ap=average_precision_score(1-np.array(lab),probs)
        try: auc=roc_auc_score(1-np.array(lab),probs)
        except: auc=-1.0
        return {"val_ap":ap,"val_auc":auc}
    trainer=Trainer(model=model, args=args, train_dataset=train, eval_dataset=test,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=compute)
    trainer.train()
    res=trainer.evaluate()
    res["fold"]=fold+1
    print(f"fold {fold+1} DONE: AP={res['eval_val_ap']:.4f} AUC={res['eval_val_auc']:.4f}", flush=True)
    results.append(res)
    del model; torch.cuda.empty_cache()

aps=[r["eval_val_ap"] for r in results]
aucs=[r["eval_val_auc"] for r in results]
print(f"\n=== S3 SUMMARY ===", flush=True)
print(f"AP mean={np.mean(aps):.4f} std={np.std(aps):.4f}", flush=True)
print(f"AUC mean={np.mean(aucs):.4f} std={np.std(aucs):.4f}", flush=True)

out={"S3_v2":{"ap_mean":float(np.mean(aps)),"ap_std":float(np.std(aps)),
              "auc_mean":float(np.mean(aucs)),"auc_std":float(np.std(aucs)),
              "per_fold":results,
              "note":"rerun S3 with python -u + save_strategy=no after previous hang"}}
with open("P1/Output/results_phase2/bert-150m-s3-v2.json","w") as f:
    json.dump(out, f, indent=2)
print("\nSAVED bert-150m-s3-v2.json", flush=True)