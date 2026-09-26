"""
Verify BERT 150M AP artifact on V2: 5-fold CV thay vi single 450 holdout.
Neu AP giam tu 0.9999 -> ~0.5 thi holdout artifact (do chi 2 pos).
Neu AP van ~0.9+ thi BERT that su strong.
"""
import json, random, numpy as np, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
ALERTS="P1/Output/data/alerts-enriched-v2.jsonl"
GT="P1/Output/data/gt_and_scores.json"
OUT="P1/Output/results_phase2"
MODEL="answerdotai/modernbert-base"

gt=set(json.load(open(GT))["gt_nids"])
alerts=[json.loads(l) for l in open(ALERTS) if l.strip()]
def build_text(a):
    chain=a.get("parent_chain",[]) or []; seq=a.get("event_seq",[]) or []
    parts=[c.get("msg","") for c in chain[-5:]]
    parts+=[f"{e.get('src_msg','')} -> {e.get('dst_msg','')}" for e in seq[-5:]]
    return " | ".join([p for p in parts if p])
recs=[{"text":build_text(a),"label":0 if str(a["nid"]) in gt else 1} for a in alerts]
random.seed(SEED); random.shuffle(recs)
labels=[r["label"] for r in recs]
print(f"total={len(recs)} pos={labels.count(0)} neg={labels.count(1)}")

# Stratified K-fold (preserve pos:neg ratio per fold)
skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

tok=AutoTokenizer.from_pretrained(MODEL)
results=[]
for fold,(train_idx,test_idx) in enumerate(skf.split(recs, labels)):
    train=Dataset.from_list([recs[i] for i in train_idx]).map(
        lambda e: {**tok(e["text"], truncation=True, max_length=512),
                   "labels": e["label"]}, remove_columns=["text","label"])
    test=Dataset.from_list([recs[i] for i in test_idx]).map(
        lambda e: {**tok(e["text"], truncation=True, max_length=512),
                   "labels": e["label"]}, remove_columns=["text","label"])
    pos_in_test=sum(1 for i in test_idx if recs[i]["label"]==0)
    print(f"\n=== Fold {fold+1}/5: train={len(train_idx)} test={len(test_idx)} pos_in_test={pos_in_test} ===")

    model=AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2,
            dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable()

    args=TrainingArguments(output_dir=f"P1/Output/models/bert-150m/cv-fold-{fold}",
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        num_train_epochs=3, learning_rate=2e-5,
        logging_steps=50, save_strategy="no", report_to="none", optim="adamw_torch",
        fp16=False, bf16=True, seed=SEED, dataloader_num_workers=0)
    def compute(eval_pred):
        logits,labels=eval_pred
        probs=torch.softmax(torch.tensor(logits).float(),-1)[:,1].numpy()
        return {"val_ap":average_precision_score(1-np.array(labels),probs),
                "val_auc":roc_auc_score(1-np.array(labels),probs)}
    trainer=Trainer(model=model, args=args, train_dataset=train, eval_dataset=test,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=compute)
    trainer.train()
    res=trainer.evaluate()
    res["fold"]=fold+1
    res["pos_in_test"]=pos_in_test
    print(f"Fold {fold+1}: AP={res['eval_val_ap']:.4f} AUC={res['eval_val_auc']:.4f}")
    results.append(res)
    del model; torch.cuda.empty_cache()

aps=[r["eval_val_ap"] for r in results]
aucs=[r["eval_val_auc"] for r in results]
print(f"\n=== 5-FOLD CV SUMMARY ===")
print(f"AP mean={np.mean(aps):.4f} std={np.std(aps):.4f} min={min(aps):.4f} max={max(aps):.4f}")
print(f"AUC mean={np.mean(aucs):.4f} std={np.std(aucs):.4f} min={min(aucs):.4f} max={max(aucs):.4f}")

# save
out={"cv_5fold":{"ap_mean":float(np.mean(aps)),"ap_std":float(np.std(aps)),
                 "auc_mean":float(np.mean(aucs)),"auc_std":float(np.std(aucs)),
                 "per_fold":results},
     "single_holdout_ap":0.9999,
     "single_holdout_n_test":450,"single_holdout_n_pos":2,
     "n_total":len(recs),"n_pos_total":labels.count(0)}
json.dump(out, open(f"{OUT}/bert-150m-cv5.json","w"), indent=2)
print(f"\nSaved {OUT}/bert-150m-cv5.json")