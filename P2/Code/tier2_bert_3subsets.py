"""
So sanh BERT 150M tren 3 subsets khac nhau de xem class balance co giup gi:
  S1: ca 3 truc (474 alerts, 9 pos, prev 1.9%)
  S2: >= 2 truc (1017 alerts, ~12 pos, prev ~1.2%)
  S3: toan bo 2250 (12 pos, prev 0.53%) - baseline

Moi subset: Stratified 5-fold CV.
So sanh AP/AUC/F1/Recall@K.
"""
import json, random, numpy as np, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                              precision_recall_curve)
from sklearn.model_selection import StratifiedKFold

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
def has_cmd(r):
    chain=r.get("parent_chain",[]) or []
    for c in chain:
        msg=c.get("msg","") or ""
        if "| cmd:" in msg:
            v=msg.split("| cmd:",1)[1].strip()
            if v and v.lower()!="none": return True
    return False
def has_chain(r): return len(r.get("parent_chain",[]) or [])>=2
def has_events(r): return len(r.get("event_seq",[]) or [])>=3

# build all recs (label: 0=mal, 1=benign)
all_recs=[{"text":build_text(a),"label":0 if str(a["nid"]) in gt else 1,
           "nid":a["nid"]} for a in alerts]
print(f"All: {len(all_recs)}, pos={sum(r['label']==0 for r in all_recs)}")

# filter subsets (use raw alerts, not all_recs - those stripped parent_chain/event_seq)
def flags(a):
    return (has_cmd(a), has_chain(a), has_events(a))
s1=[a for a in alerts if all(flags(a))]  # all 3
s2=[a for a in alerts if sum(flags(a))>=2]  # >=2
s3=alerts
print(f"S1 (all 3): {len(s1)}, pos={sum(str(a['nid']) in gt for a in s1)}")
print(f"S2 (>=2):   {len(s2)}, pos={sum(str(a['nid']) in gt for a in s2)}")
print(f"S3 (all):   {len(s3)}, pos={sum(str(a['nid']) in gt for a in s3)}")

# recs for each subset: build with build_text from RAW alert
def recs_for(raw_alerts):
    return [{"text":build_text(a),
             "label":0 if str(a["nid"]) in gt else 1,
             "nid":a["nid"]} for a in raw_alerts]

tok=AutoTokenizer.from_pretrained(MODEL)

def make_ds(rec):
    return {**tok(rec["text"], truncation=True, max_length=512),
            "labels": rec["label"]}

def run_cv(name, raw_alerts, n_folds=5):
    print(f"\n{'='*60}")
    print(f"=== {name}: {len(raw_alerts)} samples, "
          f"{sum(str(a['nid']) in gt for a in raw_alerts)} positives ===")
    print(f"{'='*60}")
    recs=recs_for(raw_alerts)
    labels=[r["label"] for r in recs]
    if sum(1 for l in labels if l==0) < n_folds:
        # not enough positives for n folds; reduce folds
        n_folds = max(2, sum(1 for l in labels if l==0))
        print(f"  [warn] reducing folds to {n_folds} due to few positives")
    skf=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    results=[]
    for fold,(ti,tei) in enumerate(skf.split(recs,labels)):
        train=Dataset.from_list([recs[i] for i in ti]).map(make_ds, remove_columns=["text","label","nid"])
        test=Dataset.from_list([recs[i] for i in tei]).map(make_ds, remove_columns=["text","label","nid"])
        print(f"  fold {fold+1}/{n_folds}: train={len(ti)} test={len(tei)} "
              f"pos_test={sum(1 for i in tei if recs[i]['label']==0)}")
        model=AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2,
                dtype=torch.bfloat16).to("cuda")
        model.gradient_checkpointing_enable()
        args=TrainingArguments(output_dir=f"P1/Output/models/bert-150m/{name}-fold-{fold}",
            per_device_train_batch_size=4, gradient_accumulation_steps=4,
            num_train_epochs=3, learning_rate=2e-5,
            logging_steps=50,
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=False,
            report_to="none", optim="adamw_torch",
            fp16=False, bf16=True, seed=SEED, dataloader_num_workers=0)
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
        print(f"    AP={res['eval_val_ap']:.4f} AUC={res['eval_val_auc']:.4f}")
        results.append(res)
        del model; torch.cuda.empty_cache()

    aps=[r["eval_val_ap"] for r in results]
    aucs=[r["eval_val_auc"] for r in results]
    summary={"name":name,"n":len(raw_alerts),
             "pos":sum(str(a["nid"]) in gt for a in raw_alerts),
             "ap_mean":float(np.mean(aps)),"ap_std":float(np.std(aps)),
             "auc_mean":float(np.mean(aucs)),"auc_std":float(np.std(aucs)),
             "per_fold":results}
    print(f"  >> {name}: AP={summary['ap_mean']:.4f}±{summary['ap_std']:.4f}, "
          f"AUC={summary['auc_mean']:.4f}±{summary['auc_std']:.4f}")
    return summary

all_results=[]
all_results.append(run_cv("S1_all3", s1))
all_results.append(run_cv("S2_ge2",  s2))
all_results.append(run_cv("S3_all",   s3))

# save
out={"subsets":all_results,"timestamp":"2026-08-28",
     "note":"BERT 150M 5-fold CV tren 3 subsets khac nhau"}
json.dump(out, open("P1/Output/results_phase2/bert-150m-3subsets.json","w"), indent=2)
print(f"\n=== SUMMARY ===")
for r in all_results:
    print(f"  {r['name']:10s}: n={r['n']:4d} pos={r['pos']:3d} "
          f"AP={r['ap_mean']:.4f}±{r['ap_std']:.4f} AUC={r['auc_mean']:.4f}±{r['auc_std']:.4f}")