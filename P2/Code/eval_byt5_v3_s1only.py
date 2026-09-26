import json, random, os, re, base64
os.environ["WANDB_DISABLED"]="true"
os.environ["WANDB_MODE"]="disabled"
import numpy as np, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import average_precision_score, roc_auc_score

def normalize_text(s):
    s=s.lower()
    m=re.search(r'-enc\s+([A-Za-z0-9+/=]{20,})', s)
    if m:
        try:
            b64=m.group(1); b64+='='*(-len(b64)%4)
            dec=base64.b64decode(b64).decode('utf-16le', errors='ignore')
            s=s.replace(m.group(1), dec[:200])
        except: pass
    s=re.sub(r'c:\\\\windows\\\\system32\\\\svchost\.exe.*','svchost',s)
    s=re.sub(r'c:\\\\[^\s|]+\.exe','proc',s)
    s=re.sub(r'%[^%]+%','temp',s)
    s=re.sub(r'\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b','guid',s)
    s=re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?\b','ip',s)
    s=re.sub(r'\b[a-f0-9]{32,64}\b','hash',s)
    s=re.sub(r'%temp%','temp',s)
    return s[:800]

# Load S1 exactly like train
gt=set(json.load(open("/home/vung2/P1/Output/data/gt_and_scores.json"))["gt_nids"])
alerts_all=[json.loads(l) for l in open("/home/vung2/P1/Output/data/alerts-enriched-v2.jsonl",encoding='utf-8') if l.strip()]
def has_cmd2(a):
    for c in a.get("parent_chain",[]) or []:
        m=c.get("msg") or ""
        if "| cmd:" in m:
            v=m.split("| cmd:",1)[1].strip()
            if v and v.lower()!="none": return True
    return False
def has_chain2(a): return len(a.get("parent_chain",[]) or [])>=2
def has_events2(a): return len(a.get("event_seq",[]) or [])>=3
s1=[a for a in alerts_all if has_cmd2(a) and has_chain2(a) and has_events2(a)]
print(f"S1 filtered {len(s1)}")
for a in s1[:2]:
    print(a.get("nid"), a.get("is_malicious"))

# Build S1-only dataset (pos if in gt)
def to_text_alert(a):
    return " | ".join([normalize_text(c.get("msg","") or "") for c in a.get("parent_chain",[])[-3:]])

s1_data=[(to_text_alert(a), 1 if str(a.get("nid")) in gt else 0) for a in s1]
print(f"S1-only {len(s1_data)} pos {sum(1 for _,l in s1_data if l==1)}")
# Reproduce same 80/20 split seed 42 as standalone S1 eval? Use same shuffle as combined would have isolated S1 test roughly.
# To be fair, evaluate on WHOLE S1 as holdout (since v3 was trained on S1 subset of combined). We also do a clean 80/20 S1 split matching train_byt5_s1 logic.
random.seed(42); random.shuffle(s1_data)
split=int(len(s1_data)*0.8)
s1_train=s1_data[:split]; s1_test=s1_data[split:]
print(f"S1 train {len(s1_train)} test {len(s1_test)} pos_test {sum(1 for _,l in s1_test if l==1)}")

# Also evaluate on full S1 (to see leakage vs true)
tok=AutoTokenizer.from_pretrained("google/byt5-small")
def tok_fn(b): return tok(b["text"], truncation=True, max_length=512)

# Load v3 checkpoint (best)
import os as _os
ckpt="/home/vung2/P1/Output/models/byt5-sd-v3/checkpoint-3972"
# best is 3972 epoch3 AP 0.925088 (not 6620)
print(f"ckpt: {ckpt} exists={_os.path.isdir(ckpt)}")
model=AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)
tok2=tok

def eval_on(data, name):
    ds=Dataset.from_list([{"text":t,"label":l} for t,l in data]).map(tok_fn, batched=True)
    ds.set_format("torch", columns=["input_ids","attention_mask","label"])
    args=TrainingArguments(output_dir="/tmp/eval_byt5_s1", per_device_eval_batch_size=8, report_to="none", bf16=True)
    trainer=Trainer(model=model, args=args, eval_dataset=ds, processing_class=tok2,
        compute_metrics=lambda p: _metrics(p))
    def _metrics(eval_pred):
        if isinstance(eval_pred, tuple) and len(eval_pred)==2:
            preds, labels = eval_pred
        else:
            preds=eval_pred.predictions; labels=eval_pred.label_ids
        if isinstance(preds, tuple): preds=preds[0]
        logits=np.array(preds)
        if logits.ndim==3: logits=logits[:,0,:]
        if logits.ndim==1: logits=logits.reshape(-1,1)
        if logits.shape[-1]!=2:
            probs=1/(1+np.exp(-logits.ravel()))
        else:
            e=np.exp(logits - np.max(logits,axis=1,keepdims=True))
            probs=e[:,1]/e.sum(axis=1)
        labels=np.array(labels).astype(int).ravel()
        try: ap=average_precision_score(labels, probs)
        except: ap=0.0
        try: auc=roc_auc_score(labels, probs)
        except: auc=0.5
        return {"ap":ap,"auc":auc}
    # patch compute_metrics after init
    trainer.compute_metrics=_metrics
    res=trainer.evaluate()
    print(f"{name}: {res}")
    return res

def _metrics(eval_pred):
    if isinstance(eval_pred, tuple) and len(eval_pred)==2:
        preds, labels = eval_pred
    else:
        preds=eval_pred.predictions; labels=eval_pred.label_ids
    if isinstance(preds, tuple): preds=preds[0]
    logits=np.array(preds)
    if logits.ndim==3: logits=logits[:,0,:]
    if logits.ndim==1: logits=logits.reshape(-1,1)
    if logits.shape[-1]!=2:
        probs=1/(1+np.exp(-logits.ravel()))
    else:
        e=np.exp(logits - np.max(logits,axis=1,keepdims=True))
        probs=e[:,1]/e.sum(axis=1)
    labels=np.array(labels).astype(int).ravel()
    try: ap=average_precision_score(labels, probs)
    except: ap=0.0
    try: auc=roc_auc_score(labels, probs)
    except: auc=0.5
    return {"ap":ap,"auc":auc}

r_test=eval_on(s1_test, "S1-test 20% (94)")
r_full=eval_on(s1_data, "S1-full 474")

# Save
open("/home/vung2/P1/Output/results_phase2/byt5-v3-s1only-result.json","w").write(json.dumps({"s1_test_20pct":r_test,"s1_full_474":r_full,"s1_test_n":len(s1_test),"s1_test_pos":sum(1 for _,l in s1_test if l==1)},indent=2))
print("SAVED byt5-v3-s1only-result.json")
