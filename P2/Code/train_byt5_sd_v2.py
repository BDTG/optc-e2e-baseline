import json, random, os, re, base64, torch
os.environ["WANDB_DISABLED"]="true"
os.environ["WANDB_MODE"]="disabled"
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np

def normalize_text(s):
    s=s.lower()
    m=re.search(r'-enc\s+([A-Za-z0-9+/=]{20,})', s)
    if m:
        try:
            b64=m.group(1)
            b64+='='*(-len(b64)%4)
            dec=base64.b64decode(b64).decode('utf-16le', errors='ignore')
            s=s.replace(m.group(1), dec[:200])
        except: pass
    # deeper norm
    s=re.sub(r'c:\\windows\\system32\\svchost\.exe.*','svchost',s)
    s=re.sub(r'c:\\[^\s|]+\.exe','proc',s)
    s=re.sub(r'%[^%]+%','temp',s)
    s=re.sub(r'\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b','guid',s)
    s=re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?\b','ip',s)
    s=re.sub(r'\b[a-f0-9]{32,64}\b','hash',s)
    s=re.sub(r'%temp%','temp',s)
    return s[:800]

def load_s1():
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
    print(f"S1 filtered {len(s1)} (has_cmd+chain+events)")
    return s1, gt

s1, gt = load_s1()
evtx=[json.loads(l) for l in open("/home/vung2/P1/Output/data/evtx-chains.jsonl",encoding='utf-8') if l.strip()]
sd=[json.loads(l) for l in open("/home/vung2/P1/Output/data/sd-chains.jsonl",encoding='utf-8') if l.strip()]
random.seed(42); random.shuffle(sd)
sd_sample = sd[:1000]  # tang 500->1000
print(f"S1 {len(s1)} EVTX {len(evtx)} SD {len(sd)} sample {len(sd_sample)}")

def to_text(chain_obj):
    if "chain" in chain_obj:
        ch=chain_obj["chain"]
        if isinstance(ch, list):
            txt=" | ".join([normalize_text(c.get("msg","") or "") for c in ch[-3:]])
        else:
            txt=normalize_text(str(ch))
    else:
        txt=normalize_text(chain_obj.get("msg",""))
    return txt

combined=[]
for a in s1:
    nid=a.get("nid") or a.get("node_id") or a.get("id")
    label=1 if str(nid) in gt or a.get("is_malicious") else 0
    combined.append((to_text(a), label))
for e in evtx:
    combined.append((to_text(e), 1))
for s in sd_sample:
    combined.append((to_text(s), 1))
alerts_full=[json.loads(l) for l in open("/home/vung2/P1/Output/data/alerts-enriched-v2.jsonl",encoding='utf-8') if l.strip()]
benign=[a for a in alerts_full if str(a.get("nid")) not in gt and a not in s1]
random.shuffle(benign)
benign_sample=benign[:800]
for b in benign_sample:
    combined.append((to_text(b), 0))
print(f"Combined {len(combined)} pos {sum(1 for _,l in combined if l==1)} benign {sum(1 for _,l in combined if l==0)}")
random.shuffle(combined)
split=int(len(combined)*0.8)
train=combined[:split]
test=combined[split:]
print(f"train {len(train)} test {len(test)}")
pos=[x for x in train if x[1]==1]
neg=[x for x in train if x[1]==0]
train_bal = neg + pos*2
random.shuffle(train_bal)
print(f"train_bal {len(train_bal)} pos {sum(1 for _,l in train_bal if l==1)}")

tok=AutoTokenizer.from_pretrained("google/byt5-small")
def tok_fn(b): return tok(b["text"], truncation=True, max_length=512)
train_ds=Dataset.from_list([{"text":t,"label":l} for t,l in train_bal]).map(tok_fn, batched=True)
test_ds=Dataset.from_list([{"text":t,"label":l} for t,l in test]).map(tok_fn, batched=True)
train_ds.set_format("torch", columns=["input_ids","attention_mask","label"])
test_ds.set_format("torch", columns=["input_ids","attention_mask","label"])
model=AutoModelForSequenceClassification.from_pretrained("google/byt5-small", num_labels=2)
def compute_metrics(eval_pred):
    if isinstance(eval_pred, tuple) and len(eval_pred)==2:
        predictions, labels = eval_pred
    else:
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    logits = np.array(predictions)
    if logits.ndim == 3:
        logits = logits[:, 0, :]
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
    if logits.shape[-1] != 2:
        probs = 1/(1+np.exp(-logits.ravel()))
    else:
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = e[:,1] / e.sum(axis=1)
    labels = np.array(labels).astype(int).ravel()
    try: ap = average_precision_score(labels, probs)
    except: ap = 0.0
    try: auc = roc_auc_score(labels, probs)
    except: auc = 0.5
    return {"ap": ap, "auc": auc}

args=TrainingArguments(output_dir="/home/vung2/P1/Output/models/byt5-sd-v2", num_train_epochs=5, per_device_train_batch_size=4, per_device_eval_batch_size=8, eval_strategy="epoch", save_strategy="epoch", logging_steps=20, learning_rate=8e-6, bf16=True, seed=42, report_to="none", max_grad_norm=1.0, warmup_ratio=0.1, weight_decay=0.02, load_best_model_at_end=True, metric_for_best_model="ap", greater_is_better=True)
trainer=Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds, compute_metrics=compute_metrics, processing_class=tok)
trainer.train()
res=trainer.evaluate()
print(f"RESULT ap={res['eval_ap']:.4f} auc={res['eval_auc']:.4f}")
import json as js
open("/home/vung2/P1/Output/results_phase2/byt5-sd-v2-result.json","w").write(js.dumps(res, indent=2))
print("SAVED v2")
