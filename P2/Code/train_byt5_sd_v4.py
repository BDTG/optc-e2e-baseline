import json, random, os, re, base64
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

# FIX: build_text dung parent_chain nhu train_byt5_s1.py:32, khong phai msg rong
def build_text_alert(a):
    chain=a.get("parent_chain",[]) or []
    if chain:
        raw=" | ".join([(c.get("msg") or "") for c in chain[-5:]])[:800]
        if raw.strip(): return normalize_text(raw)
    return normalize_text(a.get("msg","") or "")

def build_text_sd(s):
    # sd: {"chain":[{"msg":...}], ...}
    if "chain" in s:
        ch=s["chain"]
        if isinstance(ch, list):
            return " | ".join([normalize_text(c.get("msg","") or "") for c in ch[-3:]])
        else:
            return normalize_text(str(ch))
    return normalize_text(s.get("msg","") or "")

def build_text_evtx(e):
    # evtx: parent_chain or self_label
    if e.get("parent_chain"):
        chain=e["parent_chain"]
        raw=" | ".join([(c.get("cmd") or c.get("msg") or c.get("image") or "") for c in chain[-3:]])
        if raw.strip(): return normalize_text(raw)
    if e.get("self_label"): return normalize_text(e["self_label"])
    return normalize_text(e.get("msg","") or str(e)[:800])

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
    print(f"S1 filtered {len(s1)}")
    return s1, gt

s1, gt = load_s1()
evtx=[json.loads(l) for l in open("/home/vung2/P1/Output/data/evtx-chains.jsonl",encoding='utf-8') if l.strip()]
sd=[json.loads(l) for l in open("/home/vung2/P1/Output/data/sd-chains.jsonl",encoding='utf-8') if l.strip()]
random.seed(42); random.shuffle(sd)
sd_sample = sd[:2000]
print(f"S1 {len(s1)} EVTX {len(evtx)} SD {len(sd)} sample {len(sd_sample)}")
# debug text len
for a in s1[:2]:
    print("S1 txt", repr(build_text_alert(a)[:200]))
for e in evtx[:2]:
    print("EVTX txt", repr(build_text_evtx(e)[:200]))
for s in sd_sample[:2]:
    print("SD txt", repr(build_text_sd(s)[:200]))

combined=[]
for a in s1:
    nid=a.get("nid") or a.get("node_id") or a.get("id")
    label=1 if str(nid) in gt or a.get("is_malicious") else 0
    combined.append((build_text_alert(a), label))
for e in evtx:
    combined.append((build_text_evtx(e), 1))
for s in sd_sample:
    combined.append((build_text_sd(s), 1))
alerts_full=[json.loads(l) for l in open("/home/vung2/P1/Output/data/alerts-enriched-v2.jsonl",encoding='utf-8') if l.strip()]
benign=[a for a in alerts_full if str(a.get("nid")) not in gt and a not in s1]
random.shuffle(benign)
benign_sample=benign[:1200]
for b in benign_sample:
    combined.append((build_text_alert(b), 0))
print(f"Combined {len(combined)} pos {sum(1 for _,l in combined if l==1)} benign {sum(1 for _,l in combined if l==0)}")
# check empty rate
empty=sum(1 for t,_ in combined if not t.strip())
print(f"empty text {empty}/{len(combined)}")
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

args=TrainingArguments(output_dir="/home/vung2/P1/Output/models/byt5-sd-v4", num_train_epochs=5, per_device_train_batch_size=4, per_device_eval_batch_size=8, eval_strategy="epoch", save_strategy="epoch", logging_steps=20, learning_rate=6e-6, bf16=True, seed=42, report_to="none", max_grad_norm=1.0, warmup_ratio=0.1, weight_decay=0.02, load_best_model_at_end=True, metric_for_best_model="ap", greater_is_better=True)
trainer=Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds, compute_metrics=compute_metrics, processing_class=tok)
trainer.train()
res=trainer.evaluate()
print(f"RESULT v4 ap={res['eval_ap']:.4f} auc={res['eval_auc']:.4f}")
import json as js
open("/home/vung2/P1/Output/results_phase2/byt5-sd-v4-result.json","w").write(js.dumps(res, indent=2))
print("SAVED v4")

# also eval S1-only test with same model (honest)
# build S1 80/20 split like train_byt5_s1
import random as _r
s1_texts=[build_text_alert(a) for a in s1]
s1_labels=[1 if str(s1[i].get("nid")) in gt else 0 for i in range(len(s1))]
idx=list(range(len(s1))); _r.seed(42); _r.shuffle(idx)
s1_test=[(s1_texts[i], s1_labels[i]) for i in idx[int(len(s1)*0.8):]]
from datasets import Dataset as _DS
tok2=tok
def tok_fn2(b): return tok2(b["text"], truncation=True, max_length=512)
test_ds2=_DS.from_list([{"text":t,"label":l} for t,l in s1_test]).map(tok_fn2, batched=True)
test_ds2.set_format("torch", columns=["input_ids","attention_mask","label"])
from transformers import TrainingArguments as _TA, Trainer as _TR
args2=_TA(output_dir="/tmp/eval_v4_s1", per_device_eval_batch_size=8, report_to="none", bf16=True)
trainer2=_TR(model=model, args=args2, eval_dataset=test_ds2, processing_class=tok2, compute_metrics=compute_metrics)
res2=trainer2.evaluate()
print(f"S1-only TEST ap={res2['eval_ap']:.4f} auc={res2['eval_auc']:.4f} n={len(s1_test)} pos={sum(l for _,l in s1_test)}")
open("/home/vung2/P1/Output/results_phase2/byt5-v4-s1only-result.json","w").write(js.dumps({"s1_test":res2,"n":len(s1_test)}, indent=2))
