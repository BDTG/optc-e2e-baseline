import json, torch, re, base64, random
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

DATA="P1/Output/data/alerts-enriched-v2.jsonl"
GT="P1/Output/data/gt_and_scores.json"
EVTX="P1/Output/data/evtx-chains.jsonl"
MODEL="google/byt5-small"
OUT="P1/Output/models/byt5-combined"

def decode_b64(m):
    try:
        if "-enc" in m.lower():
            b64=m.lower().split("-enc",1)[1].strip().split()[0]
            raw=base64.b64decode(b64)
            try: return raw.decode("utf-16le")
            except: return raw.decode("utf-8", errors="ignore")
    except: pass
    return m
def normalize_text(s):
    s=s.lower()
    s=decode_b64(s)
    s=re.sub(r"[a-z]:\\[^\s|]+", lambda m: m.group(0).split("\\")[-1], s)
    s=re.sub(r"\s+", " ", s).strip()
    return s
def build_text_opc(a):
    chain=a.get("parent_chain",[]) or []
    raw=" | ".join([(c.get("msg") or "") for c in chain[-5:]])[:800] or (a.get("msg") or "")[:800]
    return normalize_text(raw)
def build_text_evtx(a):
    # evtx-chains.jsonl has fields: chain, cmd, etc - fallback to msg/chain
    if "parent_chain" in a: return build_text_opc(a)
    raw=a.get("chain") or a.get("msg") or str(a)[:800]
    return normalize_text(raw)
def has_cmd(r):
    for c in r.get("parent_chain",[]) or []:
        m=c.get("msg") or ""
        if "| cmd:" in m:
            v=m.split("| cmd:",1)[1].strip()
            if v and v.lower()!="none": return True
    return False
def has_chain(r): return len(r.get("parent_chain",[]) or [])>=2
def has_events(r): return len(r.get("event_seq",[]) or [])>=3

gt=set(json.load(open(GT))["gt_nids"])
alerts=[json.loads(l) for l in open(DATA,encoding='utf-8') if l.strip()]
S1_idx=[i for i in range(len(alerts)) if has_cmd(alerts[i]) and has_chain(alerts[i]) and has_events(alerts[i])]
S1=[alerts[i] for i in S1_idx]
s1_texts=[build_text_opc(a) for a in S1]
s1_labels=[1 if str(S1[i]["nid"]) in gt else 0 for i in range(len(S1))]
print(f"S1 n={len(S1)} pos={sum(s1_labels)}")

# EVTX
evtx_alerts=[json.loads(l) for l in open(EVTX,encoding='utf-8') if l.strip()]
evtx_texts=[]
evtx_labels=[]
for a in evtx_alerts:
    txt=build_text_evtx(a)
    # suspicious field - check multiple keys
    is_susp = a.get("is_suspicious") or a.get("suspicious") or (a.get("label")==1) or ("mal" in str(a.get("label","")).lower())
    # evtx-chains.jsonl: 65 suspicious / 468 - use is_malicious if exists
    if "is_malicious" in a: is_susp = a["is_malicious"]==1
    if "ground_truth" in a: is_susp = a["ground_truth"]==1
    evtx_texts.append(txt)
    evtx_labels.append(1 if is_susp else 0)
print(f"EVTX n={len(evtx_texts)} pos={sum(evtx_labels)}")

# combined
texts=s1_texts+evtx_texts
labels=s1_labels+evtx_labels
print(f"Combined n={len(texts)} pos={sum(labels)} prev={sum(labels)/len(labels):.3f}")

# stratified split 80/20
from sklearn.model_selection import StratifiedShuffleSplit
sss=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(texts, labels))
train_texts=[texts[i] for i in train_idx]
train_labels=[labels[i] for i in train_idx]
test_texts=[texts[i] for i in test_idx]
test_labels=[labels[i] for i in test_idx]
print(f" train {len(train_texts)} pos {sum(train_labels)} | test {len(test_texts)} pos {sum(test_labels)}")

tok=AutoTokenizer.from_pretrained(MODEL)
def tok_fn(ex): return tok(ex["text"], truncation=True, padding="max_length", max_length=512)
train_ds=Dataset.from_dict({"text":train_texts,"labels":train_labels}).map(tok_fn, batched=True)
test_ds=Dataset.from_dict({"text":test_texts,"labels":test_labels}).map(tok_fn, batched=True)
train_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
test_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

model=AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
def compute_metrics(p):
    logits=p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    probs=torch.softmax(torch.tensor(logits), dim=-1)[:,1].numpy()
    y=np.array(p.label_ids)
    ap=average_precision_score(y, probs) if len(set(y))>1 else 0
    try: auc=roc_auc_score(y, probs)
    except: auc=0
    return {"ap": ap, "auc": auc}

args=TrainingArguments(
    output_dir=OUT,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model="ap",
    seed=42,
    report_to="none",
)
trainer=Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds, compute_metrics=compute_metrics, processing_class=tok)
trainer.train()
res=trainer.evaluate()
print(f"RESULT ap={res['eval_ap']:.4f} auc={res['eval_auc']:.4f}")
trainer.save_model(OUT)
tok.save_pretrained(OUT)
import os, json as js
os.makedirs("P1/Output/results_phase2", exist_ok=True)
with open("P1/Output/results_phase2/byt5-combined-result.json","w") as f:
    js.dump(res, f, indent=2)
print("SAVED", OUT)
