import json, torch, numpy as np, re, base64
from sklearn.metrics import average_precision_score, roc_auc_score
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

DATA="P1/Output/data/alerts-enriched-v2.jsonl"
GT="P1/Output/data/gt_and_scores.json"
MODEL="google/byt5-small"  # byte-level 300M, no BPE
OUT="P1/Output/models/byt5-s1"

def decode_b64(m):
    # powershell -enc <base64>
    try:
        if "-enc" in m.lower():
            b64=m.lower().split("-enc",1)[1].strip().split()[0]
            # b64 is utf-16le base64
            raw=base64.b64decode(b64)
            try: return raw.decode("utf-16le")
            except: return raw.decode("utf-8", errors="ignore")
    except: pass
    return m

def normalize_text(s):
    s=s.lower()
    s=decode_b64(s)
    # normalize path: c:\windows\system32\svchost.exe -k netsvcs -> svchost
    s=re.sub(r"[a-z]:\\[^\s|]+", lambda m: m.group(0).split("\\")[-1], s)
    s=re.sub(r"\s+", " ", s).strip()
    return s

def build_text(a):
    chain=a.get("parent_chain",[]) or []
    raw=" | ".join([(c.get("msg") or "") for c in chain[-5:]])[:800] or (a.get("msg") or "")[:800]
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
texts=[build_text(a) for a in S1]
labels=[1 if str(S1[i]["nid"]) in gt else 0 for i in range(len(S1))]
print(f"S1 n={len(S1)} pos={sum(labels)}")

# 380 train / 94 test + oversample pos 5x
import random; random.seed(42)
idx=list(range(len(S1))); random.shuffle(idx)
train_texts=[texts[i] for i in idx[:380]]
train_labels=[labels[i] for i in idx[:380]]
# oversample positives 5x
pos_texts=[t for t,l in zip(train_texts,train_labels) if l==1]
pos_labels=[1]*len(pos_texts)*5
train_texts=train_texts+pos_texts*5
train_labels=train_labels+pos_labels
print(f" train {len(train_texts)} (oversampled pos {sum(train_labels)})")
test_texts=[texts[i] for i in idx[380:]]
test_labels=[labels[i] for i in idx[380:]]

tok=AutoTokenizer.from_pretrained(MODEL)
def tok_fn(ex): return tok(ex["text"], truncation=True, padding="max_length", max_length=512)
train_ds=Dataset.from_dict({"text":train_texts,"labels":train_labels}).map(tok_fn, batched=True)
test_ds=Dataset.from_dict({"text":test_texts,"labels":test_labels}).map(tok_fn, batched=True)
train_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
test_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

model=AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

def compute_metrics(p):
    import numpy as np
    logits=p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    probs=torch.softmax(torch.tensor(logits), dim=-1)[:,1].numpy()
    y=np.array(p.label_ids)
    # pos=1
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
    fp16=False,  # byt5 fp16 unstable on 4B? keep fp32
    load_best_model_at_end=True,
    metric_for_best_model="ap",
    seed=42,
    report_to="none",
)
trainer=Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds, compute_metrics=compute_metrics, tokenizer=tok)
trainer.train()
res=trainer.evaluate()
print(f"RESULT ap={res['eval_ap']:.4f} auc={res['eval_auc']:.4f}")
trainer.save_model(OUT)
tok.save_pretrained(OUT)
with open("P1/Output/results_phase2/byt5-s1-result.json","w") as f:
    json.dump(res, f, indent=2)
print("SAVED", OUT)
