"""Buoc 4/2: Distill BERT 150M -> TinyBERT 4L/312D (4M params).
Custom loop (khong Trainer) để tránh Trainer strip custom keys.
"""
import json, os, random, torch
import numpy as np, torch.nn.functional as F
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
TEACHER_PATH="P1/Output/models/bert-150m"
STUDENT="huawei-noah/TinyBERT_General_4L_312D"
ALERTS="P1/Output/alerts-enriched-v2.jsonl"
GT="P1/Output/gt_and_scores.json"
OUT="P1/Output/models/tinybert-4m-distilled"
os.makedirs(OUT, exist_ok=True)

# ---- data ----
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

# ---- teacher soft labels ----
print("Loading teacher (ModernBERT 150M)...")
tch_tok=AutoTokenizer.from_pretrained(TEACHER_PATH)
teacher=AutoModelForSequenceClassification.from_pretrained(TEACHER_PATH,
        dtype=torch.bfloat16).to("cuda").eval()

tch_tr=Dataset.from_list(train).map(
    lambda e: tch_tok(e["text"], truncation=True, max_length=512),
    remove_columns=["text","label"])
tch_hd=Dataset.from_list(hold).map(
    lambda e: tch_tok(e["text"], truncation=True, max_length=512),
    remove_columns=["text","label"])

def collate_tch(b): return tch_tok.pad(b, return_tensors="pt")
teacher_logits=[]
loader=DataLoader(tch_tr, batch_size=16, collate_fn=collate_tch)
with torch.no_grad():
    for b in loader:
        ids={k:v.to("cuda") for k,v in b.items()}
        out=teacher(**ids)
        teacher_logits.append(out.logits.float().cpu())
teacher_logits=torch.cat(teacher_logits,0)
T=2.0
soft=F.softmax(teacher_logits/T, dim=-1)  # (n,2)
print(f"teacher_logits {teacher_logits.shape}, soft {soft.shape}")
del teacher; torch.cuda.empty_cache()

# ---- student training ----
print("Loading student (TinyBERT 4L/312D)...")
st_tok=AutoTokenizer.from_pretrained(STUDENT)
st_model=AutoModelForSequenceClassification.from_pretrained(STUDENT,
        num_labels=2, dtype=torch.float32).to("cuda")
st_model.train()

# tokenize with student tokenizer (no labels in collator, we manage manually)
st_tr=Dataset.from_list(train).map(
    lambda e: st_tok(e["text"], truncation=True, max_length=512),
    remove_columns=["text"])
st_hd=Dataset.from_list(hold).map(
    lambda e: {**st_tok(e["text"], truncation=True, max_length=512),
               "labels": e["label"]},
    remove_columns=["text"])

# collator: returns dict of input_ids/attention_mask [+token_type_ids]
collator=DataCollatorWithPadding(st_tok)

# build torch dataset pairing soft logits với tokenized inputs
class TrainDS(torch.utils.data.Dataset):
    def __init__(self, ds, soft, labels):
        self.ds=ds; self.soft=soft; self.lab=labels
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        item={k:torch.tensor(v) for k,v in self.ds[i].items()}
        item["labels"]=torch.tensor(self.lab[i], dtype=torch.long)
        item["teacher_soft"]=self.soft[i]
        return item

train_ds=TrainDS(st_tr, soft, [r["label"] for r in train])
train_loader=DataLoader(train_ds, batch_size=8, shuffle=True,
                        collate_fn=lambda b: {**collator(b),
                                              "teacher_soft": torch.stack([d["teacher_soft"] for d in b]),
                                              "labels": torch.tensor([d["labels"] for d in b], dtype=torch.long)})

# eval loader
def collate_eval(b):
    out=collator(b)
    out["labels"]=torch.tensor([d["labels"] for d in b], dtype=torch.long)
    return out
hold_loader=DataLoader(st_hd, batch_size=16, collate_fn=collate_eval)

EPOCHS=3
total_steps=len(train_loader)*EPOCHS
opt=torch.optim.AdamW(st_model.parameters(), lr=2e-5)
sched=get_linear_schedule_with_warmup(opt, int(total_steps*0.1), total_steps)

alpha=0.7
print(f"Training TinyBERT 4M KD: {len(train_loader)} steps/epoch x {EPOCHS} = {total_steps}")
for epoch in range(EPOCHS):
    st_model.train()
    losses=[]
    for step,b in enumerate(train_loader):
        soft_b=b.pop("teacher_soft").to("cuda")
        lab_b=b.pop("labels").to("cuda")
        ids={k:v.to("cuda") for k,v in b.items()}
        out=st_model(**ids)
        loss_kd=F.kl_div(F.log_softmax(out.logits/T,-1), soft_b, reduction="batchmean")*(T*T)
        loss_ce=F.cross_entropy(out.logits, lab_b)
        loss=alpha*loss_kd + (1-alpha)*loss_ce
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
    avg=float(np.mean(losses))
    # eval
    st_model.eval(); preds=[]; labs=[]
    with torch.no_grad():
        for b in hold_loader:
            lab=b.pop("labels")
            ids={k:v.to("cuda") for k,v in b.items()}
            out=st_model(**ids)
            probs=F.softmax(out.logits.float(),-1)[:,1].cpu()
            preds.append(probs); labs.append(lab)
    preds=torch.cat(preds).numpy(); labs=torch.cat(labs).numpy()
    ap=average_precision_score(1-labs, preds)
    try: auc=roc_auc_score(1-labs, preds)
    except: auc=-1.0
    print(f"epoch {epoch+1}/{EPOCHS} loss={avg:.4f} val_ap={ap:.4f} val_auc={auc:.4f}")

# save
res={"final_ap":float(ap),"final_auc":float(auc),"n_train":len(train),"n_hold":len(hold)}
st_model.save_pretrained(OUT); st_tok.save_pretrained(OUT)
json.dump(res, open(os.path.join(OUT,"result.json"),"w"), indent=2)
print(f"=== DONE: AP={ap:.4f} AUC={auc:.4f} ===")