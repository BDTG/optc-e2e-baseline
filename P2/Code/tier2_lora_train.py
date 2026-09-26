"""
Tier-2 SLM LoRA fine-tune (RQ1a) -- CPU-only fallback.
QLoRA/4-bit requires CUDA; on CPU we train LoRA (r=16) in fp32 on Qwen2.5-0.5B.
Dataset: alerts-enriched-v2.jsonl (2250) labeled via gt_and_scores.json (114 gt_nids).
Note.md:111 -- LoRA is the one untested axis of {zero-shot, few-shot, LoRA}.
"""
import json, os, random, sys
import numpy as np
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ALERTS = "P1/Output/alerts-enriched-v2.jsonl"
GT = "P1/Output/gt_and_scores.json"
OUT = "P1/Output/models/lora-05b"
os.makedirs(OUT, exist_ok=True)

# ---- load labels ----
gt = set(json.load(open(GT))["gt_nids"])
# alerts are ranked; top of list are detector hits. Take all 2250, label by gt.
alerts = [json.loads(l) for l in open(ALERTS) if l.strip()]
print(f"loaded {len(alerts)} alerts, {len(gt)} gt_nids")

def build_text(a):
    chain = a.get("parent_chain", []) or []
    seq = a.get("event_seq", []) or []
    parts = []
    for c in chain[-5:]:
        parts.append(c.get("msg", ""))
    for e in seq[-5:]:
        parts.append(f"{e.get('src_msg','')} -> {e.get('dst_msg','')}")
    txt = " | ".join([p for p in parts if p])
    label = "MALICIOUS" if str(a.get("nid")) in gt else "BENIGN"
    return f"Process provenance chain: {txt}\nVerdict: {label}"

records = [{"text": build_text(a), "label": (0 if str(a["nid"]) in gt else 1)} for a in alerts]
pos = sum(1 for r in records if r["label"] == 0)
print(f"pos(mal)={pos} neg(benign)={len(records)-pos}")

# ---- split 1800 train / 450 holdout ----
random.shuffle(records)
train_rec = records[:1800]
hold_rec = records[1800:]
json.dump(hold_rec, open(os.path.join(OUT, "holdout.json"), "w"), indent=0)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token

def tok_fn(ex):
    return tok(ex["text"], truncation=True, max_length=512,
               padding="max_length", return_tensors=None)

train_ds = Dataset.from_list(train_rec).map(tok_fn, remove_columns=["text","label"])
hold_ds = Dataset.from_list(hold_rec).map(tok_fn, remove_columns=["text","label"])

# ---- model + LoRA ----
model = AutoModelForCausalLM.from_pretrained(MODEL,
        dtype=torch.float16, device_map="auto")
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=OUT, per_device_train_batch_size=4,
    gradient_accumulation_steps=4, num_train_epochs=1,
    learning_rate=2e-4, logging_steps=10,
    save_strategy="epoch", save_total_limit=3,
    load_best_model_at_end=False, report_to="none", fp16=True, bf16=False,
    optim="adamw_torch", seed=SEED, dataloader_num_workers=0,
    resume_from_checkpoint=True)

trainer = Trainer(model=model, args=args, train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False,
            return_tensors="pt"))
ckpt = None
if os.path.isdir(OUT) and any(f.startswith("checkpoint") for f in os.listdir(OUT)):
    ckpts = sorted([f for f in os.listdir(OUT) if f.startswith("checkpoint")])
    ckpt = os.path.join(OUT, ckpts[-1])
    print(f"=== RESUME from {ckpt} ===")
trainer.train(resume_from_checkpoint=ckpt)
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("=== LoRA TRAIN DONE ===")
