"""FT TTP head: Qwen2.5-0.5B-Instruct + LoRA classification 12 TTP (multi-label, AD-GEN).
Split LAB/REAL chong leak: train = LAB (49.7K), test = REAL subset — moi record danh 1 TTP chinh.
Loss: BCE with logits tren 12 label. Metric: AP per TTP (dung Trung precisio@k la khong can).
Muc tieu: mean AP > 0.673 (TF-IDF head baseline) -> huong den 0.70 (muc Duong cua thay).
"""
import json, math, os, random, sys, time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import average_precision_score
import numpy as np
sys.stdout.reconfigure(line_buffering=True)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA = r"C:\Users\BDTG\optc-bench\adgen-ttp-head-data.jsonl"
OUT = r"C:\Users\BDTG\optc-bench\adgen-ttp-head-slm-result.json"
KEEP = ["T1059","T1547","T1055","T1543","T1003","T1562","T1490","T1553","T1036","T1070","T1218","T1071"]
SEED = 42
EPOCHS = int(os.environ.get("EPOCHS", "3"))
BS = 16
LR = 2e-4
MAXLEN = 512
# Split theo moi truong (dieu kien bat buoc cua thay: khong random split, hoc P0)
# train = LAB (3682), test = REAL (2840) — T1553/T1070 REAL pos <20 → skip, ghi nho result

random.seed(SEED); torch.manual_seed(SEED)

def main():
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    # LAB = train, REAL = test (khong random — dieu kien thay)
    tr = [r for r in rows if r.get("src") == "LAB"]
    te = [r for r in rows if r.get("src") == "REAL"]
    print(f"train LAB {len(tr)} / test REAL {len(te)}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(KEEP), problem_type="multi_label_classification",
        torch_dtype=torch.float16 if torch.cuda.is_bf16_supported() is False else torch.bfloat16,
    )
    model.config.pad_token_id = tok.pad_token_id
    # SLM 0.5B: fine-tune toan bo van nhe (khong LoRA — classification head gan vao), bf16
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    if dev == "cuda" and not torch.cuda.is_bf16_supported():
        model.half()
    model.gradient_checkpointing_enable()

    class DS(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            enc = tok(r["text"], truncation=True, max_length=MAXLEN, padding="max_length")
            return {**enc, "labels": torch.tensor(r["labels"], dtype=torch.float)}

    def collate(b):
        return {"input_ids": torch.tensor([x["input_ids"] for x in b]),
                "attention_mask": torch.tensor([x["attention_mask"] for x in b]),
                "labels": torch.stack([x["labels"] for x in b])}

    tr_loader = DataLoader(DS(tr), batch_size=BS, shuffle=True, collate_fn=collate)
    te_loader = DataLoader(DS(te), batch_size=BS, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = len(tr_loader) * EPOCHS
    sched = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    loss_fn = nn.BCEWithLogitsLoss()

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); tot = 0.0
        for i, b in enumerate(tr_loader):
            b = {k: v.to(dev) for k, v in b.items()}
            out = model(**b)
            loss = loss_fn(out.logits.float(), b["labels"])
            loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item()
            if (i + 1) % 50 == 0:
                print(f"ep{ep} {i+1}/{len(tr_loader)} loss {tot/(i+1):.4f} {time.time()-t0:.0f}s")
        print(f"ep{ep} done avg {tot/len(tr_loader):.4f}")

    model.eval(); preds, ys = [], []
    with torch.no_grad():
        for b in te_loader:
            b = {k: v.to(dev) for k, v in b.items()}
            p = torch.sigmoid(model(**b).logits.float()).cpu().numpy()
            preds.append(p); ys.append(b["labels"].cpu().numpy())
    P = np.vstack(preds); Y = np.vstack(ys)
    res = []
    for j, k in enumerate(KEEP):
        ap = average_precision_score(Y[:, j], P[:, j]) if Y[:, j].sum() > 0 else None
        res.append({"ttp": k, "AP": round(float(ap), 4) if ap else None, "n_test_pos": int(Y[:, j].sum())})
        print(k, res[-1])
    aps = [r["AP"] for r in res if r["AP"]]
    json.dump({"model": MODEL, "n_train": len(tr), "n_test": len(te),
               "per_ttp": res, "mean_AP": round(float(np.mean(aps)), 4),
               "baseline_tfidf_head_AP": 0.673, "latency_train_s": round(time.time() - t0)},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("SAVED", OUT, "mean AP", round(float(np.mean(aps)), 4))

if __name__ == "__main__":
    main()
