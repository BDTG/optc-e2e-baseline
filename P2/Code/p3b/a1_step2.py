"""A1 buoc 2 — SLM head inference lazily tren test REAL (978 → 2840 mau, ~10ph RX 9060 XT)
   ROI prob raw SLM, ghi p3b-slm-probs.npy
Lưu ý: T1553/T1070 REAL thiếu đủ N, sẽ được mask.
"""
import json, os, sys, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
sys.stdout.reconfigure(line_buffering=True)

BASE = r"C:\Users\BDTG\Desktop\Backup\OpTC-thesis"
DATA = BASE + r"\P1\Output\data\adgen-chains.jsonl"
KEEP = ["T1059","T1547","T1055","T1543","T1003","T1562","T1490","T1553","T1036","T1070","T1218","T1071"]
OUT = BASE + r"\P1\Output\results_phase2\p3b-slm-probs.npy"
MAXLEN = 512

def main():
    import re
    def text_of(ch):
        return " | ".join([(c.get("msg") or "") for c in (ch or [])][-10:])
    rows = []
    for l in open(DATA, encoding="utf-8"):
        try: o = json.loads(l)
        except Exception: continue
        if o.get("label") != 1: continue
        tids = set(re.match(r"(T\d{4})", t).group(1) for t in re.findall(r"T\d{4}(?:\.\d{3})?", str(o.get("technique_id",""))))
        if not tids: continue
        ch = o.get("parent_chain", [])
        if isinstance(ch, str):
            try: ch = json.loads(ch.replace("'", '"'))
            except Exception: ch = []
        src = "LAB" if "LAB" in str(o.get("src","")) else "REAL"
        rows.append({"text": text_of(ch)[:8000], "src": src,
                     "labels":[1 if k in tids else 0 for k in KEEP]})
    te = [r for r in rows if r["src"]=="REAL"]
    print("test REAL:", len(te))

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    # load checkpoint — dung adapter? khong co ckpt duo dang train_ttp_head (no FT full model)
    # luu ckpt? train_ttp_head_slm_adgen.py khong save model — chi save result metadata!
    # => Phai **FINE-TUNE LAI** hoac dung base + head random. Cach dung: reuse base + FT lai trong script nay anh quang?
    # Chon way: re-fine-tune nhanh (3 ep LAB = ~28min) roi inference test REAL.
    import time
    from transformers import get_cosine_schedule_with_warmup
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(KEEP), problem_type="multi_label_classification",
        torch_dtype=torch.float16).to("cuda")
    model.config.pad_token_id = tok.pad_token_id
    model.gradient_checkpointing_enable()

    tr = [r for r in rows if r["src"]=="LAB"]
    class DS(Dataset):
        def __init__(self, rows, tok): self.rows=rows; self.tok=tok
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            r=self.rows[i]; e=self.tok(r["text"], truncation=True, max_length=MAXLEN, padding="max_length")
            return {**e, "labels": torch.tensor(r["labels"], dtype=torch.float)}
    def collate(b):
        return {"input_ids": torch.tensor([x["input_ids"] for x in b]),
                "attention_mask": torch.tensor([x["attention_mask"] for x in b]),
                "labels": torch.stack([x["labels"] for x in b])}
    tr_loader = DataLoader(DS(tr, tok), batch_size=16, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    EPOCHS = int(os.environ.get("EPOCHS", "2"))
    sched = get_cosine_schedule_with_warmup(opt, int(0.06*len(tr_loader)*EPOCHS), len(tr_loader)*EPOCHS)
    loss_fn = nn.BCEWithLogitsLoss()
    t0=time.time()
    for ep in range(EPOCHS):
        model.train(); tot=0.0
        for i,b in enumerate(tr_loader):
            b={k:v.to("cuda") for k,v in b.items()}
            loss = loss_fn(model(**b).logits.float(), b["labels"])
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
            tot+=loss.item()
        print(f"ep{ep} loss {tot/len(tr_loader):.4f} {time.time()-t0:.0f}s")
    # inference test
    model.eval()
    te_loader = DataLoader(DS(te, tok), batch_size=16, shuffle=False, collate_fn=collate)
    preds=[]
    with torch.no_grad():
        for b in te_loader:
            b={k:v.to("cuda") for k,v in b.items()}
            p=torch.sigmoid(model(**b).logits.float()).cpu().numpy()
            preds.append(p)
    P = np.vstack(preds)
    np.save(OUT, P)
    print("SLM probs shape", P.shape, "->", OUT)
    from sklearn.metrics import average_precision_score
    Yte = np.array([r["labels"] for r in te])
    for j,k in enumerate(KEEP):
        if Yte[:,j].sum()>=20:
            ap = average_precision_score(Yte[:,j], P[:,j])
            print(f"SLM {k}: AP {ap:.3f}")
    aps = [average_precision_score(Yte[:,j], P[:,j]) for j in range(len(KEEP)) if Yte[:,j].sum()>=20]
    print("SLM mean AP (verify):", round(float(np.mean(aps)),4))

if __name__ == "__main__":
    main()
