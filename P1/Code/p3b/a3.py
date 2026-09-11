"""A3 — Size test: Qwen2.5-1.5B-Instruct (3x params) head FT, split LAB->REAL (giong A1).
Ly do khong Qwen3-1.7B: transformers pin 4.41.2 (ROCm Windows compat) chua co Qwen3 arch.
FP16 + BS8 (1.5B full-FT fp32 = 20GB > 16GB VRAM) + clip 1.0 + NaN guard.
Neu OOM/NaN -> ket luan: 1.5B can setup khac (LoRA / may thay).
"""
import json, re, sys, os, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import average_precision_score
sys.stdout.reconfigure(line_buffering=True)

BASE = r"C:\Users\BDTG\Desktop\Backup\OpTC-thesis"
DATA = BASE + r"\P1\Output\data\adgen-chains.jsonl"
OUTD = BASE + r"\P1\Output\results_phase2"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
KEEP = ["T1059","T1547","T1055","T1543","T1003","T1562","T1490","T1553","T1036","T1070","T1218","T1071"]
EPOCHS = int(os.environ.get("EPOCHS", "2")); BS = 8; LR = 2e-4; MAXLEN = 512

def text_of(ch):
    return " | ".join([(c.get("msg") or "") for c in (ch or [])][-10:])

def main():
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
    tr = [r for r in rows if r["src"]=="LAB"]; te = [r for r in rows if r["src"]=="REAL"]
    print(f"LAB {len(tr)} / REAL {len(te)} | MODEL {MODEL} fp16 BS{BS}", flush=True)
    Yte = np.array([r["labels"] for r in te])

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    torch.manual_seed(42)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=len(KEEP), problem_type="multi_label_classification",
            torch_dtype=torch.float16).to("cuda")
    except Exception as e:
        print("LOAD FAIL (OOM?):", str(e)[:200], flush=True)
        json.dump({"status":"load_fail","error":str(e)[:200]}, open(OUTD+r"\p3b-qwen15-result.json","w")); return
    model.config.pad_token_id = tok.pad_token_id
    model.gradient_checkpointing_enable()
    print("VRAM after load GB:", round(torch.cuda.memory_allocated()/1e9,2), flush=True)
    class DS(Dataset):
        def __init__(self, rows): self.rows=rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            r=self.rows[i]; e=tok(r["text"], truncation=True, max_length=MAXLEN, padding="max_length")
            return {**e, "labels": torch.tensor(r["labels"], dtype=torch.float)}
    def collate(b):
        return {"input_ids": torch.tensor([x["input_ids"] for x in b]),
                "attention_mask": torch.tensor([x["attention_mask"] for x in b]),
                "labels": torch.stack([x["labels"] for x in b])}
    tr_loader = DataLoader(DS(tr), batch_size=BS, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(0.06*len(tr_loader)*EPOCHS), len(tr_loader)*EPOCHS)
    loss_fn = nn.BCEWithLogitsLoss()
    t0=time.time()
    for ep in range(EPOCHS):
        model.train(); tot=0.0
        for i,b in enumerate(tr_loader):
            b={k:v.to("cuda") for k,v in b.items()}
            try:
                loss = loss_fn(model(**b).logits.float(), b["labels"])
            except RuntimeError as e:
                print("TRAIN FAIL (OOM?):", str(e)[:200], flush=True)
                json.dump({"status":"oom","error":str(e)[:200]}, open(OUTD+r"\p3b-qwen15-result.json","w")); return
            if torch.isnan(loss):
                print("NaN loss — DUNG (fp16 khong on, can fp32/LoRA)", flush=True)
                json.dump({"status":"NaN"}, open(OUTD+r"\p3b-qwen15-result.json","w")); return
            loss.backward(); clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); tot+=loss.item()
            if (i+1)%100==0: print(f"ep{ep} {i+1}/{len(tr_loader)} loss {tot/(i+1):.4f}", flush=True)
        print(f"ep{ep} loss {tot/len(tr_loader):.4f} {time.time()-t0:.0f}s", flush=True)
    model.eval()
    te_loader = DataLoader(DS(te), batch_size=BS, shuffle=False, collate_fn=collate)
    preds=[]
    with torch.no_grad():
        for b in te_loader:
            b={k:v.to("cuda") for k,v in b.items()}
            preds.append(torch.sigmoid(model(**b).logits.float()).cpu().numpy())
    P = np.vstack(preds)
    if int(np.isnan(P).sum()):
        print("NaN in probs — bo", flush=True)
        json.dump({"status":"NaN_probs"}, open(OUTD+r"\p3b-qwen15-result.json","w")); return
    per = {}
    for j,k in enumerate(KEEP):
        if Yte[:,j].sum() < 20: continue
        ap = average_precision_score(Yte[:,j], P[:,j])
        per[k] = round(float(ap),4); print(f"1.5B {k}: {ap:.3f}", flush=True)
    res = {"model":MODEL,"split":"LAB->REAL","per_ttp":per,
           "mean_AP": round(float(np.mean(list(per.values()))),4),
           "baseline_05B": 0.3517, "baseline_tfidf": 0.3227, "train_s": round(time.time()-t0)}
    json.dump(res, open(OUTD+r"\p3b-qwen15-result.json","w"), indent=1)
    print("1.5B mean AP:", res["mean_AP"], "(0.5B: 0.3517, TF: 0.3227)", flush=True)

if __name__ == "__main__":
    main()
