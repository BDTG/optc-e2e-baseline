"""A2 — Domain-adapt temporal REAL: train = LAB + REAL-som 70%, test = REAL-muon 30%.
Khong-leak: split theo thu tu file (proxy thoi gian), khong random.
So sanh: TF-IDF head cung split. FP32 + clip + NaN guard (bai hoc A1).
"""
import json, re, sys, os, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
sys.stdout.reconfigure(line_buffering=True)

BASE = r"C:\Users\BDTG\Desktop\Backup\OpTC-thesis"
DATA = BASE + r"\P1\Output\data\adgen-chains.jsonl"
OUTD = BASE + r"\P1\Output\results_phase2"
KEEP = ["T1059","T1547","T1055","T1543","T1003","T1562","T1490","T1553","T1036","T1070","T1218","T1071"]
EPOCHS = int(os.environ.get("EPOCHS", "2")); BS = 16; LR = 2e-4; MAXLEN = 512

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
    lab = [r for r in rows if r["src"]=="LAB"]
    real = [r for r in rows if r["src"]=="REAL"]
    cut = int(len(real)*0.7)
    tr = lab + real[:cut]; te = real[cut:]
    print(f"train LAB {len(lab)} + REAL-som {len(real[:cut])} = {len(tr)} / test REAL-muon {len(te)}", flush=True)
    Ytr = np.array([r["labels"] for r in tr]); Yte = np.array([r["labels"] for r in te])
    for j,k in enumerate(KEEP):
        print(f"  test {k}: pos {int(Yte[:,j].sum())}", flush=True)

    # TF-IDF head
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), max_features=100_000, min_df=2)
    Xtr = vec.fit_transform([r["text"] for r in tr]); Xte = vec.transform([r["text"] for r in te])
    tf = {}
    for j,k in enumerate(KEEP):
        if Ytr[:,j].sum() < 20 or Yte[:,j].sum() < 5: continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        clf.fit(Xtr, Ytr[:,j])
        ap = average_precision_score(Yte[:,j], clf.predict_proba(Xte)[:,1])
        tf[k] = round(float(ap),4); print(f"TFIDF {k}: {ap:.3f}", flush=True)

    # SLM head fp32
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    torch.manual_seed(42)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(KEEP), problem_type="multi_label_classification",
        torch_dtype=torch.float32).to("cuda")
    model.config.pad_token_id = tok.pad_token_id
    model.gradient_checkpointing_enable()
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
            loss = loss_fn(model(**b).logits.float(), b["labels"])
            if torch.isnan(loss):
                print("NaN loss — DUNG", flush=True)
                json.dump({"status":"NaN"}, open(OUTD+r"\p3b-domainadapt-result.json","w")); return
            loss.backward(); clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); tot+=loss.item()
        print(f"ep{ep} loss {tot/len(tr_loader):.4f} {time.time()-t0:.0f}s", flush=True)
    model.eval()
    te_loader = DataLoader(DS(te), batch_size=BS, shuffle=False, collate_fn=collate)
    preds=[]
    with torch.no_grad():
        for b in te_loader:
            b={k:v.to("cuda") for k,v in b.items()}
            preds.append(torch.sigmoid(model(**b).logits.float()).cpu().numpy())
    P = np.vstack(preds)
    slm = {}
    for j,k in enumerate(KEEP):
        if Yte[:,j].sum() < 5: continue
        ap = average_precision_score(Yte[:,j], P[:,j])
        slm[k] = round(float(ap),4); print(f"SLM {k}: {ap:.3f}", flush=True)
    common = sorted(set(tf) & set(slm))
    res = {"split":"LAB+REALsom70 -> REALmuon30", "n_train":len(tr), "n_test":len(te),
           "tfidf": tf, "slm": slm,
           "tfidf_mean": round(float(np.mean([tf[k] for k in common])),4),
           "slm_mean": round(float(np.mean([slm[k] for k in common])),4),
           "train_s": round(time.time()-t0)}
    json.dump(res, open(OUTD+r"\p3b-domainadapt-result.json","w"), indent=1)
    print("TF mean:", res["tfidf_mean"], "| SLM mean:", res["slm_mean"], flush=True)
    print("SAVED p3b-domainadapt-result.json", flush=True)

if __name__ == "__main__":
    main()
