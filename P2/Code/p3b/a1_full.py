"""A1 full — Ensemble SLM head (fp32) + TF-IDF head, split LAB->REAL, khong-leak.
Fix NaN: model fp32 (0.5B=2GB, vua 16GB), clip grad 1.0, NaN loss guard.
Tu chua: TF-IDF probs + Ytest tu data; FT SLM 2ep LAB; ensemble grid weights.
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
    tr = [r for r in rows if r["src"]=="LAB"]; te = [r for r in rows if r["src"]=="REAL"]
    print(f"LAB {len(tr)} / REAL {len(te)}", flush=True)
    Ytr = np.array([r["labels"] for r in tr]); Yte = np.array([r["labels"] for r in te])

    # === TF-IDF head probs ===
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), max_features=100_000, min_df=2)
    Xtr = vec.fit_transform([r["text"] for r in tr]); Xte = vec.transform([r["text"] for r in te])
    P_tf = np.zeros((len(te), len(KEEP)))
    for j, k in enumerate(KEEP):
        if Ytr[:,j].sum() < 20: continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        clf.fit(Xtr, Ytr[:,j])
        P_tf[:,j] = clf.predict_proba(Xte)[:,1]
    np.save(OUTD + r"\p3b-tfidf-probs.npy", P_tf)
    tf_aps = [average_precision_score(Yte[:,j], P_tf[:,j]) for j in range(len(KEEP)) if Yte[:,j].sum()>=20]
    print("TF-IDF mean AP:", round(float(np.mean(tf_aps)),4), flush=True)

    # === SLM head fp32 ===
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
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
    t0=time.time(); bad=False
    for ep in range(EPOCHS):
        model.train(); tot=0.0
        for i,b in enumerate(tr_loader):
            b={k:v.to("cuda") for k,v in b.items()}
            loss = loss_fn(model(**b).logits.float(), b["labels"])
            if torch.isnan(loss):
                print(f"NaN loss ep{ep} step{i} — DUNG, bao loi", flush=True); bad=True; break
            loss.backward(); clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); tot+=loss.item()
        print(f"ep{ep} loss {tot/len(tr_loader):.4f} {time.time()-t0:.0f}s", flush=True)
        if bad: break
    model.eval()
    te_loader = DataLoader(DS(te), batch_size=BS, shuffle=False, collate_fn=collate)
    preds=[]
    with torch.no_grad():
        for b in te_loader:
            b={k:v.to("cuda") for k,v in b.items()}
            p=torch.sigmoid(model(**b).logits.float()).cpu().numpy()
            preds.append(p)
    P_slm = np.vstack(preds)
    nnan = int(np.isnan(P_slm).sum())
    print("SLM probs NaN count:", nnan, "/", P_slm.size, flush=True)
    if nnan:
        print("CON NaN — bo qua ensemble, luu debug", flush=True)
        np.save(OUTD + r"\p3b-slm-probs-NAN.npy", P_slm)
        json.dump({"status":"NaN","nan_count":nnan}, open(OUTD+r"\p3b-ensemble-result.json","w"))
        return
    np.save(OUTD + r"\p3b-slm-probs.npy", P_slm)
    s_aps = [average_precision_score(Yte[:,j], P_slm[:,j]) for j in range(len(KEEP)) if Yte[:,j].sum()>=20]
    print("SLM mean AP:", round(float(np.mean(s_aps)),4), flush=True)

    # === Ensemble grid ===
    res = {"tfidf_mean_AP": round(float(np.mean(tf_aps)),4),
           "slm_mean_AP": round(float(np.mean(s_aps)),4), "grid": []}
    for w in [0.3, 0.5, 0.7]:
        P = w*P_slm + (1-w)*P_tf
        pas = [average_precision_score(Yte[:,j], P[:,j]) for j in range(len(KEEP)) if Yte[:,j].sum()>=20]
        res["grid"].append({"w_slm":w, "mean_AP": round(float(np.mean(pas)),4)})
        print(f"w_slm={w}: ensemble mean AP {np.mean(pas):.4f}", flush=True)
    json.dump(res, open(OUTD+r"\p3b-ensemble-result.json","w"), indent=1)
    print("SAVED p3b-ensemble-result.json", flush=True)

if __name__ == "__main__":
    main()
