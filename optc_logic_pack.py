"""
OpTC ByT5/TF-IDF logic pack — 1 file gom toàn bộ logic tuần qua (26/08-02/09)
- Fix parent_chain[-5:] (không phải msg rỗng)
- S1 474 (pos9) là gốc, Combined 4142 chỉ để test SD-easy
- Chạy: python optc_logic_pack.py  (cần transformers/datasets/sklearn)
Gửi thầy / copy Desktop — 02/09/2026
"""
import json, random, re, base64, os
os.environ["WANDB_DISABLED"]="true"
os.environ["WANDB_MODE"]="disabled"

# ====== 1. Chuẩn hoá (dùng chung) ======
def normalize_text(s):
    s = s.lower()
    m = re.search(r'-enc\s+([A-Za-z0-9+/=]{20,})', s)
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

def build_text_alert(a):
    chain=a.get("parent_chain",[]) or []
    if chain:
        raw=" | ".join([(c.get("msg") or "") for c in chain[-5:]])[:800]
        if raw.strip(): return normalize_text(raw)
    return normalize_text(a.get("msg","") or "")

def build_text_sd(s):
    if "chain" in s:
        ch=s["chain"]
        if isinstance(ch, list): return " | ".join([normalize_text(c.get("msg","") or "") for c in ch[-3:]])
        else: return normalize_text(str(ch))
    return normalize_text(s.get("msg","") or "")

def build_text_evtx(e):
    if e.get("parent_chain"):
        raw=" | ".join([(c.get("cmd") or c.get("msg") or c.get("image") or "") for c in e["parent_chain"][-3:]])
        if raw.strip(): return normalize_text(raw)
    if e.get("self_label"): return normalize_text(e["self_label"])
    return normalize_text(e.get("msg","") or str(e)[:800])

# ====== 2. Load S1 (OpTC) ======
def load_s1(path_alerts="P1/Output/data/alerts-enriched-v2.jsonl", path_gt="P1/Output/data/gt_and_scores.json"):
    gt=set(json.load(open(path_gt))["gt_nids"])
    alerts=[json.loads(l) for l in open(path_alerts,encoding='utf-8') if l.strip()]
    def has_cmd2(a):
        for c in a.get("parent_chain",[]) or []:
            m=c.get("msg") or ""
            if "| cmd:" in m:
                v=m.split("| cmd:",1)[1].strip()
                if v and v.lower()!="none": return True
        return False
    def has_chain2(a): return len(a.get("parent_chain",[]) or [])>=2
    def has_events2(a): return len(a.get("event_seq",[]) or [])>=3
    s1=[a for a in alerts if has_cmd2(a) and has_chain2(a) and has_events2(a)]
    print(f"S1 filtered {len(s1)} pos {sum(1 for a in s1 if str(a.get('nid')) in gt)}")
    return s1, gt, alerts

def build_combined_4142(s1, gt, alerts, evtx_path="P1/Output/data/evtx-chains.jsonl", sd_path="P1/Output/data/sd-chains.jsonl"):
    import random as R
    evtx=[json.loads(l) for l in open(evtx_path,encoding='utf-8') if l.strip()]
    sd=[json.loads(l) for l in open(sd_path,encoding='utf-8') if l.strip()]
    R.seed(42); R.shuffle(sd); sd_sample=sd[:2000]
    print(f"S1 {len(s1)} EVTX {len(evtx)} SD {len(sd)} sample {len(sd_sample)}")
    combined=[]
    for a in s1:
        nid=str(a.get("nid") or "")
        label=1 if nid in gt else 0
        combined.append((build_text_alert(a), label))
    for e in evtx: combined.append((build_text_evtx(e), 1))
    for s in sd_sample: combined.append((build_text_sd(s), 1))
    benign=[a for a in alerts if str(a.get("nid")) not in gt and a not in s1]
    R.shuffle(benign); benign_sample=benign[:1200]
    for b in benign_sample: combined.append((build_text_alert(b), 0))
    print(f"Combined {len(combined)} pos {sum(1 for _,l in combined if l==1)} empty {sum(1 for t,_ in combined if not t.strip())}")
    R.shuffle(combined)
    return combined

# ====== 3. TF-IDF baseline (S1 hoặc Combined) ======
def run_tfidf(texts, labels, test_size=0.2):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedShuffleSplit
    sss=StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    tr, te = next(sss.split(texts, labels))
    Xtr=[texts[i] for i in tr]; ytr=[labels[i] for i in tr]
    Xte=[texts[i] for i in te]; yte=[labels[i] for i in te]
    vec=TfidfVectorizer(analyzer='char', ngram_range=(2,5), max_features=50000)
    Xtrv=vec.fit_transform(Xtr); Xtev=vec.transform(Xte)
    clf=LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(Xtrv, ytr)
    probs=clf.predict_proba(Xtev)[:,1]
    ap=average_precision_score(yte, probs); auc=roc_auc_score(yte, probs)
    print(f"TF-IDF AP={ap:.4f} AUC={auc:.4f} test {len(yte)} pos {sum(yte)}")
    return ap, auc

# ====== 4. ByT5 fine-tune (cần GPU) ======
def run_byt5(train_texts, train_labels, test_texts, test_labels, out="P1/Output/models/byt5-pack", epochs=5, lr=6e-6):
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    tok=AutoTokenizer.from_pretrained("google/byt5-small")
    def tok_fn(b): return tok(b["text"], truncation=True, max_length=512)
    # oversample pos 2x cho train
    import random as R
    pos=[(t,l) for t,l in zip(train_texts,train_labels) if l==1]
    neg=[(t,l) for t,l in zip(train_texts,train_labels) if l==0]
    train_bal = neg + pos*2; R.shuffle(train_bal)
    tr_t, tr_l = zip(*train_bal) if train_bal else ([],[])
    train_ds=Dataset.from_list([{"text":t,"label":l} for t,l in zip(tr_t,tr_l)]).map(tok_fn, batched=True)
    test_ds=Dataset.from_list([{"text":t,"label":l} for t,l in zip(test_texts,test_labels)]).map(tok_fn, batched=True)
    train_ds.set_format("torch", columns=["input_ids","attention_mask","label"])
    test_ds.set_format("torch", columns=["input_ids","attention_mask","label"])
    model=AutoModelForSequenceClassification.from_pretrained("google/byt5-small", num_labels=2)
    def metrics(p):
        logits=np.array(p.predictions); 
        if logits.ndim==3: logits=logits[:,0,:]
        e=np.exp(logits - np.max(logits,axis=1,keepdims=True)); probs=e[:,1]/e.sum(axis=1)
        y=np.array(p.label_ids).astype(int).ravel()
        try: ap=average_precision_score(y, probs)
        except: ap=0
        try: auc=roc_auc_score(y, probs)
        except: auc=0.5
        return {"ap":ap,"auc":auc}
    args=TrainingArguments(output_dir=out, num_train_epochs=epochs, per_device_train_batch_size=4, per_device_eval_batch_size=8, eval_strategy="epoch", save_strategy="epoch", logging_steps=20, learning_rate=lr, bf16=True, seed=42, report_to="none", load_best_model_at_end=True, metric_for_best_model="ap")
    trainer=Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds, compute_metrics=metrics, processing_class=tok)
    trainer.train(); res=trainer.evaluate()
    print(f"ByT5 AP={res['eval_ap']:.4f} AUC={res['eval_auc']:.4f}")
    return res

# ====== 5. Demo main ======
if __name__=="__main__":
    print("=== OpTC logic pack demo ===")
    s1, gt, alerts = load_s1()
    # S1-only TF-IDF
    s1_texts=[build_text_alert(a) for a in s1]
    s1_labels=[1 if str(a.get("nid")) in gt else 0 for a in s1]
    print("\n--- TF-IDF S1-only ---")
    run_tfidf(s1_texts, s1_labels)
    # Combined 4142 TF-IDF (SD-easy, chỉ tham khảo)
    combined=build_combined_4142(s1, gt, alerts)
    c_texts=[t for t,_ in combined]; c_labels=[l for _,l in combined]
    print("\n--- TF-IDF Combined 4142 (SD-easy, không tính S1) ---")
    run_tfidf(c_texts, c_labels)
    print("\nDone. Để FT ByT5, gọi run_byt5(...) trên 5060Ti.")
