"""
Review logic: Graph Structural (không đọc text) — Approach 4/2
- Dataset: S1 474 (pos 9) duy nhất — không nhồi SD
- Features: cấu trúc provenance (chain len, event len, has_cmd, cmd_len, unique proc, rarity) — KHÔNG dùng TF-IDF/byte text
- Model: LogisticRegression structural + IsolationForest unsupervised (tùy chọn) — 1 file, chạy CPU là xong
- Để tự review: python P1/Code/review_graph_structural.py
"""
import json, re, random, collections
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit

DATA="P1/Output/data/alerts-enriched-v2.jsonl"
GT="P1/Output/data/gt_and_scores.json"

def load_s1():
    gt=set(json.load(open(GT))["gt_nids"])
    alerts=[json.loads(l) for l in open(DATA,encoding='utf-8') if l.strip()]
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
    print(f"S1 {len(s1)} pos {sum(1 for a in s1 if str(a.get('nid')) in gt)}")
    return s1, gt

# ---- feature extraction (KHÔNG dùng text) ----
def extract_structural(s1, gt):
    # tính tần suất image để làm rarity
    all_imgs=[]
    for a in s1:
        for c in a.get("parent_chain",[]) or []:
            msg=c.get("msg") or ""
            # msg dạng "subject /path/proc.exe | cmd: ..."
            m=re.search(r'subject ([^\s|]+)', msg)
            if m: all_imgs.append(m.group(1).lower().split('/')[-1])
    freq=collections.Counter(all_imgs)
    total=len(all_imgs) or 1
    # feature cho từng alert
    X=[]; y=[]
    for a in s1:
        chain=a.get("parent_chain",[]) or []
        ev=a.get("event_seq",[]) or []
        # 1. chain len
        f_chain=len(chain)
        # 2. event len
        f_event=len(ev)
        # 3. has_cmd already 1 nhưng vẫn tính
        f_has_cmd=1  # S1 filtered nên luôn 1
        # 4. cmd len (tổng)
        cmd_len=sum(len((c.get("msg") or "").split("| cmd:",1)[-1]) for c in chain if "| cmd:" in (c.get("msg") or ""))
        f_cmd_len=np.log1p(cmd_len)
        # 5. unique proc
        procs=set()
        for c in chain:
            msg=c.get("msg") or ""
            m=re.search(r'subject ([^\s|]+)', msg)
            if m: procs.add(m.group(1).lower().split('/')[-1])
        f_unique=len(procs)
        # 6. rarity: trung bình -log(freq)
        rarity=[]
        for c in chain:
            msg=c.get("msg") or ""
            m=re.search(r'subject ([^\s|]+)', msg)
            if m:
                img=m.group(1).lower().split('/')[-1]
                rarity.append(-np.log((freq[img]+1)/total))
        f_rarity=np.mean(rarity) if rarity else 0
        # 7. branching: số FILE/CREATE trong event_seq
        f_file=sum(1 for e in ev if "FILE" in str(e.get("op") or e.get("type") or ""))
        f_flow=sum(1 for e in ev if "FLOW" in str(e.get("op") or ""))
        # 8. depth: vị trí của pos trong chain (leaf)
        f_leaf_rarity=f_rarity  # reuse

        feats=[f_chain, f_event, f_cmd_len, f_unique, f_rarity, f_file, f_flow]
        X.append(feats)
        y.append(1 if str(a.get("nid")) in gt else 0)
    X=np.array(X, dtype=float)
    y=np.array(y)
    names=["chain_len","event_len","log_cmd_len","unique_proc","rarity","file_cnt","flow_cnt"]
    return X, y, names, freq

def main():
    s1, gt = load_s1()
    X, y, names, freq = extract_structural(s1, gt)
    print(f"Features {names}")
    print(f"X shape {X.shape} pos {y.sum()} prev {y.mean():.3f}")
    print(f"Example X[0] {X[0]} y {y[0]}")
    # 80/20 stratified
    sss=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(sss.split(X, y))
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]
    print(f"train {len(ytr)} pos {ytr.sum()} | test {len(yte)} pos {yte.sum()}")

    # --- Model 1: Logistic structural (supervised, giống GNN nhẹ) ---
    print("\n=== Model 1: Logistic structural (Approach 4 thu gọn) ===")
    clf=LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(Xtr, ytr)
    probs=clf.predict_proba(Xte)[:,1]
    ap=average_precision_score(yte, probs); auc=roc_auc_score(yte, probs)
    print(f"AP={ap:.4f} AUC={auc:.4f}")
    print("coef:", dict(zip(names, clf.coef_[0].round(3))))
    # so với TF-IDF 0.89 S1
    print(f"So TF-IDF 0.89 → structural {ap:.4f} {'thua' if ap<0.89 else 'thắng'}")

    # --- Model 2: IsolationForest (unsupervised, Approach 2) ---
    print("\n=== Model 2: IsolationForest (Anomaly, chỉ học benign) ===")
    benign_mask = ytr==0
    iso=IsolationForest(contamination=0.02, random_state=42)
    iso.fit(Xtr[benign_mask])
    # decision: càng âm càng bất thường → -score
    scores=-iso.score_samples(Xte)
    ap2=average_precision_score(yte, scores); auc2=roc_auc_score(yte, scores)
    print(f"AP={ap2:.4f} AUC={auc2:.4f} (học chỉ benign {benign_mask.sum()})")

    # lưu
    import json as js, os
    os.makedirs("P1/Output/results_phase2", exist_ok=True)
    js.dump({"structural_logistic":{"ap":ap,"auc":auc,"coef":dict(zip(names, clf.coef_[0].tolist()))},
             "isolation_forest":{"ap":ap2,"auc":auc2},
             "n":len(y), "pos":int(y.sum()), "names":names}, open("P1/Output/results_phase2/graph-structural-result.json","w"), indent=2)
    print("\nSAVED P1/Output/results_phase2/graph-structural-result.json")
    print("Gợi ý: logistic structural là GNN thu gọn — nếu AP ~0.3-0.5 thì cấu trúc có tín hiệu, nhưng thua text 0.89 → phải kết hợp (Approach 5 neuro-symbolic).")

if __name__=="__main__":
    main()
