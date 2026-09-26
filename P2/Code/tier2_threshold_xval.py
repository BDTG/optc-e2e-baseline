"""
So sach cho tier-2 constrained (CPU, sklearn):
 1. Cross-val nguong verdict_score (StratifiedKFold k=4, fit thr train -> do test)
 2. TF-IDF tren CUNG 40 mau, CUNG fold -> so AP/AUC cong bang
"""
import json, numpy as np
import warnings
warnings.filterwarnings("ignore")

ENRICHED = "P1/Output/data/alerts-enriched-v2.jsonl"
GT = "P1/Output/data/gt_and_scores.json"
CONSTR_A = "P1/Output/results_phase2/tier2-explain-constrained-A_optc_s1.json"
OUT = "P1/Output/results_phase2/tier2-threshold-xval.json"
SEED = 42

def elem_text(e):
    if not isinstance(e, dict):
        return str(e)
    if e.get("msg"):
        return str(e["msg"])
    img = e.get("image") or e.get("node") or ""
    cmd = e.get("cmd")
    if cmd:
        return f"{img} | cmd: {cmd}"
    return f"{img} {e.get('op') or ''}".strip()

def build_text(o):
    ch = o.get("parent_chain", None)
    if ch is None:
        ch = o.get("chain", []) or []
    return " | ".join(elem_text(e) for e in ch[-5:])[:800].lower()

def fit_thr(s_tr, y_tr):
    cands = np.unique(np.quantile(s_tr, np.linspace(0, 1, 41)))
    best = max((((s_tr >= t).astype(int) == y_tr).mean(), t) for t in cands)
    return best[1]

def prf(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    acc = (tp + tn) / len(y)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return acc, f1, (tp, fp, fn, tn)

def main():
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    ca = json.load(open(CONSTR_A, encoding="utf-8"))["records"]
    nids = [r["nid"] for r in ca]
    y = np.array([r["is_mal"] for r in ca])
    s = np.array([r["verdict_score"] for r in ca])
    print(f"tap: n={len(y)} pos={int(y.sum())}", flush=True)

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))

    # 1. SLM cross-val nguong
    accs, f1s, aucs = [], [], []
    oof = np.zeros(len(y))
    for tr, te in folds:
        t = fit_thr(s[tr], y[tr])
        p = (s[te] >= t).astype(int)
        oof[te] = p
        a, f, _ = prf(y[te], p)
        accs.append(a); f1s.append(f); aucs.append(roc_auc_score(y[te], s[te]))
    slm = {"acc_mean": round(float(np.mean(accs)), 4), "acc_std": round(float(np.std(accs)), 4),
           "f1_mean": round(float(np.mean(f1s)), 4), "f1_std": round(float(np.std(f1s)), 4),
           "auc_mean": round(float(np.mean(aucs)), 4), "auc_std": round(float(np.std(aucs)), 4),
           "auc_pooled": round(float(roc_auc_score(y, s)), 4),
           "ap_pooled": round(float(average_precision_score(y, s)), 4),
           "oof_acc": round(float((oof == y).mean()), 4)}
    print("SLM xval:", slm, flush=True)

    # 2. TF-IDF cung 40 mau, cung fold
    by_nid = {}
    for l in open(ENRICHED, encoding="utf-8"):
        if l.strip():
            o = json.loads(l)
            by_nid[str(o.get("nid"))] = o
    X = [build_text(by_nid[n]) for n in nids]
    pipe = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=50000),
                         LogisticRegression(class_weight="balanced", max_iter=1000))
    t_acc, t_f1, t_auc, t_ap, oof_p = [], [], [], [], np.zeros(len(y))
    oof_s = np.zeros(len(y))
    for tr, te in folds:
        Xtr = [X[i] for i in tr]; Xte = [X[i] for i in te]
        pipe.fit(Xtr, y[tr])
        prob = pipe.predict_proba(Xte)[:, 1]
        p = (prob >= 0.5).astype(int)
        oof_p[te] = p; oof_s[te] = prob
        a, f, _ = prf(y[te], p)
        t_acc.append(a); t_f1.append(f)
        t_auc.append(roc_auc_score(y[te], prob)); t_ap.append(average_precision_score(y[te], prob))
    tfidf = {"acc_mean": round(float(np.mean(t_acc)), 4), "acc_std": round(float(np.std(t_acc)), 4),
             "f1_mean": round(float(np.mean(t_f1)), 4), "f1_std": round(float(np.std(t_f1)), 4),
             "auc_mean": round(float(np.mean(t_auc)), 4), "auc_std": round(float(np.std(t_auc)), 4),
             "auc_pooled": round(float(roc_auc_score(y, oof_s)), 4),
             "ap_pooled": round(float(average_precision_score(y, oof_s)), 4),
             "oof_acc": round(float((oof_p == y).mean()), 4)}
    print("TFIDF xval:", tfidf, flush=True)

    json.dump({"n": len(y), "pos": int(y.sum()), "k": 4, "seed": SEED,
               "slm_constrained": slm, "tfidf_same40": tfidf,
               "note": "Cung 40 mau S1 (12 pos + 28 neg top-rank), cung fold. "
                       "SLM fit nguong train-fold; TF-IDF char2-5 50k + LR balanced."},
              open(OUT, "w"), indent=2)
    print(f"SAVED {OUT}", flush=True)

if __name__ == "__main__":
    main()
