"""
TF-IDF H0 baseline — correct evaluation (CV, not in-sample)
Compares v1 (generic IDs) vs v2 (real msg) on 2250 alerts.

Metrics per Note.md:
- AP, precision@k, recall@k, fp_reduction @ fixed k
- Also recall @{1,10,100} alert/host/day (approx: 3 hosts? H051 single host 3 days test -> k=3,30,300 for 1,10,100 per host per day)
But we report k=500,1000,2000 for compatibility with previous.

Uses StratifiedKFold (5 folds) + temporal rank split as secondary.

Outputs: P1/Output/tfidf_cv_results.json
"""
import json, pickle, re
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

INPUT_V1 = r"D:\OpTC-thesis\P1\Output\alerts_enriched_partial.jsonl"
INPUT_V2 = r"D:\OpTC-thesis\P1\Output\alerts_enriched_v2.jsonl"
GT_PATH = r"D:\OpTC-thesis\P1\Output\gt_and_scores.json"
OUTPUT = r"D:\OpTC-thesis\P1\Output\tfidf_cv_results.json"

def load_alerts(path):
    alerts=[]
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            alerts.append(json.loads(line))
    return alerts

def build_texts(alerts, use_msg=True):
    texts=[]
    for a in alerts:
        parts=[]
        # self_label always now enriched if use_msg and available
        parts.append(a.get("self_label",""))
        for p in a.get("parent_chain",[]):
            # include msg if available
            if use_msg and p.get("msg"):
                parts.append(f"{p.get('op','')} {p.get('msg')}")
            else:
                parts.append(f"{p.get('op','')} {p.get('node','')}")
        for e in a.get("event_seq",[])[:10]:
            if use_msg:
                src = e.get("src_msg") or e.get("src","")
                dst = e.get("dst_msg") or e.get("dst","")
                parts.append(f"{src} {e.get('op','')} {dst}")
            else:
                parts.append(f"{e.get('src','')} {e.get('op','')} {e.get('dst','')}")
        texts.append(" ".join(parts))
    return texts

def evaluate_cv(texts, labels, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aps=[]
    # for ranking within test folds we compute scores per fold then aggregate
    # also compute global scores by out-of-fold predictions
    oof_scores = np.zeros(len(labels))
    for fold, (tr, te) in enumerate(skf.split(texts, labels)):
        tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), max_features=50000, sublinear_tf=True)
        Xtr = tfidf.fit_transform([texts[i] for i in tr])
        Xte = tfidf.transform([texts[i] for i in te])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        clf.fit(Xtr, labels[tr])
        scores = clf.predict_proba(Xte)[:,1]
        oof_scores[te] = scores
        ap = average_precision_score(labels[te], scores) if labels[te].sum()>0 else 0
        aps.append(ap)
        print(f"  Fold {fold}: train pos {labels[tr].sum()} test pos {labels[te].sum()} AP {ap:.4f}")
    # global AP
    global_ap = average_precision_score(labels, oof_scores)
    print(f"  Global OOF AP: {global_ap:.4f} (mean fold AP {np.mean(aps):.4f} +- {np.std(aps):.4f})")
    return oof_scores, global_ap, aps

def evaluate_at_k(labels, scores, k_list):
    # rank by scores descending globally
    order = np.argsort(-scores)
    results={}
    for k in k_list:
        if k > len(labels):
            continue
        top = order[:k]
        tp_before = int(sum(labels[i] for i in top)) # actually this is TP after ranking by TF-IDF, not before. For "before" we need original rank? But H0 is standalone ranker, so "before" = ranking by TF-IDF itself?
        # For fair comparison to SLM tier2, we define:
        # - Baseline ranking is TF-IDF scores
        # - Filtering is threshold 0.5 on TF-IDF scores
        fp_before = k - tp_before
        # threshold filtering within top-k
        tp_after = int(sum(1 for i in top if scores[i] > 0.5 and labels[i]==1))
        fp_after = int(sum(1 for i in top if scores[i] > 0.5 and labels[i]==0))
        prec_before = tp_before / max(k,1)
        prec_after = tp_after / max(tp_after+fp_after,1) if (tp_after+fp_after)>0 else 0
        fp_red = 1 - fp_after / max(fp_before,1) if fp_before>0 else 0
        recall_before = tp_before / max(labels.sum(),1)
        recall_after = tp_after / max(labels.sum(),1)
        results[f"k{k}"] = {
            "tp_ranked": tp_before,
            "fp_ranked": fp_before,
            "prec_ranked": prec_before,
            "tp_after_thresh0.5": tp_after,
            "fp_after_thresh0.5": fp_after,
            "prec_after": prec_after,
            "fp_reduction": fp_red,
            "recall_ranked": recall_before,
            "recall_after": recall_after
        }
        print(f"  k={k}: ranked TP={tp_before} FP={fp_before} prec {prec_before:.4f} -> after thresh TP {tp_after} FP {fp_after} prec {prec_after:.4f} FP_red {fp_red:.4f} recall {recall_before:.3f}->{recall_after:.3f}")
    return results

def evaluate_in_sample(texts, labels):
    # for comparison: old leakage method
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), max_features=50000, sublinear_tf=True)
    X = tfidf.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    clf.fit(X, labels)
    scores = clf.predict_proba(X)[:,1]
    ap = average_precision_score(labels, scores)
    print(f"  In-sample AP (leakage): {ap:.4f}")
    return scores, ap

def main():
    with open(GT_PATH) as f:
        gt_data = json.load(f)
    gt_nids = set(gt_data["gt_nids"])
    print(f"GT nids: {len(gt_nids)}")

    for label, path in [("V1-generic", INPUT_V1), ("V2-msg-enriched", INPUT_V2)]:
        print(f"\n{'='*60}")
        print(f"Evaluating {label}: {path}")
        alerts = load_alerts(path)
        print(f"  alerts: {len(alerts)}")
        labels = np.array([1 if str(a.get("nid")) in gt_nids else 0 for a in alerts])
        print(f"  pos {labels.sum()} neg {(labels==0).sum()} ({labels.sum()/len(labels)*100:.3f}%)")
        texts = build_texts(alerts, use_msg=True)
        # show text length stats
        lens = [len(t) for t in texts]
        print(f"  text len p50 {np.median(lens):.0f} p90 {np.percentile(lens,90):.0f} max {max(lens)}")
        print(f"  sample GT text: {texts[np.where(labels==1)[0][0]][:400] if labels.sum()>0 else 'none'}")
        print(f"  sample benign text: {texts[np.where(labels==0)[0][0]][:400]}")

        print(f"\n  --- In-sample (old method, leakage) ---")
        leakage_scores, leakage_ap = evaluate_in_sample(texts, labels)
        leakage_at_k = evaluate_at_k(labels, leakage_scores, [500,1000,2000])

        print(f"\n  --- CV 5-fold OOF (correct) ---")
        oof_scores, global_ap, fold_aps = evaluate_cv(texts, labels, n_splits=5)
        cv_at_k = evaluate_at_k(labels, oof_scores, [500,1000,2000])

        # Temporal split: sort by rank (which is score order originally, but we already have alerts sorted by rank)
        # For temporal realism, we should split by time: but we don't have timestamp; use rank order as proxy (lower rank = higher original score, not time)
        # So we also do 80/20 stratified split as sanity
        from sklearn.model_selection import train_test_split
        tr_idx, te_idx = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42)
        tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), max_features=50000, sublinear_tf=True)
        Xtr = tfidf.fit_transform([texts[i] for i in tr_idx])
        Xte = tfidf.transform([texts[i] for i in te_idx])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        clf.fit(Xtr, labels[tr_idx])
        te_scores = clf.predict_proba(Xte)[:,1]
        te_labels = labels[te_idx]
        ap_holdout = average_precision_score(te_labels, te_scores)
        print(f"\n  --- Holdout 80/20 stratified ---")
        print(f"  train pos {labels[tr_idx].sum()} test pos {te_labels.sum()} AP {ap_holdout:.4f}")
        # store results
        if label == "V1-generic":
            v1_results = {"leakage_ap": leakage_ap, "global_ap": global_ap, "fold_aps": fold_aps, "leakage_at_k": leakage_at_k, "cv_at_k": cv_at_k, "holdout_ap": ap_holdout}
        else:
            v2_results = {"leakage_ap": leakage_ap, "global_ap": global_ap, "fold_aps": fold_aps, "leakage_at_k": leakage_at_k, "cv_at_k": cv_at_k, "holdout_ap": ap_holdout}
            v2_oof = oof_scores
            v2_labels = labels

    # Save
    output = {
        "v1": v1_results,
        "v2": v2_results,
        "note": "v1 leakage vs cv; v2 enriched with real msg. k evaluation is on TF-IDF ranked list; threshold 0.5 filtering. For tier2 FP reduction, compare cv_at_k."
    }
    with open(OUTPUT, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {OUTPUT}")

    # Also compute simple baseline: how many GT in top-k by OOF scores for v2?
    print("\n--- V2 OOF ranking detail ---")
    order = np.argsort(-v2_oof)
    ranked_labels = v2_labels[order]
    for k in [10,50,100,500,1000,2000]:
        tp = ranked_labels[:k].sum()
        print(f"  top-{k}: TP {tp}/12 recall {tp/12:.2%} prec {tp/k:.3%}")

if __name__ == "__main__":
    main()
