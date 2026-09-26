"""
SLM Tier-2 Zero-shot Classifier
================================
Phân loại FP/TP từ alerts của ORTHRUS/Velox tầng 1.

Pipeline:
1. Load enriched alerts (JSONL: parent_chain + event_seq per node)
2. Prompt SLM (Qwen2.5-3B/8B int4) với context thực tế
3. Parse verdict: malicious / benign / uncertain
4. Tính FP reduction @ fixed recall

Note.md alignment:
- RQ1a: reduce false positive at fixed recall
- RQ1b: detection gain on holdout techniques
- Confidence scoring for risk-coverage curve
"""
import json
import os
import sys
import time
import argparse
from typing import List, Dict, Optional, Tuple

# SLM prompt template — Kerckhoffs compliant (format-only, no detection logic)
# Per Note.md: prompt chỉ hướng dẫn format, không chứa signature/TTP.
SYSTEM_PROMPT = """You are a cybersecurity analyst. Examine the endpoint telemetry provided and classify the activity.

Respond with exactly one of:
CLASSIFICATION: MALICIOUS
CONFIDENCE: [0.0-1.0]
REASON: [one sentence explaining the decision]
"""

USER_PROMPT_TEMPLATE = """Classify this endpoint activity:

=== Process Identity ===
{self_label}

=== Process Tree (up to 3 ancestors) ===
{parent_chain}

=== Recent Events (chronological) ===
{event_seq}

Is this MALICIOUS or BENIGN?"""


def load_enriched_alerts(jsonl_path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load enriched alerts from JSONL file."""
    alerts = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if limit and len(alerts) >= limit:
                break
            alerts.append(json.loads(line.strip()))
    return alerts


def format_alert_for_prompt(alert: Dict) -> str:
    """Format a single enriched alert into the user prompt. Uses msg if available."""
    def _pc_label(p):
        # prefer msg (real path/cmd/netflow) over generic node id
        label = p.get("msg") or p.get("node", "?")
        return f"  {label} [{p.get('op', '?')}]"
    parent_chain = "\n".join(_pc_label(p) for p in alert.get("parent_chain", [])) or "  (no ancestors found)"
    
    def _ev_label(e):
        src = e.get("src_msg") or e.get("src", "?")
        dst = e.get("dst_msg") or e.get("dst", "?")
        return f"  {src} -> [{e.get('op', '?')}] -> {dst}"
    event_seq = "\n".join(_ev_label(e) for e in alert.get("event_seq", [])[:20]) or "  (no events)"
    
    return USER_PROMPT_TEMPLATE.format(
        self_label=alert.get("self_label", "unknown"),
        parent_chain=parent_chain,
        event_seq=event_seq,
    )


def classify_with_slm(
    alerts: List[Dict],
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    quantization: str = "int4",
    max_tokens: int = 128,
    batch_size: int = 1,
) -> List[Dict]:
    """
    Classify alerts using SLM.
    
    Args:
        alerts: List of enriched alert dicts
        model_name: HuggingFace model ID
        quantization: quantization level (int4/int8/fp16)
        max_tokens: max generation length
        batch_size: concurrent inference (1 for CPU)
    
    Returns:
        List of {rank, nid, score, slm_pred, slm_confidence, slm_reason}
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError:
        print("ERROR: pip install transformers accelerate bitsandbytes torch")
        sys.exit(1)
    
    # Load model with quantization
    if quantization == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantization == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = None
    
    print(f"Loading {model_name} ({quantization})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")
    
    # Classify each alert
    results = []
    for i, alert in enumerate(alerts):
        user_msg = format_alert_for_prompt(alert)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # greedy for reproducibility
                temperature=1.0,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse response
        pred, confidence, reason = parse_slm_response(response)
        
        results.append({
            "rank": alert.get("rank"),
            "nid": alert.get("nid"),
            "score": alert.get("score"),
            "slm_pred": pred,
            "slm_confidence": confidence,
            "slm_reason": reason,
        })
        
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(alerts)} classified ({time.time()-t0:.0f}s)")
    
    return results


def parse_slm_response(response: str) -> Tuple[str, float, str]:
    """Parse CLASSIFICATION/CONFIDENCE/REASON from SLM output."""
    pred = "uncertain"
    confidence = 0.5
    reason = response.strip()[:200]
    
    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith("CLASSIFICATION:"):
            val = line.split(":", 1)[1].strip().upper()
            if "MALICIOUS" in val:
                pred = "malicious"
            elif "BENIGN" in val:
                pred = "benign"
        elif line.upper().startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()[:200]
    
    return pred, confidence, reason


def compute_metrics(
    results: List[Dict],
    gt_nids: set,
    alert_k: int,
) -> Dict:
    """
    Compute FP reduction and precision at fixed alert budget.
    
    Args:
        results: classified results sorted by score descending
        gt_nids: set of ground truth malicious node IDs (str)
        alert_k: top-k alerts to evaluate
    
    Returns:
        Dict with TP, FP, precision, recall, fp_reduction, etc.
    """
    top_k = results[:alert_k]
    
    tp_before = sum(1 for r in top_k if str(r["nid"]) in gt_nids)
    fp_before = alert_k - tp_before
    
    # SLM filter: keep only "malicious" predictions
    tp_after = sum(1 for r in top_k if r["slm_pred"] == "malicious" and str(r["nid"]) in gt_nids)
    fp_after = sum(1 for r in top_k if r["slm_pred"] == "malicious" and str(r["nid"]) not in gt_nids)
    filtered_out = sum(1 for r in top_k if r["slm_pred"] != "malicious")
    
    precision_before = tp_before / max(alert_k, 1)
    precision_after = tp_after / max(tp_after + fp_after, 1)
    fp_reduction = 1 - (fp_after / max(fp_before, 1))
    
    total_gt = len(gt_nids)
    recall_before = tp_before / max(total_gt, 1)
    recall_after = tp_after / max(total_gt, 1)
    
    return {
        "k": alert_k,
        "tp_before": tp_before,
        "fp_before": fp_before,
        "precision_before": precision_before,
        "recall_before": recall_before,
        "tp_after_slm": tp_after,
        "fp_after_slm": fp_after,
        "precision_after_slm": precision_after,
        "recall_after_slm": recall_after,
        "fp_reduction": fp_reduction,
        "filtered_out": filtered_out,
    }


def _build_tfidf_texts(alerts: List[Dict], gt_nids: set):
    """Helper: build texts using msg-enriched fields if available."""
    texts, labels = [], []
    for alert in alerts:
        parts = []
        parts.append(alert.get("self_label", ""))
        for p in alert.get("parent_chain", []):
            # prefer msg over node id (P1/Output/alerts_enriched_v2.jsonl has msg)
            label = p.get("msg") or p.get("node", "")
            parts.append(f"{p.get('op','')} {label}")
        for e in alert.get("event_seq", [])[:10]:
            src = e.get("src_msg") or e.get("src","")
            dst = e.get("dst_msg") or e.get("dst","")
            parts.append(f"{src} {e.get('op','')} {dst}")
        texts.append(" ".join(parts))
        labels.append(1 if str(alert.get("nid")) in gt_nids else 0)
    return texts, labels

def run_baseline_tfidf(
    alerts: List[Dict],
    gt_nids: set,
    alert_k: int,
    cv: bool = False,
) -> Dict:
    """
    Baseline: TF-IDF char n-gram + logistic regression.
    - cv=False (default): leakage-free evaluation not, kept for backward compat but warns.
      For correct H0, use cv=True which does 5-fold OOF.
    - cv=True: Stratified 5-fold OOF, metrics computed on OOF scores ranked globally.
    This serves as the encoder baseline (H0 elimination test). Use cv=True for paper.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import average_precision_score
    except ImportError:
        print("ERROR: pip install scikit-learn")
        sys.exit(1)
    
    texts, labels = _build_tfidf_texts(alerts, gt_nids)
    import numpy as np
    labels_np = np.array(labels)

    if not cv:
        # LEGACY leakage path — warn but keep for compat
        print("WARNING: run_baseline_tfidf(cv=False) is LEAKAGE (fit+predict same data). Use cv=True for H0 paper numbers.")
        tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), max_features=50000, sublinear_tf=True)
        X = tfidf.fit_transform(texts)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        clf.fit(X, labels)
        scores = clf.predict_proba(X)[:, 1]
        ap = average_precision_score(labels_np, scores) if labels_np.sum()>0 else 0
        order = np.argsort(-scores)
        top_k_idx = order[:alert_k]
        tp_before = int(sum(labels[i] for i in top_k_idx))
        fp_before = alert_k - tp_before
        tp_after = int(sum(1 for i in top_k_idx if scores[i] > 0.5 and labels[i] == 1))
        fp_after = int(sum(1 for i in top_k_idx if scores[i] > 0.5 and labels[i] == 0))
        filtered = int(sum(1 for i in top_k_idx if scores[i] <= 0.5))
        precision_before = tp_before / max(alert_k, 1)
        precision_after = tp_after / max(tp_after + fp_after, 1)
        fp_reduction = 1 - (fp_after / max(fp_before, 1))
        return {
            "model": "tfidf_logreg_leakage",
            "k": alert_k, "ap_leakage": ap,
            "tp_before": tp_before, "fp_before": fp_before, "precision_before": precision_before,
            "tp_after_baseline": tp_after, "fp_after_baseline": fp_after, "precision_after_baseline": precision_after,
            "fp_reduction_baseline": fp_reduction, "filtered_baseline": filtered,
        }
    else:
        # CORRECT: 5-fold OOF
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(labels_np))
        fold_aps=[]
        for tr, te in skf.split(texts, labels_np):
            tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), max_features=50000, sublinear_tf=True)
            Xtr = tfidf.fit_transform([texts[i] for i in tr])
            Xte = tfidf.transform([texts[i] for i in te])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
            clf.fit(Xtr, labels_np[tr])
            oof[te] = clf.predict_proba(Xte)[:, 1]
            ap_fold = average_precision_score(labels_np[te], oof[te]) if labels_np[te].sum()>0 else 0
            fold_aps.append(ap_fold)
        ap_oof = average_precision_score(labels_np, oof) if labels_np.sum()>0 else 0
        order = np.argsort(-oof)
        top_k_idx = order[:alert_k]
        tp_before = int(sum(labels[i] for i in top_k_idx))
        fp_before = alert_k - tp_before
        tp_after = int(sum(1 for i in top_k_idx if oof[i] > 0.5 and labels[i] == 1))
        fp_after = int(sum(1 for i in top_k_idx if oof[i] > 0.5 and labels[i] == 0))
        filtered = int(sum(1 for i in top_k_idx if oof[i] <= 0.5))
        precision_before = tp_before / max(alert_k, 1)
        precision_after = tp_after / max(tp_after + fp_after, 1) if (tp_after+fp_after)>0 else 0
        fp_reduction = 1 - (fp_after / max(fp_before, 1)) if fp_before>0 else 0
        recall_before = tp_before / max(labels_np.sum(),1)
        recall_after = tp_after / max(labels_np.sum(),1)
        return {
            "model": "tfidf_logreg_cv5_oof",
            "k": alert_k, "ap_oof": ap_oof, "fold_aps": fold_aps,
            "tp_before": tp_before, "fp_before": fp_before, "precision_before": precision_before,
            "recall_before": recall_before,
            "tp_after_baseline": tp_after, "fp_after_baseline": fp_after,
            "precision_after_baseline": precision_after, "recall_after": recall_after,
            "fp_reduction_baseline": fp_reduction, "filtered_baseline": filtered,
        }


def main():
    parser = argparse.ArgumentParser(description="SLM Tier-2 Zero-shot + Encoder Baseline")
    parser.add_argument("--alerts_jsonl", type=str, required=True,
                        help="Path to enriched alerts JSONL")
    parser.add_argument("--gt_csv", type=str, default=None,
                        help="Ground truth CSV (if None, use TP nids from results pth)")
    parser.add_argument("--gt_json", type=str, default=None,
                        help="Ground truth JSON (gt_and_scores.json with gt_nids)")
    parser.add_argument("--orthrus_results_pth", type=str, default=None,
                        help="ORTHRUS/Velox result pth for GT + score ranking")
    parser.add_argument("--alert_k", type=int, default=10000,
                        help="Top-k alert budget")
    parser.add_argument("--slm_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--quantization", type=str, default="int4",
                        choices=["int4", "int8", "fp16"])
    parser.add_argument("--skip_slm", action="store_true",
                        help="Skip SLM inference (run baseline only)")
    parser.add_argument("--output", type=str, default="slm_tier2_results.json")
    parser.add_argument("--cv", action="store_true",
                        help="Use 5-fold OOF for TF-IDF (correct H0, not leakage)")
    args = parser.parse_args()
    
    # Load enriched alerts
    print(f"Loading alerts from {args.alerts_jsonl}...")
    alerts = load_enriched_alerts(args.alerts_jsonl)
    print(f"  Loaded {len(alerts)} alerts")
    
    # Build GT set
    gt_nids = set()
    if args.orthrus_results_pth:
        import torch
        results = torch.load(args.orthrus_results_pth, weights_only=False)
        gt_nids = {str(nid) for nid, r in results.items() if r.get("y_true", 0) == 1}
        # Sort alerts by original score (descending)
        score_map = {str(r.get("nid")): r.get("score", 0) for r in results if "nid" in r}
        alerts.sort(key=lambda a: score_map.get(str(a.get("nid", "")), 0), reverse=True)
        print(f"  GT nodes: {len(gt_nids)}, alerts sorted by score")
    elif args.gt_json:
        import json as _js
        with open(args.gt_json) as _f:
            _d = _js.load(_f)
        gt_nids = set(str(x) for x in _d.get("gt_nids", []))
        print(f"  GT nodes from gt_json: {len(gt_nids)}")
    elif args.gt_csv:
        import csv as _csv
        with open(args.gt_csv) as _f:
            _r = _csv.DictReader(_f)
            for row in _r:
                # try common column names
                for col in ["node_id","nid","id","object_id"]:
                    if col in row and row[col]:
                        gt_nids.add(str(row[col]).strip())
                        break
        print(f"  GT nodes from gt_csv: {len(gt_nids)}")
    else:
        # try default location
        default_gt = r"D:\OpTC-thesis\P1\Output\gt_and_scores.json"
        if os.path.exists(default_gt):
            import json as _js
            with open(default_gt) as _f:
                _d = _js.load(_f)
            gt_nids = set(str(x) for x in _d.get("gt_nids", []))
            print(f"  GT nodes from default gt_and_scores.json: {len(gt_nids)}")
    
    print(f"  Alert budget: k={args.alert_k}")
    print(f"  GT malicious: {len(gt_nids)}")
    
    all_results = {}
    
    # --- Baseline: TF-IDF + LogReg ---
    print("\n=== BASELINE: TF-IDF char n-gram + LogReg ===")
    if len(gt_nids)==0:
        print("  WARNING: GT empty, cannot compute TF-IDF baseline (single class). Skipping.")
        all_results["baseline"] = {"error": "empty GT"}
    else:
        baseline_results = run_baseline_tfidf(alerts, gt_nids, args.alert_k, cv=args.cv)
        all_results["baseline"] = baseline_results
        for k, v in baseline_results.items():
            print(f"  {k}: {v}")
    
    # --- SLM Zero-shot ---
    if not args.skip_slm:
        print(f"\n=== SLM ZERO-SHOT: {args.slm_model} ({args.quantization}) ===")
        slm_results = classify_with_slm(
            alerts[:args.alert_k],
            model_name=args.slm_model,
            quantization=args.quantization,
        )
        
        # Save raw SLM outputs
        slm_raw_path = args.output.replace(".json", "_slm_raw.jsonl")
        with open(slm_raw_path, "w") as f:
            for r in slm_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved {len(slm_results)} SLM results to {slm_raw_path}")
        
        # Compute metrics at various k values
        print("\n--- SLM FP reduction at various k ---")
        for k_val in [1000, 2000, 5000, 10000]:
            if k_val > len(slm_results):
                continue
            metrics = compute_metrics(slm_results, gt_nids, k_val)
            all_results[f"slm_k{k_val}"] = metrics
            print(f"  k={k_val}: TP={metrics['tp_before']}→{metrics['tp_after_slm']}, "
                  f"FP={metrics['fp_before']}→{metrics['fp_after_slm']}, "
                  f"precision={metrics['precision_before']:.4f}→{metrics['precision_after_slm']:.4f}, "
                  f"fp_reduction={metrics['fp_reduction']:.4f}")
    
    # Save all results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
    main()
