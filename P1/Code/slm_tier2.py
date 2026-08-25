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

# SLM prompt template (English, designed for Qwen2.5 instruction format)
SYSTEM_PROMPT = """You are a cybersecurity analyst examining endpoint telemetry for malicious activity.

Given a sequence of system events (process creation, file access, network connections) and the process tree context, classify whether this activity is MALICIOUS or BENIGN.

Consider:
- Command-line arguments: obfuscation, LOLBin usage, encoded scripts
- Process tree: unusual parent-child relationships
- Network destinations: known-good IPs vs suspicious (C2, data exfil)
- File paths: normal locations vs staging paths (Downloads, Temp, AppData)
- Timing: burst patterns, unusual hours

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
    """Format a single enriched alert into the user prompt."""
    parent_chain = "\n".join(
        f"  {p.get('node', '?')} [{p.get('op', '?')}]"
        for p in alert.get("parent_chain", [])
    ) or "  (no ancestors found)"
    
    event_seq = "\n".join(
        f"  {e.get('src', '?')} -> [{e.get('op', '?')}] -> {e.get('dst', '?')}"
        for e in alert.get("event_seq", [])[:20]
    ) or "  (no events)"
    
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


def run_baseline_tfidf(
    alerts: List[Dict],
    gt_nids: set,
    alert_k: int,
) -> Dict:
    """
    Baseline: TF-IDF char n-gram + logistic regression.
    Trains on enriched text (parent_chain + event_seq), predicts on held-out.
    
    This serves as the encoder baseline (H0 elimination test).
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import precision_score, recall_score
    except ImportError:
        print("ERROR: pip install scikit-learn")
        sys.exit(1)
    
    # Prepare text features
    texts = []
    labels = []
    for alert in alerts:
        parts = []
        parts.append(alert.get("self_label", ""))
        for p in alert.get("parent_chain", []):
            parts.append(f"{p.get('op', '')} {p.get('node', '')}")
        for e in alert.get("event_seq", [])[:10]:
            parts.append(f"{e.get('src', '')} {e.get('op', '')} {e.get('dst', '')}")
        texts.append(" ".join(parts))
        labels.append(1 if str(alert.get("nid")) in gt_nids else 0)
    
    # TF-IDF char n-gram (captures obfuscation patterns)
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=50000,
        sublinear_tf=True,
    )
    X = tfidf.fit_transform(texts)
    
    # Logistic regression (fast, interpretable)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, labels)
    
    # Predict scores and rank by confidence
    scores = clf.predict_proba(X)[:, 1]
    
    # Rank by score (descending)
    import numpy as np
    order = np.argsort(-scores)
    top_k_idx = order[:alert_k]
    
    tp_before = int(sum(labels[i] for i in top_k_idx))
    fp_before = alert_k - tp_before
    
    # Threshold: keep only predictions > 0.5
    tp_after = int(sum(1 for i in top_k_idx if scores[i] > 0.5 and labels[i] == 1))
    fp_after = int(sum(1 for i in top_k_idx if scores[i] > 0.5 and labels[i] == 0))
    filtered = int(sum(1 for i in top_k_idx if scores[i] <= 0.5))
    
    precision_before = tp_before / max(alert_k, 1)
    precision_after = tp_after / max(tp_after + fp_after, 1)
    fp_reduction = 1 - (fp_after / max(fp_before, 1))
    
    return {
        "model": "tfidf_logreg",
        "k": alert_k,
        "tp_before": tp_before,
        "fp_before": fp_before,
        "precision_before": precision_before,
        "tp_after_baseline": tp_after,
        "fp_after_baseline": fp_after,
        "precision_after_baseline": precision_after,
        "fp_reduction_baseline": fp_reduction,
        "filtered_baseline": filtered,
    }


def main():
    parser = argparse.ArgumentParser(description="SLM Tier-2 Zero-shot + Encoder Baseline")
    parser.add_argument("--alerts_jsonl", type=str, required=True,
                        help="Path to enriched alerts JSONL")
    parser.add_argument("--gt_csv", type=str, default=None,
                        help="Ground truth CSV (if None, use TP nids from results pth)")
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
    
    print(f"  Alert budget: k={args.alert_k}")
    print(f"  GT malicious: {len(gt_nids)}")
    
    all_results = {}
    
    # --- Baseline: TF-IDF + LogReg ---
    print("\n=== BASELINE: TF-IDF char n-gram + LogReg ===")
    baseline_results = run_baseline_tfidf(alerts, gt_nids, args.alert_k)
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
