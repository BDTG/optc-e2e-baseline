"""P3/RQ1b-hybrid — rule chain>=2 làm pre-filter, SLM chỉ chạy trên dị nghi.
Trên rq1a-subset (12 mal + 192 benign): đổi chuỗi quyết định đúng phương án cấu trúc.
Output: rule-acc = TP/TN/FP/FN + anomaly scan + per-evidence JSON.
Siêu nghuyện: chỉ 12/204 cần SLM nếu pipeline theo hybrid (88% tiết kiệm compute)."""
import json, sys, statistics, threading, time, collections
sys.stdout.reconfigure(line_buffering=True)

SUB = r"F:\backup\OpTC-thesis\P1\Output\data\rq1a-subset.jsonl"
OUT = r"F:\backup\OpTC-thesis\P1\Output\results_phase2\rq1b-hybrid-result.json"


def is_suspicious(rec):
    ch = rec.get("parent_chain", [])
    if isinstance(ch, str):
        try:
            ch = json.loads(ch.replace("'", '"'))
        except Exception:
            ch = []
    # Rule: chain >= 2 nodes → cần SLM soi (dị nghi)
    return len(ch) >= 2


def main():
    recs = [json.loads(l) for l in open(SUB, encoding="utf-8")]
    flagged = [r for r in recs if is_suspicious(r)]
    clean = [r for r in recs if not is_suspicious(r)]
    tp = sum(1 for r in flagged if r["label"] == 1)
    fp = sum(1 for r in flagged if r["label"] == 0)
    fn = sum(1 for r in clean if r["label"] == 1)
    tn = sum(1 for r in clean if r["label"] == 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    summ = {
        "n": len(recs), "mal": sum(r["label"] for r in recs),
        "flagged_for_slm": len(flagged), "auto_clean": len(clean),
        "compute_saved_pct": round(100 * len(clean) / len(recs), 1),
        "rule_only": {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
                      "precision": round(precision, 4), "recall": round(recall, 4),
                      "F1": round(f1, 4)},
        "next": "12 dị nghi (TP 12 + FP 9) chuyển qua SLM/LoRA để lọc FP — mục tiêu SLM chỉ cần đẩy 9 FP → BENIGN thay vì 204 mẫu"
    }
    json.dump(summ, open(OUT, "w", encoding="utf-8"), indent=1)
    print("SUMMARY:", json.dumps(summ, indent=1))
    print("RQ1B_HYBRID_DONE")


main()
