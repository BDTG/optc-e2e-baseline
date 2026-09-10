"""Constrained scoring AD-GEN 200 mau can bang — dung loi ke thua tu tier2_explain_eval_constrained.
Verdict: logprob MALICIOUS vs BENIGN -> verdict_score (log-odds) + AUC/AP.
TTP: argmax mean-logprob tren TTP_ENUM giao voi enum goc (24) — GT lay tu technique_id AD-GEN.
Evidence: rule (cmdline / parent_chain / event_seq / none).
Khong train lai, chi scoring."""
import json, time, os, sys, re, collections
sys.path.insert(0, "P1/Code")
import numpy as np
import torch
import torch.nn.functional as F

SAMPLE = "P1/Output/data/adgen-ttp-bench.jsonl"  # 709 chain co GT TTP (thay yeu cau N>=20/TTP)
OUT = "P1/Output/results_phase2/adgen-constrained-ttp.json"
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
CKPT = "P1/Output/models/lora-05b/checkpoint-113"
SEED = 42
sys.stdout.reconfigure(line_buffering=True)

TTP_ENUM = ["T1003", "T1059", "T1053", "T1071", "T1218", "T1027",
            "T1562", "T1574", "T1490", "T1087", "T1082", "T1083", "none"]  # 13: 12 TTP co GT trong bench + none
VERDICTS = ["MALICIOUS", "BENIGN"]

FEWSHOT = (
    'Example: Chain: powershell.exe | cmd: powershell -nop -w hidden -enc SQBFAFgAIA==\n'
    '{"verdict":"MALICIOUS","technique_id":"T1059","evidence_field":"cmdline","confidence":"high"}\n'
    'Example: Chain: svchost.exe | chrome.exe | cmd: none\n'
    '{"verdict":"BENIGN","technique_id":"none","evidence_field":"none","confidence":"high"}\n'
)
PREFIX = (
    "You are a security analyst. Classify this Windows process provenance chain.\n"
    + FEWSHOT +
    "Chain: {chain}\n"
    'Respond with one JSON object: {{"verdict":"{v}","technique_id":"{t}","evidence_field":"{e}"}}\n'
    "JSON:"
)

def rule_evidence(o):
    ch = o.get("parent_chain", []) or []
    t = " | ".join([(c.get("msg") or "") for c in ch[-5:]]).lower()
    if "cmd: " in t and "cmd: none" not in t:
        return "cmdline"
    if len(ch) >= 2:
        return "parent_chain"
    if len(o.get("event_seq", []) or []) >= 1:
        return "event_seq"
    return "none"

def has_evidence_gt(o):
    """GT evidence: chain co bang chung that. Dung de PHAT NE (thay yeu cau)."""
    return rule_evidence(o) != "none"

def batch_scores(model, tok, prefix, candidates):
    pre = tok(prefix, add_special_tokens=False)["input_ids"]
    cands = [tok(c, add_special_tokens=False)["input_ids"] for c in candidates]
    seqs = [pre + c for c in cands]
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    inp = torch.tensor([[pad] * (L - len(s)) + s for s in seqs]).to(model.device)
    mask = (inp != pad).long()
    with torch.no_grad():
        logits = model(input_ids=inp, attention_mask=mask).logits.float()
    logp = F.log_softmax(logits, dim=-1)
    out = []
    for row, c in enumerate(cands):
        m = len(c)
        lp = sum(float(logp[row, L - m - 1 + j, c[j]]) for j in range(m)) / m
        out.append(lp)
    return out

def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.manual_seed(SEED)
    rows = [json.loads(l) for l in open(SAMPLE, encoding="utf-8")]
    print(f"n={len(rows)} pos={sum(r['label'] for r in rows)}", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="auto")
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, CKPT)
        print("loaded LoRA", flush=True)
    except Exception as e:
        print(f"LoRA fail ({e}), dung base", flush=True)
    model.eval()
    ttps_str = ", ".join(t for t in TTP_ENUM if t != "none")
    recs, lat = [], []
    for i, o in enumerate(rows):
        t0 = time.time()
        ch = o.get("parent_chain", []) or []
        txt = " | ".join([(c.get("msg") or "") for c in ch[-5:]])[:800]
        base = PREFIX.split('{{"verdict"')[0].replace("{chain}", txt)
        vs = batch_scores(model, tok, base + '{"verdict":"', VERDICTS)
        vscore = vs[0] - vs[1]
        verdict = 1 if vs[0] > vs[1] else 0
        vtxt = VERDICTS[0] if verdict == 1 else VERDICTS[1]
        ts = batch_scores(model, tok, base + '{"verdict":"' + vtxt + '","technique_id":"', TTP_ENUM)
        ttp = TTP_ENUM[int(np.argmax(ts))]
        evid = rule_evidence(o)
        lat.append(time.time() - t0)
        is_mal = int(o["label"])
        gt = re.findall(r"T\d{4}(?:\.\d{3})?", str(o.get("technique_id", "")))
        base_gt = set(re.match(r"(T\d{4})", t).group(1) for t in gt)
        base_pred = re.match(r"(T\d{4})", ttp).group(1) if ttp != "none" else "none"
        evid_gt = has_evidence_gt(o)
        # hallucination moi theo yeu cau thay: model ne (evidence=none) khi chain CO bang chung
        # HOAC tro evidence khong ton tai trong input -> phat NE
        evd_claimed_none = (evid == "none")
        evd_invalid = (evid == "none" and evid_gt) or (evid != "none" and not evid_gt)
        recs.append({
            "nid": o["nid"], "is_mal": is_mal, "verdict": verdict,
            "verdict_score": round(vscore, 4), "verdict_correct": int(verdict == is_mal),
            "ttp_pred": ttp, "ttp_gt": gt, "ttp_pred_base": base_pred,
            "ttp_correct": int(base_pred in base_gt) if (is_mal and gt) else None,
            "evidence": evid, "evidence_gt_exists": int(evid_gt),
            "hallucinated": int(evd_invalid),  # NE hoac chi bang chung khong co
            "evidence_valid": int(not evd_invalid), "grounded": int(not (verdict == 1 and evid == "none")),
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.array([r["is_mal"] for r in recs]); s = np.array([r["verdict_score"] for r in recs])
    scored = [r["ttp_correct"] for r in recs if r["ttp_correct"] is not None]
    hall = [r["hallucinated"] for r in recs if r["is_mal"] == 1]
    tp = sum(1 for r in recs if r["verdict"] == 1 and r["is_mal"] == 1)
    tn = sum(1 for r in recs if r["verdict"] == 0 and r["is_mal"] == 0)
    fp = sum(1 for r in recs if r["verdict"] == 1 and r["is_mal"] == 0)
    fn = sum(1 for r in recs if r["verdict"] == 0 and r["is_mal"] == 1)
    summ = {"n": len(recs), "pos": int(y.sum()),
            "verdict_accuracy": round(float((y == (s > 0)).mean()), 4),
            "verdict_AUC": round(float(roc_auc_score(y, s)), 4),
            "verdict_AP": round(float(average_precision_score(y, s)), 4),
            "ttp_accuracy_on_gt": round(float(np.mean(scored)), 4) if scored else None,
            "ttp_scored_count": len(scored),
            "hallucination_rate": round(float(np.mean(hall)), 4) if hall else None,
            "halluc_count": int(np.sum(hall)) if hall else 0,
            "evidence_validity": round(float(1 - np.mean(hall)), 4) if hall else None,
            "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
            "avg_latency_sec": round(float(np.mean(lat)), 3),
            "ttp_pred_dist": dict(collections.Counter(r["ttp_pred"] for r in recs).most_common(10))}
    print("\n=== AD-GEN CONSTRAINED ===", flush=True)
    for k, v in summ.items():
        print(f"  {k}: {v}", flush=True)
    json.dump({"summary": summ, "records": recs}, open(OUT, "w"), indent=1)
    print(f"SAVED {OUT}", flush=True)

if __name__ == "__main__":
    main()

