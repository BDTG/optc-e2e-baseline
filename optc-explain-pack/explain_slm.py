"""Constrained scoring AD-GEN bench (704 chain co GT TTP) — gop explain_05b/3shot_05b/11b/15b/teacher.
Verdict: logprob MALICIOUS vs BENIGN -> verdict_score (log-odds) + AUC/AP.
TTP: argmax mean-logprob tren TTP_ENUM. Evidence: rule (cmdline / parent_chain / event_seq / none).
Khong train lai, chi scoring.

  python explain_slm.py --model Qwen/Qwen2.5-0.5B-Instruct --out output/raw-explain-05b-rerun.json
  python explain_slm.py --model Qwen/Qwen2.5-0.5B-Instruct --shots 3 --out output/raw-explain-3shot-05b-rerun.json
  python explain_slm.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out output/raw-explain-11b-rerun.json
  python explain_slm.py --model Qwen/Qwen2.5-1.5B-Instruct --out output/raw-explain-15b-rerun.json
  python explain_slm.py --model Qwen/Qwen2.5-7B-Instruct --dtype bf16 --resume --out output/raw-explain-teacher-rerun.json
"""
import argparse
import collections
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

TTP_ENUM = ["T1003", "T1059", "T1053", "T1071", "T1218", "T1027",
            "T1562", "T1574", "T1490", "T1087", "T1082", "T1083", "none"]  # 12 TTP co GT trong bench + none
VERDICTS = ["MALICIOUS", "BENIGN"]
EXAMPLES = [
    ('Example: Chain: powershell.exe | cmd: powershell -nop -w hidden -enc SQBFAFgAIA==\n'
     '{"verdict":"MALICIOUS","technique_id":"T1059","evidence_field":"cmdline","confidence":"high"}\n'),
    ('Example: Chain: rundll32.exe | cmd: rundll32 comsvcs.dll MiniDump 624 dump.bin full\n'
     '{"verdict":"MALICIOUS","technique_id":"T1003","evidence_field":"cmdline","confidence":"high"}\n'),
    ('Example: Chain: svchost.exe | chrome.exe | cmd: none\n'
     '{"verdict":"BENIGN","technique_id":"none","evidence_field":"none","confidence":"high"}\n'),
]


def prompt_head(shots):
    """Phan prompt truoc '{"verdict"'; 2 shot = powershell + svchost, 3 shot them comsvcs MiniDump."""
    ex = EXAMPLES if shots == 3 else [EXAMPLES[0], EXAMPLES[2]]
    return "You are a security analyst. Classify this Windows process provenance chain.\n" + "".join(ex) + \
        "Chain: {chain}\nRespond with one JSON object: "


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


def batch_scores(model, tok, prefix, candidates):
    """Mean logprob tung ung vien; chia cum nho vi logits [batch, L, vocab] rat lon."""
    pre = tok(prefix, add_special_tokens=False)["input_ids"]
    cands = [tok(c, add_special_tokens=False)["input_ids"] for c in candidates]
    seqs = [pre + c for c in cands]
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    inp = torch.tensor([[pad] * (L - len(s)) + s for s in seqs]).to(model.device)
    mask = (inp != pad).long()
    out = []
    V = int(getattr(model.config, "vocab_size", 151936))
    with torch.no_grad():
        ch = max(1, min(4, int(1.2e9 / max(L * V * 4, 1))))
        for s0 in range(0, inp.size(0), ch):
            logits = model(input_ids=inp[s0:s0 + ch], attention_mask=mask[s0:s0 + ch]).logits
            for k in range(logits.size(0)):
                c = cands[s0 + k]
                logp = F.log_softmax(logits[k, L - len(c) - 1:L, :].float(), dim=-1)
                out.append(sum(float(logp[j, c[j]]) for j in range(len(c))) / len(c))
            del logits
    return out


def explain_one(model, tok, o, head):
    t0 = time.time()
    ch = o.get("parent_chain", []) or []
    base = head.replace("{chain}", " | ".join([(c.get("msg") or "") for c in ch[-5:]])[:800])
    vs = batch_scores(model, tok, base + '{"verdict":"', VERDICTS)
    verdict = 1 if vs[0] > vs[1] else 0
    ts = batch_scores(model, tok, base + '{"verdict":"' + VERDICTS[1 - verdict] + '","technique_id":"', TTP_ENUM)
    ttp = TTP_ENUM[int(np.argmax(ts))]
    evid = rule_evidence(o)
    lat = time.time() - t0
    is_mal = int(o["label"])
    gt = re.findall(r"T\d{4}(?:\.\d{3})?", str(o.get("technique_id", "")))
    base_gt = {t[:5] for t in gt}
    base_pred = ttp[:5] if ttp != "none" else "none"
    evid_gt = evid != "none"
    # phat NE: model ne (evidence=none) khi chain CO bang chung, hoac tro evidence khong co trong input
    evd_invalid = (evid == "none" and evid_gt) or (evid != "none" and not evid_gt)
    return {
        "nid": o["nid"], "is_mal": is_mal, "verdict": verdict,
        "verdict_score": round(vs[0] - vs[1], 4), "verdict_correct": int(verdict == is_mal),
        "ttp_pred": ttp, "ttp_gt": gt, "ttp_pred_base": base_pred,
        "ttp_correct": int(base_pred in base_gt) if (is_mal and gt) else None,
        "evidence": evid, "evidence_gt_exists": int(evid_gt),
        "hallucinated": int(evd_invalid),
        "evidence_valid": int(not evd_invalid), "grounded": int(not (verdict == 1 and evid == "none")),
    }, lat


def summarize(recs, lat):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.array([r["is_mal"] for r in recs])
    s_raw = np.array([r["verdict_score"] for r in recs])
    nan_mask = ~np.isfinite(s_raw)
    s = np.where(nan_mask, 0.0, s_raw)  # NaN -> trung lap (NaN>0 la False nen khong doi accuracy)
    yf, sf = y[~nan_mask], s[~nan_mask]
    try:
        auc = round(float(roc_auc_score(yf, sf)), 4) if len(np.unique(yf)) == 2 and len(yf) else None
    except Exception:
        auc = None
    try:
        ap = round(float(average_precision_score(yf, sf)), 4) if len(yf) else None
    except Exception:
        ap = None
    scored = [r["ttp_correct"] for r in recs if r["ttp_correct"] is not None]
    hall = [r["hallucinated"] for r in recs if r["is_mal"] == 1]
    conf = collections.Counter(("TP" if r["is_mal"] else "FP") if r["verdict"] == 1 else ("FN" if r["is_mal"] else "TN")
                               for r in recs)
    return {"n": len(recs), "pos": int(y.sum()),
            "verdict_accuracy": round(float((y == (s > 0)).mean()), 4),
            "verdict_AUC": auc, "verdict_AP": ap,
            "ttp_accuracy_on_gt": round(float(np.mean(scored)), 4) if scored else None,
            "ttp_scored_count": len(scored),
            "nan_verdict_scores": int(nan_mask.sum()),
            "nan_nids": [recs[i]["nid"] for i in list(np.where(nan_mask)[0][:20])],
            "hallucination_rate": round(float(np.mean(hall)), 4) if hall else None,
            "halluc_count": int(np.sum(hall)) if hall else 0,
            "evidence_validity": round(float(1 - np.mean(hall)), 4) if hall else None,
            "confusion": {k: conf[k] for k in ("TP", "TN", "FP", "FN")},
            "avg_latency_sec": round(float(np.mean(lat)), 3) if lat else None,
            "ttp_pred_dist": dict(collections.Counter(r["ttp_pred"] for r in recs).most_common(10))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/adgen-ttp-bench.jsonl")
    ap.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    ap.add_argument("--shots", type=int, choices=(2, 3), default=2)
    ap.add_argument("--lora", default=None, help="checkpoint LoRA (peft); bo trong = dung base")
    ap.add_argument("--resume", action="store_true", help="luu <out>-partial.json moi 50 mau va chay tiep tu do")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.stdout.reconfigure(line_buffering=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.manual_seed(a.seed)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    print(f"n={len(rows)} pos={sum(r['label'] for r in rows)}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, device_map="auto")
    if a.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.lora)
        print("loaded LoRA", flush=True)
    model.eval()
    head = prompt_head(a.shots)
    part = a.out.replace(".json", "-partial.json")
    recs, done = [], set()
    if a.resume and os.path.exists(part):
        recs = json.load(open(part, encoding="utf-8")).get("records", [])
        done = {x["nid"] for x in recs}
        print(f"RESUME: {len(recs)} mau co san, bo qua", flush=True)
    lat = []
    for i, o in enumerate(rows):
        if o.get("nid") in done:
            continue
        r, t = explain_one(model, tok, o, head)
        recs.append(r)
        lat.append(t)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)
        if a.resume and len(recs) % 50 == 0:
            json.dump({"records": recs}, open(part, "w"))
    summ = summarize(recs, lat)
    print("\n=== AD-GEN CONSTRAINED ===", flush=True)
    for k, v in summ.items():
        print(f"  {k}: {v}", flush=True)
    json.dump({"summary": summ, "records": recs}, open(a.out, "w"), indent=1)
    print(f"SAVED {a.out}", flush=True)


if __name__ == "__main__":
    main()
