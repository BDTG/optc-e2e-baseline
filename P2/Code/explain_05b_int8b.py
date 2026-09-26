"""INT8 refine: quantize than model NHUNG giu lm_head fp32 (logits chinh xac hon).
Chay 200 mau de do lai acc + latency."""
import json, time, os, sys, re, collections
import numpy as np
import torch
import torch.nn.functional as F

SAMPLE = r"C:\Users\BDTG\Desktop\Backup\OpTC-thesis\P1\Output\data\adgen-ttp-bench.jsonl"
OUT = r"C:\Users\BDTG\Desktop\Backup\OpTC-thesis\P1\Output\results_phase2\raw-explain-int8-lmfp32.json"
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
LIMIT = 200
sys.stdout.reconfigure(line_buffering=True)

TTP_ENUM = ["T1003", "T1059", "T1053", "T1071", "T1218", "T1027",
            "T1562", "T1574", "T1490", "T1087", "T1082", "T1083", "none"]
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
    import torch.quantization as tq
    torch.manual_seed(42)
    rows = [json.loads(l) for l in open(SAMPLE, encoding="utf-8")][:LIMIT]
    print(f"n={len(rows)} mode=int8-lmfp32 threads={torch.get_num_threads()}", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32).to("cpu")
    model.eval()
    lm_head_fp32 = model.lm_head
    model = tq.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.lm_head = lm_head_fp32  # giu lm_head fp32
    model.eval()
    print("INT8 than model + lm_head fp32", flush=True)
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
        lat.append(time.time() - t0)
        is_mal = int(o["label"])
        gt = re.findall(r"T\d{4}(?:\.\d{3})?", str(o.get("technique_id", "")))
        base_gt = set(re.match(r"(T\d{4})", t).group(1) for t in gt)
        base_pred = re.match(r"(T\d{4})", ttp).group(1) if ttp != "none" else "none"
        recs.append({"nid": o["nid"], "is_mal": is_mal, "verdict": verdict,
                     "verdict_score": round(vscore, 4),
                     "ttp_pred": ttp, "ttp_pred_base": base_pred,
                     "ttp_correct": int(base_pred in base_gt) if (is_mal and gt) else None})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)
    scored = [r["ttp_correct"] for r in recs if r["ttp_correct"] is not None]
    n_mal_pred = sum(1 for r in recs if r["verdict"] == 1)
    summ = {"n": len(recs), "mode": "int8-lmfp32",
            "verdict_acc": round(sum(1 for r in recs if r["verdict"] == r["is_mal"]) / len(recs), 4),
            "n_pred_malicious": n_mal_pred,
            "ttp_accuracy_on_gt": round(float(np.mean(scored)), 4) if scored else None,
            "avg_latency_sec": round(float(np.mean(lat)), 3),
            "ttp_pred_dist": dict(collections.Counter(r["ttp_pred"] for r in recs).most_common(8))}
    json.dump({"summary": summ, "recs": recs}, open(OUT, "w", encoding="utf-8"), indent=1)
    print("SUMMARY:", json.dumps(summ), flush=True)
    print("SAVED", OUT, flush=True)


main()
