"""P3/RQ1a — SLM 0.5B chay tren subset cascade (12 mal + 192 ben) — RQ1a FP-real analysis
o CPU GPD 7800X3D, fp32, constrained scoring giong product. Do p50/p95 + confusion.
Doi chieu: SLM verdict-tier-2 xval 0.775 (train tren OpTC) — tap nho nay kiem tra true positive."""
import json, sys, os, time, re, threading, statistics
sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn.functional as F

try:
    import psutil
    PROC = psutil.Process()
    def rss(): return PROC.memory_info().rss / 1e9
except ImportError:
    PROC = None
    def rss(): return -1.0

SUB = r"F:\backup\OpTC-thesis\P1\Output\data\rq1a-subset.jsonl"
OUT = r"F:\backup\OpTC-thesis\P1\Output\results_phase2\rq1a-subset-result.json"
OUTT = r"F:\backup\OpTC-thesis\P1\Output\results_phase2\raw-explain-rq1a.json"
TTP_ENUM = ["T1003", "T1059", "T1053", "T1071", "T1218", "T1027",
            "T1562", "T1574", "T1490", "T1087", "T1082", "T1083", "none"]
VERDICTS = ["MALICIOUS", "BENIGN"]
FEWSHOT = (
    'Example: Chain: powershell.exe | cmd: powershell -nop -w hidden -enc SQBFAFgAIA==\n'
    '{"verdict":"MALICIOUS","technique_id":"T1059","evidence_field":"cmdline","confidence":"high"}\n'
    'Example: Chain: svchost.exe | chrome.exe | cmd: none\n'
    '{"verdict":"BENIGN","technique_id":"none","evidence_field":"none","confidence":"high"}\n'
)
PREFIX = ("You are a security analyst. Classify this Windows process provenance chain.\n"
          + FEWSHOT +
          "Chain: {chain}\n"
          'Respond with one JSON object: {{"verdict":"{v}","technique_id":"{t}","evidence_field":"{e}"}}\n'
          "JSON:")


def flat(ch):
    if isinstance(ch, str):
        try: ch = json.loads(ch.replace("'", '"'))
        except Exception: ch = []
    return " | ".join([(c.get("msg") or "") for c in (ch or [])][-5:])


def batch_scores(model, tok, prefix, candidates, chunk=4):
    pre = tok(prefix, add_special_tokens=False)["input_ids"]
    out = []
    for ci in range(0, len(candidates), chunk):
        cands = [tok(c, add_special_tokens=False)["input_ids"] for c in candidates[ci:ci + chunk]]
        seqs = [pre + c for c in cands]
        L = max(len(s) for s in seqs)
        pad = tok.pad_token_id
        inp = torch.tensor([[pad] * (L - len(s)) + s for s in seqs])
        mask = torch.tensor([[(1 if i - (L - len(s)) >= 0 else 0) for i in range(L)] for s in seqs])
        with torch.no_grad():
            logits = model(input_ids=inp, attention_mask=mask).logits[:, -L - 1:-1].float()
        logp = F.log_softmax(logits, dim=-1)
        for row, c in enumerate(cands):
            m = len(c)
            lp = sum(float(logp[row, j, c[j]]) for j in range(m)) / m
            out.append(lp)
        del logits, logp, inp, mask
    return out


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    rows = [json.loads(l) for l in open(SUB, encoding="utf-8")]
    print(f"n={len(rows)} mal={sum(r['label'] for r in rows)}", flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                                 torch_dtype=torch.float32).to("cpu")
    model.eval()
    recs = []; lat = []
    pk = [0.0]; stop = threading.Event()
    def s2():
        while not stop.is_set():
            v = rss()
            if v > pk[0]: pk[0] = v
            time.sleep(0.05)
    th = threading.Thread(target=s2, daemon=True); th.start()
    for i, o in enumerate(rows):
        t0 = time.time()
        txt = flat(o.get("parent_chain", []))
        try:
            raw = None
            if not isinstance(o.get("parent_chain"), list):
                try:
                    o["parent_chain"] = json.loads(str(o["parent_chain"]).replace("'", '"'))
                except Exception:
                    o["parent_chain"] = []
            txt = flat(o.get("parent_chain", []))[:800]
        except Exception:
            txt = ""
        base = PREFIX.replace("{chain}", txt)
        vs = batch_scores(model, tok, base + '{"verdict":"', VERDICTS)
        vscore = vs[0] - vs[1]
        verdict = 1 if vs[0] > vs[1] else 0
        vtxt = VERDICTS[0] if verdict == 1 else VERDICTS[1]
        ts = batch_scores(model, tok, base + '{"verdict":"' + vtxt + '","technique_id":"', TTP_ENUM)
        ttp = TTP_ENUM[int(torch.tensor(ts).argmax())]
        lat.append(time.time() - t0)
        is_mal = int(o["label"])
        recs.append({"nid": o["nid"], "label": is_mal, "verdict": verdict,
                     "verdict_score": round(vscore, 4), "ttp_pred": ttp})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)
    stop.set()
    tp = sum(1 for r in recs if r["verdict"] == 1 and r["label"] == 1)
    tn = sum(1 for r in recs if r["verdict"] == 0 and r["label"] == 0)
    fp = sum(1 for r in recs if r["verdict"] == 1 and r["label"] == 0)
    fn = sum(1 for r in recs if r["verdict"] == 0 and r["label"] == 1)

    try:
        from sklearn.metrics import roc_auc_score
        y = np.array([r["label"] for r in recs])
        s = np.array([r["verdict_score"] for r in recs])
        auc = round(float(roc_auc_score(y, s)), 4) if len(set(y)) > 1 else None
    except Exception:
        auc = None

    summ = {"n": len(recs), "mal": sum(r["label"] for r in recs),
            "verdict_acc": round((tp + tn) / max(tp+tn+fp+fn, 1), 4),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "verdict_AUC": auc,
            "ttp_pred_dist": dict(collections.Counter(r["ttp_pred"] for r in recs).most_common(6)),
            "p50_s": round(statistics.median(lat), 2), "p95_s": round(statistics.quantiles(lat, n=20)[18], 2) if len(lat) > 10 else None,
            "peak_rss_gb": round(pk[0], 2)}
    json.dump({"summary": summ, "recs": recs}, open(OUTT, "w", encoding="utf-8"), indent=1)
    json.dump(summ, open(OUT, "w", encoding="utf-8"), indent=1)
    print("SUMMARY:", json.dumps(summ), flush=True)
    print("SAVED", OUT, flush=True)
    print("RQ1A_DONE", flush=True)
import collections
main()
