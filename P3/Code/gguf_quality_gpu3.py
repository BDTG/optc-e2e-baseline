"""Do chat luong GGUF Qwen3.5-4B Q4_K_M (GPU, endpoint /chat/completions) full 704,
do reasoning_content + thinking template enable_thinking=False (Qwen3 khanh chinh).
Doi chieu: fp32 0.5B = 0.939, INT8 = 0.092, CPU GGUF raw = 0.014."""
import json, subprocess, sys, time, os, re
import urllib.request
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
PORT = 8766
BENCH = r"F:\backup\OpTC-thesis\P1\Output\data\adgen-ttp-bench.jsonl"
OUT = r"F:\backup\OpTC-thesis\P1\Output\results_phase2\rq4-q80-quality.json"


def q(msg, max_tokens=2000):
    body = {"messages": [{"role": "user", "content": msg}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    ch = r["choices"][0]
    return ch["message"], ch.get("finish_reason"), r.get("timings", {})


def main():
    rows = []
    for l in open(r"F:\backup\OpTC-thesis\P1\Output\data\adgen-ttp-bench.jsonl", encoding="utf-8"):
        try:
            rows.append(json.loads(l))
        except Exception:
            continue
    print(f"n={len(rows)} (thinking off via chat_template_kwargs)", flush=True)
    tp = fp = tn = fn = 0; tacc = []; lat = []
    t0 = time.time()
    for i, o in enumerate(rows):
        ch = o.get("parent_chain", []) or []
        txt = " | ".join([(c.get("msg") or "") for c in ch[-5:]])[:800]
        prompt = ('You are a security analyst. Classify this chain. Reply ONLY one JSON line in the exact format '
                  '{"verdict":"MALICIOUS|BENIGN","technique_id":"Txxxx or none","evidence_field":"cmdline|parent_chain|event_seq|none"}. '
                  "No thinking process — answer immediately.\n"
                  "Chain: " + txt)
        t1 = time.time()
        try:
            m, fin, tim = q(prompt, max_tokens=2000)
        except Exception as e:
            print(f"  [{i}] FAIL: {str(e)[:50]}", flush=True)
            continue
        lat.append(time.time() - t1)
        content = (m.get("content") or "")
        mt_v = re.search(r'"verdict"\s*:\s*"(\w+)"', content)
        verdict = 1 if (mt_v and mt_v.group(1).upper() == "MALICIOUS") else 0
        mt_t = re.search(r'"technique_id"\s*:\s*"?(T\d{4}|none)', content)
        pred = mt_t.group(1) if mt_t else "none"
        gt = re.findall(r"T\d{4}(?:\.\d{3})?", str(o.get("technique_id", "")))
        base_gt = set(re.match(r"(T\d{4})", t).group(1) for t in gt)
        base_pred = re.match(r"(T\d{4})", pred).group(1) if re.match(r"T\d{4}", pred) else "none"
        is_mal = int(o["label"])
        if verdict == 1 and is_mal == 1: tp += 1
        elif verdict == 0 and is_mal == 0: tn += 1
        elif verdict == 1: fp += 1
        else: fn += 1
        if is_mal and gt:
            tacc.append(int(base_pred in base_gt))
        if (i + 1) % 50 == 0:
            acc = (tp + tn) / max(tp + tn + fp + fn, 1)
            print(f"  {i+1}/{len(rows)} acc {acc:.3f} tacc {float(np.mean(tacc)):.3f} "
                  f"s/it {float(np.mean(lat)):.1f}", flush=True)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    summ = {"n_total": len(rows),
            "verdict_acc": round(acc, 4), "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "ttp_acc_on_gt": round(float(np.mean(tacc)), 4) if tacc else None,
            "ttp_n": len(tacc),
            "s_per_sample": round(float(np.mean(lat)), 2) if lat else None,
            "wall_s": round(time.time() - t0),
            "backend": "GPU vulkan ngl=-99", "endpoint": "chat+chat_template_kwargs disable_thinking"}
    json.dump(summ, open(OUT, "w", encoding="utf-8"), indent=1)
    print("SUMMARY:", json.dumps(summ), flush=True)
    print("GGUF_Q_GPU_DONE", flush=True)


main()
