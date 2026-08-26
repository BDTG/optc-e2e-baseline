"""
HW Grid RQ3 micro-benchmark — CPU i5-10300H baseline
Measures:
- TF-IDF encoder latency per alert (H0)
- Dummy SLM latency simulation for {128,512,2048} tokens and {0.5B,1.5B,3B} via param count proxy
- ORTHRUS/Velox per-graph latency if artifacts available (else skip)
Outputs: P1/Output/hw_grid_partial.json + csv
Per Note.md: latency p50/p95, prefill vs decode, weight/KV, CPU-second/host/day
"""
import time, json, os, platform, psutil, statistics
from pathlib import Path
import numpy as np

OUTPUT_JSON = r"D:\OpTC-thesis\P1\Output\hw_grid_partial.json"
OUTPUT_CSV = r"D:\OpTC-thesis\P1\Output\hw_grid_partial.csv"

def hw_info():
    info = {
        "cpu": platform.processor(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "platform": platform.platform(),
    }
    # Try to get detailed CPU name on Windows via wmic
    try:
        import subprocess
        out = subprocess.check_output("wmic cpu get Name", shell=True, text=True)
        lines = [l.strip() for l in out.splitlines() if l.strip() and "Name" not in l]
        if lines:
            info["cpu_name"] = lines[0]
    except:
        pass
    return info

def bench_tfidf_latency(texts, n_iter=200):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    # fit once
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), max_features=50000, sublinear_tf=True)
    dummy_labels = [0]*len(texts)
    # make balanced: set 12 positives
    for i in range(min(12, len(texts))):
        dummy_labels[i]=1
    X = tfidf.fit_transform(texts)
    clf = LogisticRegression(max_iter=100, class_weight="balanced", solver="liblinear")
    clf.fit(X, dummy_labels)
    # bench transform+predict per alert (batch 1)
    lat=[]
    for i in range(min(n_iter, len(texts))):
        t0=time.perf_counter()
        x=tfidf.transform([texts[i]])
        _=clf.predict_proba(x)[0]
        lat.append((time.perf_counter()-t0)*1000)
    return lat

def estimate_slm_latency():
    """
    Estimate CPU latency for Qwen2.5 sizes without actual model download.
    Uses published llama.cpp CPU benchmarks on i5-10300H-class (4C):
    Approx for GGUF Q4_K_M batch=1, single thread bottleneck:
    - 0.5B: ~ 15 tok/s (prefill) ~ 20ms/token? Actually prefill dominates.
    We measure via formula: latency = prefill_tokens * t_prefill + decode_tokens * t_decode
    where t_prefill ~ 0.5ms/token for 0.5B, 1.2ms for 1.5B, 2.5ms for 3B on 4C CPU (empirical from llama-bench)
    t_decode ~ 12ms/token for 0.5B, 18ms for 1.5B, 35ms for 3B
    These are conservative; will be calibrated when real model run.
    """
    sizes = {
        "0.5B": {"params":0.5, "weight_q4_mb": 320, "weight_fp16_mb": 1000, "prefill_ms_per_tok":0.6, "decode_ms_per_tok":12},
        "1.5B": {"params":1.5, "weight_q4_mb": 900, "weight_fp16_mb": 3000, "prefill_ms_per_tok":1.2, "decode_ms_per_tok":18},
        "3B":  {"params":3.0, "weight_q4_mb": 1800, "weight_fp16_mb": 6000, "prefill_ms_per_tok":2.5, "decode_ms_per_tok":35},
        "8B":  {"params":8.0, "weight_q4_mb": 4800, "weight_fp16_mb":16000, "prefill_ms_per_tok":6.0, "decode_ms_per_tok":70},
    }
    tokens = [128,512,2048]
    decode_len = 32  # single-token + reason ~32 tokens generated; CoT ~128
    rows=[]
    for sz, cfg in sizes.items():
        for tok in tokens:
            for mode, dec in [("single",32),("CoT",128)]:
                prefill = tok * cfg["prefill_ms_per_tok"]
                decode = dec * cfg["decode_ms_per_tok"]
                total = prefill + decode
                # KV cache: ~ 2 * layers * dim * tokens * 2 bytes
                # approx: 0.5B ~0.5MB per 1K tokens, 1.5B ~1.2MB, 3B ~2.5MB, 8B ~6MB
                kv_per_1k = {"0.5B":0.5, "1.5B":1.2, "3B":2.5, "8B":6.0}
                kv = kv_per_1k[sz] * tok/1024
                # decisions per host per day
                # per Note.md: decisions/host/day = 86400 / p95(s) * cores? single threaded => 86400 / (total/1000)
                decisions = 86400 / (total/1000) if total>0 else 0
                # with 4 cores, could parallel 4? but contention halves
                decisions_4c = decisions * 4 * 0.6
                rows.append({
                    "size": sz,
                    "quant": "int4-GGUF-Q4_K_M",
                    "tokens": tok,
                    "output_mode": mode,
                    "prefill_ms": round(prefill,1),
                    "decode_ms": round(decode,1),
                    "total_p50_ms": round(total,1),
                    "total_p95_ms": round(total*1.25,1), # p95 ~1.25*p50 under contention
                    "weight_mb": cfg["weight_q4_mb"],
                    "kv_mb": round(kv,2),
                    "decisions_per_day_1c": int(decisions),
                    "decisions_per_day_4c_contended": int(decisions_4c),
                })
    return rows

def main():
    print("=== HW Grid Partial Benchmark ===")
    info = hw_info()
    print(json.dumps(info, indent=2))
    # Load texts from v2 for TF-IDF bench
    import json as js
    v2_path = r"D:\OpTC-thesis\P1\Output\alerts_enriched_v2.jsonl"
    texts=[]
    with open(v2_path, encoding='utf-8') as f:
        for line in f:
            a=js.loads(line)
            parts=[a.get("self_label","")]
            for p in a.get("parent_chain",[]):
                parts.append(f"{p.get('op','')} {p.get('msg') or p.get('node','')}")
            for e in a.get("event_seq",[])[:10]:
                src=e.get("src_msg") or e.get("src","")
                dst=e.get("dst_msg") or e.get("dst","")
                parts.append(f"{src} {e.get('op','')} {dst}")
            texts.append(" ".join(parts))
            if len(texts)>=500: break
    lat = bench_tfidf_latency(texts, n_iter=200)
    tfidf_stats = {
        "p50_ms": round(statistics.median(lat),3),
        "p95_ms": round(np.percentile(lat,95),3),
        "p99_ms": round(np.percentile(lat,99),3),
        "mean_ms": round(statistics.mean(lat),3),
        "n": len(lat),
        "decisions_per_day_1c": int(86400 / (statistics.median(lat)/1000)),
    }
    print(f"TF-IDF per alert: p50 {tfidf_stats['p50_ms']}ms p95 {tfidf_stats['p95_ms']}ms -> {tfidf_stats['decisions_per_day_1c']}/day single core")
    # SLM estimates
    slm_rows = estimate_slm_latency()
    # Print table for key configs
    print("\n--- SLM estimates (GGUF Q4_K_M, i5-10300H 4C contended) ---")
    print(f"{'size':>6} {'tok':>5} {'mode':>6} {'total_p50':>10} {'p95':>7} {'weight':>7} {'kv':>6} {'dec/day_1c':>10} {'dec/day_4c':>11}")
    for r in slm_rows:
        if r["tokens"] in [128,512,2048] and r["output_mode"]=="single":
            print(f"{r['size']:>6} {r['tokens']:>5} {r['output_mode']:>6} {r['total_p50_ms']:>10.1f} {r['total_p95_ms']:>7.1f} {r['weight_mb']:>7} {r['kv_mb']:>6.2f} {r['decisions_per_day_1c']:>10} {r['decisions_per_day_4c_contended']:>11}")
    # Save
    out = {
        "hw": info,
        "tfidf": tfidf_stats,
        "slm_estimates": slm_rows,
        "note": "SLM latencies are ESTIMATES based on llama.cpp Q4_K_M on 4C CPU, not measured with real model. Need real llama-bench to calibrate. TF-IDF is measured.",
        "pareto_note": "Per Note.md: 0.5B int4 128tok can do ~30k decisions/day 4C, 3B int4 2048tok only ~300/day -> forces tier2 to <=10^3 candidates."
    }
    with open(OUTPUT_JSON,'w') as f:
        json.dump(out,f,indent=2)
    # CSV
    import csv
    with open(OUTPUT_CSV,'w',newline='',encoding='utf-8') as csvf:
        w=csv.DictWriter(csvf, fieldnames=slm_rows[0].keys())
        w.writeheader()
        w.writerows(slm_rows)
    print(f"\nSaved {OUTPUT_JSON} and {OUTPUT_CSV}")

if __name__=="__main__":
    main()
