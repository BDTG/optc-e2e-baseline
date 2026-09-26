import json, time, re, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import psutil, pynvml, threading, os

MODELS = [
    "protectai/deberta-v3-base-prompt-injection-v2",
    "jackaduma/SecBERT",
    "AungMoonLord/bert-log-anomaly-detection",
]
DATA="P1/Output/data/alerts-enriched-v2.jsonl"
GT="P1/Output/data/gt_and_scores.json"
SAMPLE_SIZE=100
BATCH=16

def build_chain_text_opc(a):
    chain=a.get("parent_chain",[]) or []
    seq=a.get("event_seq",[]) or []
    parts=[c.get("msg","") for c in chain[-5:]]
    parts+=[f"{e.get('src_msg','')} -> {e.get('dst_msg','')}" for e in seq[-5:]]
    return " | ".join([p for p in parts if p])[:800]
def has_cmd(r):
    chain=r.get("parent_chain",[]) or []
    for c in chain:
        msg=c.get("msg","") or ""
        if "| cmd:" in msg:
            v=msg.split("| cmd:",1)[1].strip()
            if v and v.lower()!="none": return True
    return False
def has_chain(r): return len(r.get("parent_chain",[]) or [])>=2
def has_events(r): return len(r.get("event_seq",[]) or [])>=3

gt=set(json.load(open(GT))["gt_nids"])
alerts=[json.loads(l) for l in open(DATA, encoding='utf-8') if l.strip()]
s1=[a for a in alerts if all([has_cmd(a), has_chain(a), has_events(a)])][:SAMPLE_SIZE]
print(f"Dataset v2 S1 n={len(s1)} suspicious={sum(str(a['nid']) in gt for a in s1)}", flush=True)
texts=[build_chain_text_opc(a) for a in s1]
labels=[1 if str(a["nid"]) in gt else 0 for a in s1]

# system info
pynvml.nvmlInit()
handle=pynvml.nvmlDeviceGetHandleByIndex(0)
sys_info={"gpu_name": pynvml.nvmlDeviceGetName(handle), "ram_gb": psutil.virtual_memory().total/1e9}
print(f"GPU {sys_info['gpu_name']} RAM {sys_info['ram_gb']:.1f}GB", flush=True)

results=[]
for mid in MODELS:
    print(f"\n=== Loading {mid} ===", flush=True)
    try:
        tok=AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        model=AutoModelForSequenceClassification.from_pretrained(mid, trust_remote_code=True, dtype=torch.float16, device_map="auto")
    except Exception as e:
        print(f"  load fail {e}, trying without trust_remote_code", flush=True)
        tok=AutoTokenizer.from_pretrained(mid)
        model=AutoModelForSequenceClassification.from_pretrained(mid, dtype=torch.float16, device_map="auto")
    model.eval()
    # warmup
    try:
        with torch.no_grad():
            out=model(**tok(["hello world"], return_tensors="pt", truncation=True, max_length=512).to(model.device))
    except: pass

    # benchmark
    latencies=[]
    preds=[]
    t_start=time.time()
    for i in range(0, len(texts), BATCH):
        batch=texts[i:i+BATCH]
        t0=time.time()
        enc=tok(batch, return_tensors="pt", truncation=True, max_length=512, padding=True).to(model.device)
        with torch.no_grad():
            out=model(**enc)
            logits=out.logits
            # binary: take argmax, or if 2 labels, 1 = injection/malicious
            pred=torch.argmax(logits, dim=-1).cpu().numpy()
            # For models with 2 labels where label 1 = malicious/injection, use that
            # For SecBERT etc, labels may be different - just use argmax
            preds.extend(pred.tolist())
        latencies.extend([ (time.time()-t0)/len(batch) ] * len(batch))
    total=time.time()-t_start

    # metrics - note: these classifiers are NOT trained on OpTC, so accuracy is zero-shot transfer
    # For deberta prompt-injection, label 1 = injection, which we map to malicious
    # For SecBERT, label 1 may not be malicious - just measure raw
    n_correct=sum(1 for p,y in zip(preds, labels) if int(p)==y)
    # Also compute per-class
    from sklearn.metrics import average_precision_score
    try:
        # Use logits for AP if available (need to re-run to get probs)
        pass
    except: pass

    summary={"model":mid,"n":len(texts),"batch":BATCH,"verdict_accuracy":n_correct/len(texts),"pred_malicious_rate":sum(preds)/len(preds),"true_malicious_rate":sum(labels)/len(labels),"latency_mean_sec":float(np.mean(latencies)),"latency_p95_sec":float(np.percentile(latencies,95)),"total_time_sec":float(total),"decisions_per_sec":float(len(texts)/total),"decisions_per_hour":float(len(texts)/total*3600),"vram_gb":float(pynvml.nvmlDeviceGetMemoryInfo(handle).used/1e9)}
    print(f"  acc={summary['verdict_accuracy']:.3f} pred_mal={summary['pred_malicious_rate']:.3f} true_mal={summary['true_malicious_rate']:.3f} latency={summary['latency_mean_sec']:.4f}s {summary['decisions_per_hour']:.0f}/h VRAM={summary['vram_gb']:.2f}GB", flush=True)
    results.append(summary)
    del model, tok
    torch.cuda.empty_cache()

with open("P1/Output/results_phase2/classifier-benchmark.json","w") as f:
    json.dump({"system":sys_info,"runs":results}, f, indent=2)
print(f"\nSAVED classifier-benchmark.json", flush=True)
