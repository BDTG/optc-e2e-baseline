"""
Go/No-go: SLM vs TF-IDF 0.254 AP
- Loads Qwen2.5-{0.5B,1.5B,3B} via transformers CPU
- Runs zero-shot on V2 alerts subset (12 GT + 188 benign = 200)
- Measures latency p50/p95 and AP
- Compares to TF-IDF baseline OOF AP 0.254

Usage:
  python slm_go_nogo.py --model Qwen/Qwen2.5-0.5B-Instruct --subset 200 --max_tokens 32 --max_input_tokens 512
  python slm_go_nogo.py --model Qwen/Qwen2.5-3B-Instruct --subset 500

Requires: transformers, accelerate, torch
"""
import json, time, random, argparse, re, os, statistics
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

V2_PATH = r"D:\OpTC-thesis\P1\Output\alerts_enriched_v2.jsonl"
GT_PATH = r"D:\OpTC-thesis\P1\Output\gt_and_scores.json"
OUTPUT_DIR = r"D:\OpTC-thesis\P1\Output"

SYSTEM_PROMPT = """You are a cybersecurity analyst. Examine the endpoint telemetry provided and classify the activity.

Respond with exactly one of:
CLASSIFICATION: MALICIOUS
CONFIDENCE: [0.0-1.0]
REASON: [one sentence explaining the decision]
"""

USER_TEMPLATE = """Classify this endpoint activity:

=== Process Identity ===
{self_label}

=== Process Tree (up to 3 ancestors) ===
{parent_chain}

=== Recent Events (chronological) ===
{event_seq}

Is this MALICIOUS or BENIGN?"""

def format_alert(alert):
    def pc(p):
        label = p.get("msg") or p.get("node","?")
        return f"  {label} [{p.get('op','?')}]"
    parent_chain = "\n".join(pc(p) for p in alert.get("parent_chain",[])) or "  (no ancestors)"
    def ev(e):
        src = e.get("src_msg") or e.get("src","?")
        dst = e.get("dst_msg") or e.get("dst","?")
        return f"  {src} -> [{e.get('op','?')}] -> {dst}"
    event_seq = "\n".join(ev(e) for e in alert.get("event_seq",[])[:20]) or "  (no events)"
    return USER_TEMPLATE.format(self_label=alert.get("self_label","unknown"), parent_chain=parent_chain, event_seq=event_seq)

def parse_response(resp):
    pred="uncertain"; conf=0.5; reason=resp.strip()[:200]
    for line in resp.split("\n"):
        l=line.strip()
        if l.upper().startswith("CLASSIFICATION:"):
            v=l.split(":",1)[1].strip().upper()
            if "MALICIOUS" in v: pred="malicious"
            elif "BENIGN" in v: pred="benign"
        elif l.upper().startswith("CONFIDENCE:"):
            try: conf=float(l.split(":",1)[1].strip())
            except: pass
        elif l.upper().startswith("REASON:"):
            reason=l.split(":",1)[1].strip()[:200]
    return pred, conf, reason

def load_subset(v2_path, gt_nids, subset_n=200, seed=42):
    random.seed(seed); np.random.seed(seed)
    alerts=[json.loads(l) for l in open(v2_path, encoding='utf-8')]
    # alerts are sorted by original score; we need stratified sample: all GT (12) + random benign
    gt_alerts=[a for a in alerts if str(a["nid"]) in gt_nids]
    benign=[a for a in alerts if str(a["nid"]) not in gt_nids]
    print(f"Full V2: {len(alerts)} GT {len(gt_alerts)} benign {len(benign)}")
    # sample benign
    need_benign = subset_n - len(gt_alerts)
    if need_benign <0:
        # if subset smaller than GT, sample GT too
        gt_alerts = random.sample(gt_alerts, subset_n)
        benign_sample=[]
    else:
        benign_sample = random.sample(benign, min(need_benign, len(benign)))
    subset = gt_alerts + benign_sample
    random.shuffle(subset)
    print(f"Subset {len(subset)}: GT {sum(1 for a in subset if str(a['nid']) in gt_nids)} benign {sum(1 for a in subset if str(a['nid']) not in gt_nids)}")
    return subset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--subset", type=int, default=200)
    ap.add_argument("--max_tokens", type=int, default=32)
    ap.add_argument("--max_input_tokens", type=int, default=512)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    gt_nids=set(json.load(open(GT_PATH))["gt_nids"])
    subset=load_subset(V2_PATH, gt_nids, subset_n=args.subset)

    # Load model
    print(f"\nLoading {args.model} ...")
    t0=time.time()
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        print("Missing transformers/torch", e); return

    tokenizer=AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # CPU fp32, low_mem
    model=AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s, params approx {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    # Warmup
    print("Warmup...")
    warm_prompt = format_alert(subset[0])
    messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":warm_prompt}]
    text=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs=tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_input_tokens)
    with torch.no_grad():
        _=model.generate(**inputs, max_new_tokens=5, do_sample=False)

    # Inference
    results=[]
    latencies=[]
    print(f"\nRunning {len(subset)} alerts (max_input {args.max_input_tokens} tok, gen {args.max_tokens})...")
    t_all=time.time()
    for i,a in enumerate(subset):
        user_msg=format_alert(a)
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_msg}]
        text=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs=tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_input_tokens)
        input_len=inputs.input_ids.shape[1]
        t1=time.perf_counter()
        with torch.no_grad():
            out=model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        dt=(time.perf_counter()-t1)*1000
        latencies.append(dt)
        # decode only new tokens
        resp=tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        pred,conf,reason=parse_response(resp)
        # score for AP: malicious -> conf, benign -> 1-conf, uncertain -> 0.5
        if pred=="malicious": score=conf
        elif pred=="benign": score=1-conf  # lower conf benign should be lower score? Actually benign high conf => low malicious score
            # need invert: if benign with 0.9 conf, malicious score = 0.1
            # so 1-conf works
        else: score=0.5
        # alternative: confidence directly as malicious probability? For benign, 1-conf is correct.
        results.append({
            "nid": a["nid"], "gt": 1 if str(a["nid"]) in gt_nids else 0,
            "score": score, "pred": pred, "conf": conf, "resp": resp[:300],
            "latency_ms": dt, "input_len": input_len
        })
        if (i+1)%20==0:
            print(f"  {i+1}/{len(subset)} avg {statistics.mean(latencies):.0f}ms p50 {statistics.median(latencies):.0f}ms | last pred {pred} conf {conf:.2f} score {score:.2f}")

    total_t=time.time()-t_all
    print(f"\nDone {len(subset)} in {total_t:.1f}s avg {statistics.mean(latencies):.1f}ms")

    # Metrics
    y_true=np.array([r["gt"] for r in results])
    y_score=np.array([r["score"] for r in results])
    ap_score=average_precision_score(y_true, y_score) if y_true.sum()>0 else 0
    print(f"\n=== RESULTS {args.model} subset {args.subset} ===")
    print(f"AP: {ap_score:.4f} (TF-IDF baseline OOF 0.254, holdout 0.52)")
    print(f"Latency p50 {statistics.median(latencies):.1f}ms p95 {np.percentile(latencies,95):.1f}ms mean {statistics.mean(latencies):.1f}ms")
    print(f"Input len avg {statistics.mean([r['input_len'] for r in results]):.0f} tok")
    # Detailed
    for r in results:
        if r["gt"]==1:
            print(f" GT {r['nid']} pred {r['pred']} conf {r['conf']:.2f} score {r['score']:.2f} latency {r['latency_ms']:.0f}ms | resp: {r['resp'][:100]}")
    # Confusion at threshold 0.5 (score>0.5 => predict malicious)
    pred_mal = (y_score>0.5).astype(int)
    tp=(pred_mal & y_true).sum(); fp=(pred_mal & (1-y_true)).sum(); fn=((1-pred_mal) & y_true).sum(); tn=((1-pred_mal) & (1-y_true)).sum()
    prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
    print(f"Confusion score>0.5: TP {tp} FP {fp} FN {fn} TN {tn} prec {prec:.3f} rec {rec:.3f}")
    # Go/no-go
    baseline=0.254
    verdict="GO" if ap_score > baseline else "NO-GO"
    print(f"\nVERDICT: {verdict} (SLM AP {ap_score:.3f} {'>' if ap_score>baseline else '<='} baseline {baseline})")
    if ap_score <= baseline:
        print("-> H0 NOT eliminated. Recommend REFRAME per Note.md: SLM not competitive zero-shot, need few-shot/LoRA or claim change to FP reduction with human-in-loop, not detection.")
    else:
        print("-> H0 eliminated. Proceed to full 2250 + 10K enrich + host 201/501.")

    # Save
    out_path = args.output or f"{OUTPUT_DIR}/slm_go_nogo_{Path(args.model).name}_n{args.subset}.json"
    with open(out_path,'w') as f:
        json.dump({"model":args.model, "subset":args.subset, "ap":ap_score, "baseline":baseline, "verdict":verdict, "lat_p50":statistics.median(latencies), "lat_p95":float(np.percentile(latencies,95)), "results":results}, f, indent=2)
    print(f"Saved {out_path}")

if __name__=="__main__":
    main()
