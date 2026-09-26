import json, time, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from sklearn.metrics import average_precision_score
import statistics

V2=r"D:\OpTC-thesis\P1\Output\alerts_enriched_v2.jsonl"
GT=r"D:\OpTC-thesis\P1\Output\gt_and_scores.json"

SYSTEM_BASE="""You are a cybersecurity analyst. Examine the endpoint telemetry provided and classify the activity.

Respond with exactly one of:
CLASSIFICATION: MALICIOUS
CONFIDENCE: [0.0-1.0]
REASON: [one sentence explaining the decision]
"""
FEWSHOT="""
Example 1:
Process Identity: subject /Device/HarddiskVolume1/Windows/Temp/evil.exe | cmd: "powershell -enc aGVsbG8="
Process Tree: explorer.exe [CREATE] -> evil.exe
Recent Events: evil.exe -> [CREATE] -> powershell.exe
CLASSIFICATION: MALICIOUS
CONFIDENCE: 0.95
REASON: Encoded PowerShell with suspicious parent.

Example 2:
Process Identity: subject /Device/HarddiskVolume1/Windows/System32/svchost.exe | cmd: None
Process Tree: services.exe [CREATE] -> svchost.exe
Recent Events: svchost.exe -> [OPEN] -> netflow 10.0.0.1:80
CLASSIFICATION: BENIGN
CONFIDENCE: 0.85
REASON: Normal service host with expected network.
"""
TEMPLATE="""Classify this endpoint activity:

=== Process Identity ===
{self_label}

=== Process Tree (up to 3 ancestors) ===
{parent_chain}

=== Recent Events (chronological) ===
{event_seq}

Is this MALICIOUS or BENIGN?"""

def fmt(a, max_events=10):
    def pc(p): return f"  {p.get('msg') or p.get('node','?')} [{p.get('op','?')}]"
    pc_str="\n".join(pc(p) for p in a.get("parent_chain",[])) or "  (no ancestors)"
    def ev(e): return f"  {e.get('src_msg') or e.get('src','?')} -> [{e.get('op','?')}] -> {e.get('dst_msg') or e.get('dst','?')}"
    ev_str="\n".join(ev(e) for e in a.get("event_seq",[])[:max_events]) or "  (no events)"
    return TEMPLATE.format(self_label=a.get("self_label",""), parent_chain=pc_str, event_seq=ev_str)

def parse(resp):
    pred="uncertain"; conf=0.5
    for l in resp.split("\n"):
        l=l.strip()
        if l.upper().startswith("CLASSIFICATION:"):
            v=l.split(":",1)[1].strip().upper()
            if "MALICIOUS" in v: pred="malicious"
            elif "BENIGN" in v: pred="benign"
        elif l.upper().startswith("CONFIDENCE:"):
            try: conf=float(l.split(":",1)[1].strip())
            except: pass
    return pred, conf

def run(model_id, subset_n=40, fewshot=True, max_input=2048, max_new=64):
    print(f"\n=== {model_id} subset {subset_n} fewshot={fewshot} max_input={max_input} ===")
    gt_nids=set(json.load(open(GT))["gt_nids"])
    alerts=[json.loads(l) for l in open(V2, encoding='utf-8')]
    gt_alerts=[a for a in alerts if str(a["nid"]) in gt_nids]
    ben=[a for a in alerts if str(a["nid"]) not in gt_nids]
    random.seed(42)
    ben_sample=random.sample(ben, subset_n - len(gt_alerts))
    subset=gt_alerts+ben_sample
    random.shuffle(subset)
    print(f"subset {len(subset)} GT {sum(1 for a in subset if str(a['nid']) in gt_nids)}")
    tok=AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    SYSTEM=SYSTEM_BASE + (FEWSHOT if fewshot else "")
    print("loading model...")
    t0=time.time()
    model=AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True)
    model.eval()
    print(f"loaded {time.time()-t0:.1f}s")
    # warmup
    warm=fmt(subset[0])
    msg=[{"role":"system","content":SYSTEM},{"role":"user","content":warm}]
    text=tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inputs=tok(text, return_tensors="pt", truncation=True, max_length=max_input)
    with torch.no_grad(): _=model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id)
    results=[]; lats=[]
    for i,a in enumerate(subset):
        user=fmt(a)
        messages=[{"role":"system","content":SYSTEM},{"role":"user","content":user}]
        text=tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs=tok(text, return_tensors="pt", truncation=True, max_length=max_input)
        t1=time.perf_counter()
        with torch.no_grad():
            out=model.generate(**inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        dt=(time.perf_counter()-t1)*1000
        lats.append(dt)
        resp=tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred,conf=parse(resp)
        score=conf if pred=="malicious" else (1-conf if pred=="benign" else 0.5)
        results.append({"nid":a["nid"],"gt":1 if str(a["nid"]) in gt_nids else 0,"pred":pred,"conf":conf,"score":score,"resp":resp[:300],"lat":dt})
        if (i+1)%10==0:
            print(f" {i+1}/{len(subset)} avg {statistics.mean(lats):.0f}ms | last {pred} {conf:.2f} score {score:.2f}")
    y_true=np.array([r["gt"] for r in results])
    y_score=np.array([r["score"] for r in results])
    ap=average_precision_score(y_true, y_score) if y_true.sum()>0 else 0
    print(f"AP {ap:.4f} baseline 0.254")
    print(f"lat p50 {statistics.median(lats):.0f} p95 {np.percentile(lats,95):.0f} mean {statistics.mean(lats):.0f}")
    for r in results:
        if r["gt"]==1:
            print(f" GT {r['nid']} pred {r['pred']} conf {r['conf']:.2f} score {r['score']:.2f} | {r['resp'][:120]}")
    # confusion at 0.5
    pred_mal=(y_score>0.5).astype(int)
    tp=(pred_mal & y_true).sum(); fp=(pred_mal & (1-y_true)).sum(); fn=((1-pred_mal) & y_true).sum()
    print(f"TP {tp} FP {fp} FN {fn} prec {tp/max(tp+fp,1):.3f} rec {tp/max(tp+fn,1):.3f}")
    verdict="GO" if ap>0.254 else "NO-GO"
    print(f"VERDICT {verdict}")
    return ap, statistics.median(lats)

for mid in ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]:
    try:
        run(mid, subset_n=40, fewshot=True, max_input=2048, max_new=64)
    except Exception as e:
        print(f"failed {mid}: {e}")
        import traceback; traceback.print_exc()
