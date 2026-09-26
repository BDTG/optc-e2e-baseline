"""SLM explain batch=8 de do latency GPU that su"""
import json, time, re, torch
import numpy as np
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE="Qwen/Qwen2.5-0.5B-Instruct"
CKPT="P1/Output/models/lora-05b/checkpoint-113"
DATA="P1/Output/data/alerts-enriched-v2.jsonl"
GT="P1/Output/data/gt_and_scores.json"
OUT="P1/Output/results_phase2/slm-explain-s1-b8.json"

gt=set(json.load(open(GT))["gt_nids"])
alerts=[json.loads(l) for l in open(DATA, encoding='utf-8') if l.strip()]
gt_set=set([str(a["nid"]) for a in alerts if str(a["nid"]) in gt])

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

s1_alerts=[a for a in alerts if all([has_cmd(a), has_chain(a), has_events(a)])]
print(f"S1: {len(s1_alerts)} alerts", flush=True)

tok=AutoTokenizer.from_pretrained(BASE, padding_side="left")
if tok.pad_token is None: tok.pad_token = tok.eos_token
base=AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16,
    device_map="auto")
model=PeftModel.from_pretrained(base, CKPT)
model.eval()

PROMPT_TEMPLATE = (
    "You are a security analyst. Given the provenance chain of a process, "
    "explain what this process is doing and classify it.\n"
    "Chain: {chain}\n"
    "Output format:\n"
    "Verdict: [MALICIOUS or BENIGN]\n"
    "Subject: <main subject process>\n"
    "Action: <what it does in 1 sentence>\n"
    "TTP: <ATT&CK technique ID if suspicious, else NONE>\n"
    "Explanation: <1 sentence citing specific process/cmd evidence>"
)

def build_chain_text(a):
    chain=a.get("parent_chain",[]) or []
    seq=a.get("event_seq",[]) or []
    parts=[c.get("msg","") for c in chain[-5:]]
    parts+=[f"{e.get('src_msg','')} -> {e.get('dst_msg','')}" for e in seq[-5:]]
    return " | ".join([p for p in parts if p])[:800]

prompts=[PROMPT_TEMPLATE.format(chain=build_chain_text(a)) for a in s1_alerts]
labels=[1 if str(a["nid"]) in gt_set else 0 for a in s1_alerts]

BATCH=8
results=[]
latencies=[]
n_correct=0
n_ttp=0
n_hallu=0

print(f"Generating with batch={BATCH} on {len(prompts)} alerts...", flush=True)
t_start=time.time()
for i in range(0, len(prompts), BATCH):
    batch_p=prompts[i:i+BATCH]
    batch_y=labels[i:i+BATCH]
    t0=time.time()
    try:
        enc=tok(batch_p, return_tensors="pt", truncation=True,
                max_length=480, padding=True).to(model.device)
        out=model.generate(**enc, max_new_tokens=120, do_sample=False,
                          pad_token_id=tok.eos_token_id)
        gens=[tok.decode(out[k][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
              for k in range(len(batch_p))]
    except Exception as e:
        gens=[f"ERROR: {e}"] * len(batch_p)
    elapsed=time.time()-t0
    per_alert_lat=elapsed/len(batch_p)
    latencies.extend([per_alert_lat]*len(batch_p))

    for j,gen in enumerate(gens):
        a=s1_alerts[i+j]
        v_match=re.search(r"Verdict:\s*(MALICIOUS|BENIGN)", gen, re.I)
        slm_v=1 if v_match and v_match.group(1).upper()=="MALICIOUS" else 0
        if slm_v == batch_y[j]: n_correct += 1
        if re.search(r"TTP:\s*T\d{4}", gen, re.I): n_ttp += 1
        # rough hallucination check
        out_exe=set(re.findall(r"\b[a-z]+\.exe\b", gen.lower()))
        in_exe=set(re.findall(r"subject\s+([^\s|]+\.exe)", build_chain_text(a).lower()))
        if out_exe - in_exe: n_hallu += 1

        results.append({
            "nid":a["nid"],"is_mal":batch_y[j],
            "gen":gen,"slm_verdict":slm_v,
            "latency_sec":per_alert_lat,
        })
    if (i+BATCH) % 80 == 0:
        print(f"  {i+BATCH}/{len(prompts)} | per-alert={per_alert_lat:.2f}s "
              f"| throughput={BATCH/per_alert_lat:.1f}/s", flush=True)

total=time.time()-t_start
summary={
    "n":len(s1_alerts), "batch":BATCH,
    "verdict_accuracy":n_correct/len(s1_alerts),
    "ttp_rate":n_ttp/len(s1_alerts),
    "hallucination_rate":n_hallu/len(s1_alerts),
    "latency_mean_sec":float(np.mean(latencies)),
    "latency_p95_sec":float(np.percentile(latencies,95)),
    "total_time_sec":float(total),
    "decisions_per_sec":float(len(s1_alerts)/total),
    "decisions_per_hour":float(len(s1_alerts)/total*3600),
}
with open(OUT,"w") as f:
    json.dump({"summary":summary,"per_alert":results}, f, indent=2)
print(f"\n=== BATCH={BATCH} SUMMARY ===", flush=True)
print(f"Verdict accuracy: {summary['verdict_accuracy']:.4f}", flush=True)
print(f"TTP rate: {summary['ttp_rate']:.4f}", flush=True)
print(f"Hallucination: {summary['hallucination_rate']:.4f}", flush=True)
print(f"Latency mean: {summary['latency_mean_sec']:.2f}s ({summary['decisions_per_hour']:.0f}/hour)", flush=True)
print(f"Total: {summary['total_time_sec']:.0f}s = {summary['total_time_sec']/60:.1f}min", flush=True)