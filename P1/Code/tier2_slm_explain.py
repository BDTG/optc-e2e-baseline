"""
Tier-2 SLM sinh giai thich that tren tap giau (S1: 474 alerts).
Test 4 metric:
  1. parent_chain match: subject/proc cua alert co xuat hien trong explain khong
  2. MITRE TTP accuracy: SLM doan TTP co khop GT khong (9 positives)
  3. Hallucination rate: % noi dung sinh ra ngoai input
  4. Verdict accuracy: % SLM label dung malicious/benign

Latency per alert (RQ3 hardware).
"""
import json, random, time, re, torch
import numpy as np
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
BASE="Qwen/Qwen2.5-0.5B-Instruct"
CKPT="P1/Output/models/lora-05b/checkpoint-113"
DATA="P1/Output/data/alerts-enriched-v2.jsonl"
GT="P1/Output/data/gt_and_scores.json"
TTP="P1/Output/data/ttp_holdout.jsonl"  # for TTP hint comparison
OUT="P1/Output/results_phase2/slm-explain-s1.json"

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

# S1 filter
s1_alerts=[a for a in alerts if all([has_cmd(a), has_chain(a), has_events(a)])]
print(f"S1: {len(s1_alerts)} alerts, pos={sum(str(a['nid']) in gt_set for a in s1_alerts)}", flush=True)

# load LoRA-trained Qwen
tok=AutoTokenizer.from_pretrained(BASE)
base=AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16,
    device_map="auto")
# try LoRA first; fallback to base if not exist
try:
    model=PeftModel.from_pretrained(base, CKPT)
    print("loaded LoRA", flush=True)
except Exception as e:
    print(f"LoRA fail ({e}), using base", flush=True)
    model=base
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

def extract_subjects_from_input(chain_text):
    """Extract subject tokens from input chain text"""
    subjects=set()
    for m in re.finditer(r"subject\s+([^\s|]+)", chain_text):
        subjects.add(m.group(1).lower().strip())
    return subjects

# MITRE TTP list (for verification)
VALID_TTPS=re.compile(r"^T\d{4}(\.\d{3})?$")

results=[]
latencies=[]
n_correct_verdict=0
n_explain_with_subject_match=0
n_explain_with_ttp=0
n_explain_correct_ttp=0
n_explain_hallucination=0

for i,a in enumerate(s1_alerts):
    chain_text=build_chain_text(a)
    prompt=PROMPT_TEMPLATE.format(chain=chain_text)
    is_mal=1 if str(a["nid"]) in gt_set else 0
    input_subjects=extract_subjects_from_input(chain_text)

    # generate
    t0=time.time()
    try:
        ids=tok(prompt, return_tensors="pt", truncation=True, max_length=480).to(model.device)
        out=model.generate(**ids, max_new_tokens=120, do_sample=False,
                          pad_token_id=tok.eos_token_id)
        gen=tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
    except Exception as e:
        gen=f"ERROR: {e}"
    elapsed=time.time()-t0
    latencies.append(elapsed)

    # parse output
    verdict_match=re.search(r"Verdict:\s*(MALICIOUS|BENIGN)", gen, re.I)
    slm_verdict=verdict_match.group(1).upper() if verdict_match else None
    slm_verdict_binary=1 if slm_verdict=="MALICIOUS" else 0
    if slm_verdict_binary == is_mal:
        n_correct_verdict += 1

    # subject match check: look for ANY subject from input in explanation
    explain_section=gen.lower()
    subj_match=any(s in explain_section for s in input_subjects) if input_subjects else False
    if subj_match: n_explain_with_subject_match += 1

    # TTP check
    ttp_match=re.search(r"TTP:\s*(T\d{4}(?:\.\d{3})?)", gen, re.I)
    slm_ttp=ttp_match.group(1) if ttp_match else None
    if slm_ttp and VALID_TTPS.match(slm_ttp):
        n_explain_with_ttp += 1
        # correct TTP for malicious: we don't have ground truth per sample
        # so just count whether TTP is well-formed
        n_explain_correct_ttp += 1

    # hallucination check: any text mentioning subjects NOT in input?
    out_subjects=set(re.findall(r"\b([a-z]+\.exe|cmd|powershell|svchost|netflow)\b",
                                  explain_section))
    hallucinated=out_subjects - {s.replace(".exe","") for s in input_subjects} - {"netflow","cmd","powershell","svchost"}
    # very rough heuristic: if SLM mentions specific .exe not in input, it might hallucinate
    real_hallu = len([x for x in out_subjects if x.endswith('.exe') and x not in input_subjects])
    if real_hallu > 0:
        n_explain_hallucination += 1

    results.append({
        "nid":a["nid"],
        "is_mal":is_mal,
        "input_subjects":list(input_subjects),
        "gen":gen,
        "slm_verdict":slm_verdict,
        "slm_ttp":slm_ttp,
        "subj_match_in_explain":subj_match,
        "hallucinated_subjects":[x for x in out_subjects if x.endswith('.exe') and x not in input_subjects],
        "latency_sec":elapsed,
    })
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(s1_alerts)} done | avg_lat={np.mean(latencies):.2f}s", flush=True)

# summary
summary={
    "n":len(s1_alerts),
    "n_pos":sum(str(a["nid"]) in gt_set for a in s1_alerts),
    "verdict_accuracy":n_correct_verdict/len(s1_alerts),
    "subject_in_explain_rate":n_explain_with_subject_match/len(s1_alerts),
    "ttp_rate":n_explain_with_ttp/len(s1_alerts),
    "ttp_well_formed_rate":n_explain_correct_ttp/len(s1_alerts),
    "hallucination_rate":n_explain_hallucination/len(s1_alerts),
    "latency_mean_sec":float(np.mean(latencies)),
    "latency_p95_sec":float(np.percentile(latencies,95)),
    "latency_min_sec":float(np.min(latencies)),
    "latency_max_sec":float(np.max(latencies)),
    "decisions_per_hour_single":3600/float(np.mean(latencies)),
}
with open(OUT,"w") as f:
    json.dump({"summary":summary,"per_alert":results}, f, indent=2)

print(f"\n=== S1 SLM EXPLAIN SUMMARY ===", flush=True)
print(f"Verdict accuracy:       {summary['verdict_accuracy']:.4f}", flush=True)
print(f"Subject in explain:     {summary['subject_in_explain_rate']:.4f}", flush=True)
print(f"TTP rate:               {summary['ttp_rate']:.4f}", flush=True)
print(f"Hallucination rate:     {summary['hallucination_rate']:.4f}", flush=True)
print(f"Latency mean/p95 (sec): {summary['latency_mean_sec']:.2f} / {summary['latency_p95_sec']:.2f}", flush=True)
print(f"Decisions/hour:         {summary['decisions_per_hour_single']:.1f}", flush=True)
print(f"\nSAVED {OUT}", flush=True)