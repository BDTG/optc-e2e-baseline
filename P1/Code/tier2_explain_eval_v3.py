"""
Tier-2 explain EVAL v3 — chuan hoa output (khong train lai).
Fix tu soi raw v2: model chep prompt + sai key + bia TTP + duoi Human:.
"""
import json, time, re, os, sys, random
import numpy as np
sys.stdout.reconfigure(line_buffering=True)
SEED = 42
random.seed(SEED)

DATA = "P1/Output/data/evtx-chains.jsonl"
OUT = "P1/Output/results_phase2/tier2-explain-eval-v3.json"
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
CKPT = "P1/Output/models/lora-05b/checkpoint-113"
N_PER_CLASS = 30

TTP_ENUM = ["T1003", "T1003.001", "T1059", "T1059.001", "T1053", "T1053.005",
            "T1071", "T1071.001", "T1218", "T1218.001", "T1218.011", "T1027",
            "T1027.001", "T1562", "T1572", "T1574.002", "T1060", "T1490",
            "T1064", "T1087", "T1087.001", "T1082", "T1083", "none"]
TTP_SET = set(TTP_ENUM)

def get_ci(d, *names):
    """Lay value khong phan biet hoa/thuong, chap nhan bien the so nhieu."""
    if not isinstance(d, dict):
        return None
    low = {str(k).lower(): v for k, v in d.items()}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None

def norm_verdict(v):
    if isinstance(v, list):
        v = v[0] if v else "none"
    s = str(v or "")
    if "|" in s:
        return None  # chep nguyen enum prompt -> invalid
    u = s.upper()
    if "MALICIOUS" in u:
        return 1
    if "BENIGN" in u:
        return 0
    return None

def norm_ttp(v):
    if isinstance(v, list):
        v = v[0] if v else "none"
    if v is None:
        return "none"
    m = re.search(r"T\d{4}(?:\.\d{3})?", str(v))
    if not m:
        return "none"
    t = m.group(0)
    # giu nguyen neu trong enum, neu chi co technique goc ma enum co sub -> giu goc
    if t in TTP_SET:
        return t
    base = re.match(r"(T\d{4})", t).group(1)
    return base

def norm_evid(v):
    if isinstance(v, dict):
        # {"cmdline": "..."} -> lay key
        for k in v.keys():
            kl = str(k).lower()
            if "cmd" in kl:
                return "cmdline"
            if "parent" in kl:
                return "parent_chain"
            if "event" in kl:
                return "event_seq"
            if kl == "none":
                return "none"
        return "none"
    if isinstance(v, list):
        v = v[0] if v else "none"
    s = str(v or "").lower()
    if "cmd" in s:
        return "cmdline"
    if "parent" in s:
        return "parent_chain"
    if "event" in s:
        return "event_seq"
    if s.strip() == "none":
        return "none"
    return "none"

def gt_ttp_set(chain):
    raw = chain.get("technique_id") or chain.get("ttp") or ""
    ids = re.findall(r"T\d{4}", str(raw))
    return set(ids) if ids else set()

def input_has_field(chain, field):
    pc = chain.get("parent_chain", []) or []
    if field == "cmdline":
        for c in pc:
            cmd = c.get("cmd")
            if cmd and str(cmd).lower() != "none":
                return True
        for c in pc:
            m = c.get("msg") or ""
            if "| cmd:" in m:
                v = m.split("| cmd:", 1)[1].strip()
                if v and v.lower() != "none":
                    return True
        return False
    if field == "parent_chain":
        return len(pc) >= 2
    if field == "event_seq":
        return len(chain.get("event_seq", []) or []) >= 1
    return True

def build_chain_text(c):
    chain = c.get("parent_chain", []) or []
    parts = [f"{pc.get('image', '')} | cmd: {pc.get('cmd', 'None') or 'None'}"
             for pc in chain[-5:]]
    return " | ".join(parts)[:800]

FEWSHOT = (
    'Example: Chain: C:\\Windows\\System32\\rundll32.exe | cmd: rundll32 systemnet.dll -E\n'
    '{"verdict":"MALICIOUS","technique_id":"T1218","evidence_field":"cmdline","confidence":"high"}\n'
)

PROMPT = (
    "You are a security analyst. Classify this Windows process provenance chain.\n"
    + FEWSHOT +
    "Chain: {chain}\n"
    "Respond ONLY with one JSON object, no other text:\n"
    '{{"verdict":"MALICIOUS|BENIGN","technique_id":"one of {ttps} or none",'
    '"evidence_field":"cmdline|parent_chain|event_seq|none","confidence":"high|low"}}\n'
    "JSON:"
)

def parse_json_out(raw):
    s = raw.strip()
    s = s.split("Human:")[0]  # cat duoi role confusion
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```.*$", "", s, flags=re.DOTALL)
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*?\}", s, re.DOTALL)
    if m:
        cand = m.group(0)
        for fixer in (lambda x: x,
                      lambda x: x.replace("'", '"'),
                      lambda x: re.sub(r",\s*}", "}", x.replace("'", '"'))):
            try:
                return json.loads(fixer(cand))
            except Exception:
                continue
    return None

def eval_record(rec, chain):
    v_raw = get_ci(rec, "verdict")
    t_raw = get_ci(rec, "technique_id", "techniqueid", "techniques", "technique_ids")
    e_raw = get_ci(rec, "evidence_field", "evidencefield", "evidence_fields", "evidence")
    v = norm_verdict(v_raw)
    ttp = norm_ttp(t_raw)
    evid = norm_evid(e_raw)
    gt = gt_ttp_set(chain)
    is_mal = int(bool(chain.get("is_suspicious")))
    verdict = v if v is not None else 0  # invalid -> dem nhu sai (bao thu)
    invalid = int(v is None)
    ttp_correct = int(ttp in gt) if (is_mal and gt) else None
    evid_valid = int(input_has_field(chain, evid))
    grounded = int(not (verdict == 1 and evid == "none"))
    return {"verdict": verdict, "invalid": invalid,
            "verdict_correct": int(verdict == is_mal),
            "ttp_pred": ttp, "ttp_gt": sorted(gt), "ttp_correct": ttp_correct,
            "evidence": evid, "evidence_valid": evid_valid, "grounded": grounded}

def summarize(records):
    n = len(records)
    vacc = np.mean([r["verdict_correct"] for r in records]) if n else 0
    ev = np.mean([r["evidence_valid"] for r in records]) if n else 0
    gr = np.mean([r["grounded"] for r in records]) if n else 0
    inv = np.mean([r["invalid"] for r in records]) if n else 0
    scored = [r["ttp_correct"] for r in records if r["ttp_correct"] is not None]
    ttp_acc = np.mean(scored) if scored else None
    return {"n": n, "verdict_accuracy": round(float(vacc), 4),
            "invalid_rate": round(float(inv), 4),
            "ttp_accuracy_on_gt": round(float(ttp_acc), 4) if ttp_acc is not None else None,
            "ttp_scored_count": len(scored),
            "evidence_validity": round(float(ev), 4),
            "hallucination_rate": round(float(1 - ev), 4),
            "grounding_rate": round(float(gr), 4)}

def run_slm(chains):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="auto")
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, CKPT)
        print("loaded LoRA", flush=True)
    except Exception as e:
        print(f"LoRA fail ({e}), dung base", flush=True)
    model.eval()
    ttps_str = ", ".join(t for t in TTP_ENUM if t != "none")
    records, lat, fails = [], [], 0
    for i, c in enumerate(chains):
        prompt = PROMPT.replace("{chain}", build_chain_text(c)).replace("{ttps}", ttps_str)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        t0 = time.time()
        out = model.generate(**enc, max_new_tokens=80, do_sample=False,
                             no_repeat_ngram_size=3,
                             pad_token_id=tok.eos_token_id,
                             eos_token_id=tok.eos_token_id)
        lat.append(time.time() - t0)
        raw = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        parsed = parse_json_out(raw)
        if parsed is None:
            fails += 1
            parsed = {"verdict": "BENIGN", "technique_id": "none",
                      "evidence_field": "none", "confidence": "low"}
        rec = eval_record(parsed, c)
        rec["nid"] = c.get("nid"); rec["raw"] = raw[:200]; rec["is_mal"] = int(bool(c.get("is_suspicious")))
        records.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(chains)}", flush=True)
    return records, float(np.mean(lat)) if lat else 0, fails

def main():
    chains = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    susp = [c for c in chains if c.get("is_suspicious")]
    ben = [c for c in chains if not c.get("is_suspicious")]
    print(f"Tong {len(chains)} (susp {len(susp)}, benign {len(ben)})", flush=True)
    susp_s = random.sample(susp, min(N_PER_CLASS, len(susp)))
    ben_s = random.sample(ben, min(N_PER_CLASS, len(ben)))
    subset = susp_s + ben_s
    random.shuffle(subset)
    print(f"Tap CAN BANG: {len(subset)}", flush=True)
    records, avg_lat, fails = run_slm(subset)
    s = summarize(records)
    s["avg_latency_sec"] = round(avg_lat, 3)
    s["parse_fail"] = fails
    s["decisions_per_hour"] = round(3600 / avg_lat, 1) if avg_lat else 0
    print("\n=== TIER-2 EXPLAIN v3 (chuan hoa + fewshot) ===")
    for k, v in s.items():
        print(f"  {k}: {v}")
    tp = sum(1 for r in records if r["verdict"] == 1 and r["is_mal"] == 1)
    tn = sum(1 for r in records if r["verdict"] == 0 and r["is_mal"] == 0)
    fp = sum(1 for r in records if r["verdict"] == 1 and r["is_mal"] == 0)
    fn = sum(1 for r in records if r["verdict"] == 0 and r["is_mal"] == 1)
    print(f"  confusion: TP={tp} TN={tn} FP={fp} FN={fn}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"summary": s, "records": records}, open(OUT, "w"), indent=2)
    print(f"\nSAVED {OUT}", flush=True)

if __name__ == "__main__":
    main()
