"""
Tier-2 explain EVAL multi-source — chay theo thu tu:
  A. alerts-enriched-v2 + gt_and_scores (in-domain OpTC, GT that) -> verdict/evidence/grounding
  B. ttp_holdout 319 (GT TTP tu src) -> TTP correctness
  C. sd-chains 3133 (Splunk attack_data, pos-only) -> transfer sanity (khong do accuracy)
Dung lai chuan hoa v3 (key case-insensitive, cat duoi Human:, fewshot).
Chon mau theo rank/file-order (khong random) de ton trong temporal.
"""
import json, time, re, os, sys, collections
sys.stdout.reconfigure(line_buffering=True)

ENRICHED = "P1/Output/data/alerts-enriched-v2.jsonl"
GT = "P1/Output/data/gt_and_scores.json"
HOLDOUT = "P1/Output/data/ttp_holdout.jsonl"
SD = "P1/Output/data/sd-chains.jsonl"
OUTDIR = "P1/Output/results_phase2"
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
CKPT = "P1/Output/models/lora-05b/checkpoint-113"

TTP_ENUM = ["T1003", "T1003.001", "T1059", "T1059.001", "T1053", "T1053.005",
            "T1071", "T1071.001", "T1218", "T1218.001", "T1218.011", "T1027",
            "T1027.001", "T1562", "T1572", "T1574.002", "T1060", "T1490",
            "T1064", "T1087", "T1087.001", "T1082", "T1083", "none"]
TTP_SET = set(TTP_ENUM)

# ---------- chuan hoa (tu v3) ----------
def get_ci(d, *names):
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
        return None
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
    if t in TTP_SET:
        return t
    return re.match(r"(T\d{4})", t).group(1)

def norm_evid(v):
    if isinstance(v, dict):
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
    return "none"

def elem_text(e):
    """1 node thanh text — chiu 3 schema: msg / image+cmd / node+op."""
    if not isinstance(e, dict):
        return str(e)
    if e.get("msg"):
        return str(e["msg"])
    img = e.get("image") or e.get("node") or ""
    cmd = e.get("cmd")
    if cmd:
        return f"{img} | cmd: {cmd}"
    op = e.get("op")
    return f"{img} {op or ''}".strip()

def build_text(o):
    ch = o.get("parent_chain", None)
    if ch is None:
        ch = o.get("chain", []) or []
    return " | ".join(elem_text(e) for e in ch[-5:])[:800]

def has_cmd_text(o):
    t = build_text(o).lower()
    if "| cmd:" in t:
        v = t.split("| cmd:", 1)[1].strip()[:60]
        return v and v != "none"
    return "cmd:" in t

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
    s = raw.strip().split("Human:")[0]
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

def input_has_field(o, field):
    t = build_text(o)
    if field == "cmdline":
        return bool(has_cmd_text(o))
    if field == "parent_chain":
        ch = o.get("parent_chain", o.get("chain", [])) or []
        return len(ch) >= 2
    if field == "event_seq":
        return len(o.get("event_seq", []) or []) >= 1
    return True

# ---------- load tap ----------
def load_stage_a():
    gt = set(map(str, json.load(open(GT, encoding="utf-8"))["gt_nids"]))
    rows = [json.loads(l) for l in open(ENRICHED, encoding="utf-8") if l.strip()]
    for r in rows:
        r["_mal"] = int(str(r.get("nid")) in gt)
        r["_src"] = "optc_s1"
    pos = [r for r in rows if r["_mal"] == 1]
    neg = sorted([r for r in rows if r["_mal"] == 0],
                 key=lambda r: (r.get("rank") if r.get("rank") is not None else 10**9))
    print(f"A: enriched {len(rows)} (pos {len(pos)}), lay {len(pos)} pos + 28 neg top-rank", flush=True)
    return pos + neg[:28], "A_optc_s1"

def load_stage_b():
    rows = [json.loads(l) for l in open(HOLDOUT, encoding="utf-8") if l.strip()]
    atk = [r for r in rows if str(r.get("src", "")).startswith("ttp:")]
    tpl = [r for r in rows if str(r.get("src", "")) == "ttp_template"]
    for r in atk:
        r["_mal"] = 1
        m = re.findall(r"T\d{4}", str(r.get("src", "")))
        r["_gt_ttp"] = m
    for r in tpl:
        r["_mal"] = 0
        r["_gt_ttp"] = []
    print(f"B: holdout {len(rows)} (atk {len(atk)} co GT TTP, template {len(tpl)}), "
          f"lay {len(atk)} atk + {len(atk)} template (label dao: 0=attack)", flush=True)
    return atk + tpl[:len(atk)], "B_ttp_holdout"

def load_stage_c(n=30):
    rows = [json.loads(l) for l in open(SD, encoding="utf-8") if l.strip()]
    by_src = collections.OrderedDict()
    for r in rows:
        by_src.setdefault(str(r.get("src", "?")), []).append(r)
    out = []
    keys = list(by_src.keys())
    i = 0
    while len(out) < n and i < max(len(v) for v in by_src.values()):
        for k in keys:
            if len(out) >= n:
                break
            if i < len(by_src[k]):
                r = by_src[k][i]
                r["_mal"] = 1  # pos-only
                r["_src"] = r.get("src", "?")
                out.append(r)
        i += 1
    print(f"C: sd {len(rows)} ({len(by_src)} attack), lay {len(out)} mau round-robin "
          f"(pos-only, khong do accuracy)", flush=True)
    return out, "C_sd_transfer"

# ---------- eval ----------
def eval_record(rec, o, gt_ttp):
    v_raw = get_ci(rec, "verdict")
    t_raw = get_ci(rec, "technique_id", "techniqueid", "techniques", "technique_ids")
    e_raw = get_ci(rec, "evidence_field", "evidencefield", "evidence_fields", "evidence")
    v = norm_verdict(v_raw)
    ttp = norm_ttp(t_raw)
    evid = norm_evid(e_raw)
    is_mal = int(o.get("_mal", 0))
    verdict = v if v is not None else 0
    invalid = int(v is None)
    ttp_correct = int(ttp in set(gt_ttp)) if (is_mal and gt_ttp) else None
    evid_valid = int(input_has_field(o, evid))
    grounded = int(not (verdict == 1 and evid == "none"))
    return {"verdict": verdict, "invalid": invalid,
            "verdict_correct": int(verdict == is_mal),
            "ttp_pred": ttp, "ttp_gt": list(gt_ttp), "ttp_correct": ttp_correct,
            "evidence": evid, "evidence_valid": evid_valid, "grounded": grounded,
            "is_mal": is_mal}

def run_stage(model, tok, ttps_str, items, tag):
    records, lat, fails = [], [], 0
    for i, o in enumerate(items):
        prompt = PROMPT.replace("{chain}", build_text(o)).replace("{ttps}", ttps_str)
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
        rec = eval_record(parsed, o, o.get("_gt_ttp", []))
        rec["nid"] = str(o.get("nid", o.get("src", "?")))
        rec["raw"] = raw[:200]
        records.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  [{tag}] {i+1}/{len(items)}", flush=True)
    return records, lat, fails

def summarize(records):
    import numpy as np
    n = len(records)
    s = {"n": n,
         "verdict_accuracy": round(float(np.mean([r["verdict_correct"] for r in records])), 4) if n else 0,
         "invalid_rate": round(float(np.mean([r["invalid"] for r in records])), 4) if n else 0,
         "evidence_validity": round(float(np.mean([r["evidence_valid"] for r in records])), 4) if n else 0,
         "grounding_rate": round(float(np.mean([r["grounded"] for r in records])), 4) if n else 0}
    s["hallucination_rate"] = round(1 - s["evidence_validity"], 4)
    scored = [r["ttp_correct"] for r in records if r["ttp_correct"] is not None]
    s["ttp_accuracy_on_gt"] = round(float(np.mean(scored)), 4) if scored else None
    s["ttp_scored_count"] = len(scored)
    tp = sum(1 for r in records if r["verdict"] == 1 and r["is_mal"] == 1)
    tn = sum(1 for r in records if r["verdict"] == 0 and r["is_mal"] == 0)
    fp = sum(1 for r in records if r["verdict"] == 1 and r["is_mal"] == 0)
    fn = sum(1 for r in records if r["verdict"] == 0 and r["is_mal"] == 1)
    s["confusion"] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
    return s

def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
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
    os.makedirs(OUTDIR, exist_ok=True)
    combined = {}
    for loader in (load_stage_a, load_stage_b, load_stage_c):
        items, tag = loader()
        records, lat, fails = run_stage(model, tok, ttps_str, items, tag)
        s = summarize(records)
        import numpy as np
        s["avg_latency_sec"] = round(float(np.mean(lat)), 3) if lat else 0
        s["parse_fail"] = fails
        s["decisions_per_hour"] = round(3600 / s["avg_latency_sec"], 1) if s["avg_latency_sec"] else 0
        print(f"\n=== {tag} ===", flush=True)
        for k, v in s.items():
            print(f"  {k}: {v}", flush=True)
        if tag.startswith("C_"):
            from collections import Counter
            print("  ttp_pred_dist:", dict(Counter(r["ttp_pred"] for r in records).most_common(8)), flush=True)
            print("  NOTE: sd pos-only -> confusion khong co y nghia accuracy; "
                  "chi doc sensitivity/evidence/parse.", flush=True)
        p = os.path.join(OUTDIR, f"tier2-explain-{tag}.json")
        json.dump({"summary": s, "records": records}, open(p, "w"), indent=2)
        print(f"SAVED {p}", flush=True)
        combined[tag] = s
    json.dump(combined, open(os.path.join(OUTDIR, "tier2-explain-multi-summary.json"), "w"), indent=2)
    print("\nSAVED tier2-explain-multi-summary.json", flush=True)

if __name__ == "__main__":
    main()
