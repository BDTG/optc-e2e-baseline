"""
Tier-2 SLM explanation EVALUATION — do CHAT LUONG giai thich, khong do detection.

Khac voi tier2_slm_constrained.py (chi do verdict_accuracy + ttp_rate),
file nay do 3 truc quyet dinh gia tri cua SLM o tang 2:

  1. TTP CORRECTNESS  — technique_id model chon co KHOP ground-truth khong
                        (khong phai chi "co xuat T-ID hay khong")
  2. EVIDENCE VALIDITY — evidence_field model chon co THUC SU ton tai trong
                        input khong (chong hallucination co kiem chung):
                        neu chon "cmdline" ma input cmd=None -> hallucinate
  3. GROUNDING         — verdict + evidence co nhat quan khong:
                        MALICIOUS + evidence=none -> khong co can cu -> ke

So sanh voi baseline: TF-IDF chi ra label, KHONG giai thich duoc
=> 3 truc tren la san SLM doc quyen. Do de chung minh vai tro SLM,
khong phai de thang AP.

Chay tren TAP GIAU (EVTX suspicious hoac 8-filter 39), khong chay toan bo.
"""
import json, time, re, os, sys
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
SEED = 42

DATA = "P1/Output/data/evtx-chains.jsonl"
OUT = "P1/Output/results_phase2/tier2-explain-eval.json"
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
CKPT = "P1/Output/models/lora-05b/checkpoint-113"

# Danh sach TTP hop le (enum cho constrained). Dong bo voi tier2_slm_constrained
TTP_ENUM = ["T1003", "T1003.001", "T1059", "T1059.001", "T1053", "T1053.005",
            "T1071", "T1071.001", "T1218", "T1218.001", "T1218.011", "T1027",
            "T1027.001", "T1562", "T1572", "T1574.002", "T1060", "T1490",
            "T1064", "T1087", "T1087.001", "T1082", "T1083", "none"]


def norm_ttp(t):
    """Chuan hoa TTP ve technique goc (bo sub-technique) de so khop mem."""
    if not t or t in ("none", "PARSE_FAIL", None):
        return "none"
    m = re.match(r"(T\d{4})", str(t))
    return m.group(1) if m else "none"


def gt_ttp_set(chain):
    """Lay tap TTP ground-truth cua 1 chain (co the nhieu, dang 'T1218+T1053')."""
    raw = chain.get("technique_id") or chain.get("ttp") or ""
    ids = re.findall(r"T\d{4}", str(raw))
    return set(norm_ttp(i) for i in ids) if ids else set()


def input_has_field(chain, field):
    """
    Kiem chung evidence: field model chon co THUC SU ton tai trong input khong.
    Day la hang rao chong hallucination.
    """
    pc = chain.get("parent_chain", []) or []
    if field == "cmdline":
        for c in pc:
            cmd = c.get("cmd")
            if cmd and str(cmd).lower() != "none":
                return True
        # co the cmd nam trong msg
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
    if field == "none":
        return True  # "none" luon hop le
    return False


def build_chain_text(c):
    chain = c.get("parent_chain", []) or []
    parts = [f"{pc.get('image', '')} | cmd: {pc.get('cmd', 'None') or 'None'}"
             for pc in chain[-5:]]
    return " | ".join(parts)[:800]


PROMPT = (
    "You are a security analyst. Classify this Windows process provenance chain.\n"
    "Chain: {chain}\n"
    "Respond ONLY with one JSON object, no other text:\n"
    '{{"verdict":"MALICIOUS|BENIGN","technique_id":"one of {ttps} or none",'
    '"evidence_field":"cmdline|parent_chain|event_seq|none","confidence":"high|low"}}\n'
    "JSON:"
)


def parse_json_out(raw):
    """Rut JSON tu output. Tra ve dict hoac None."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def eval_record(rec, chain):
    """
    Cham 1 output SLM tren 3 truc chat luong.
    rec: dict da parse tu SLM. chain: alert goc (co ground-truth).
    """
    verdict = 1 if rec.get("verdict") == "MALICIOUS" else 0
    ttp = norm_ttp(rec.get("technique_id"))
    evid = rec.get("evidence_field", "none")

    gt = gt_ttp_set(chain)
    is_mal = int(bool(chain.get("is_suspicious")))

    # Truc 1: TTP dung? (chi tinh tren node malicious co GT)
    ttp_correct = None
    if is_mal and gt:
        ttp_correct = int(ttp in gt)

    # Truc 2: evidence co that trong input? (chong hallucinate)
    evid_valid = int(input_has_field(chain, evid))

    # Truc 3: grounding — MALICIOUS ma khong co evidence -> ke
    grounded = int(not (verdict == 1 and evid == "none"))

    return {
        "verdict": verdict,
        "verdict_correct": int(verdict == is_mal),
        "ttp_pred": ttp,
        "ttp_gt": sorted(gt),
        "ttp_correct": ttp_correct,
        "evidence": evid,
        "evidence_valid": evid_valid,
        "grounded": grounded,
    }


def summarize(records):
    n = len(records)
    verdict_acc = np.mean([r["verdict_correct"] for r in records]) if n else 0
    evid_valid = np.mean([r["evidence_valid"] for r in records]) if n else 0
    grounded = np.mean([r["grounded"] for r in records]) if n else 0
    # hallucination = evidence KHONG hop le (model bia can cu khong co)
    halluc = 1 - evid_valid
    # TTP accuracy chi tren cac node co GT
    ttp_scored = [r["ttp_correct"] for r in records if r["ttp_correct"] is not None]
    ttp_acc = np.mean(ttp_scored) if ttp_scored else None
    return {
        "n": n,
        "verdict_accuracy": round(float(verdict_acc), 4),
        "ttp_accuracy_on_gt": round(float(ttp_acc), 4) if ttp_acc is not None else None,
        "ttp_scored_count": len(ttp_scored),
        "evidence_validity": round(float(evid_valid), 4),
        "hallucination_rate": round(float(halluc), 4),
        "grounding_rate": round(float(grounded), 4),
    }


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
    records = []
    lat = []
    parse_fail = 0
    for i, c in enumerate(chains):
        prompt = PROMPT.format(chain=build_chain_text(c), ttps=ttps_str)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=480).to(model.device)
        t0 = time.time()
        out = model.generate(**enc, max_new_tokens=80, do_sample=False,
                             no_repeat_ngram_size=3,  # chan loop
                             pad_token_id=tok.eos_token_id,
                             eos_token_id=tok.eos_token_id)
        lat.append(time.time() - t0)
        raw = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        parsed = parse_json_out(raw)
        if parsed is None:
            parse_fail += 1
            parsed = {"verdict": "BENIGN", "technique_id": "none",
                      "evidence_field": "none", "confidence": "low"}
        rec = eval_record(parsed, c)
        rec["nid"] = c.get("nid")
        rec["raw"] = raw[:200]
        records.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(chains)}", flush=True)
    return records, float(np.mean(lat)) if lat else 0, parse_fail


def main():
    chains = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    # Chi lay tap GIAU: node suspicious (co tin hieu de giai thich)
    rich = [c for c in chains if c.get("is_suspicious")]
    if not rich:
        rich = chains  # fallback neu chua co nhan
    print(f"Tong {len(chains)} chains, tap giau (suspicious) = {len(rich)}", flush=True)

    records, avg_lat, parse_fail = run_slm(rich)
    summary = summarize(records)
    summary["avg_latency_sec"] = round(avg_lat, 3)
    summary["parse_fail"] = parse_fail
    summary["decisions_per_hour"] = round(3600 / avg_lat, 1) if avg_lat else 0

    print("\n=== TIER-2 EXPLANATION QUALITY (tap giau) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n--- Doc ket qua ---")
    print(f"  Verdict acc {summary['verdict_accuracy']}: bao malicious/benign dung bao nhieu")
    if summary["ttp_accuracy_on_gt"] is not None:
        print(f"  TTP acc {summary['ttp_accuracy_on_gt']} tren {summary['ttp_scored_count']} node co GT: "
              f"map dung technique — thu TF-IDF KHONG lam duoc")
    print(f"  Hallucination {summary['hallucination_rate']}: ti le bia can cu khong co trong input")
    print(f"  Grounding {summary['grounding_rate']}: ti le verdict co can cu that")
    print(f"  => So voi TF-IDF (chi ra label, 0 giai thich): SLM cung cap TTP+evidence "
          f"kiem chung duoc. Day la truc SLM doc quyen.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"summary": summary, "records": records}, open(OUT, "w"), indent=2)
    print(f"\nSAVED {OUT}", flush=True)


if __name__ == "__main__":
    main()
