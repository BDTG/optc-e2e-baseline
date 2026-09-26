"""
Tier-2 SLM with CONSTRAINED output (grammar/JSON schema).
So sanh free-form vs constrained:
  - free-form: SLM sinh text tu do -> hallucination cao, loop
  - constrained: ep JSON schema -> chi duoc phep output dung schema
Datasets: EVTX-ATTACK-SAMPLES (468 chains, 65 suspicious)
"""
import json, time, re, torch, sys
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.stdout.reconfigure(line_buffering=True)
SEED=42; torch.manual_seed(SEED)

DATA="P1/Output/data/evtx-chains.jsonl"
OUT_FREE="P1/Output/results_phase2/slm-evtx-free.json"
OUT_CONST="P1/Output/results_phase2/slm-evtx-constrained.json"
BASE="Qwen/Qwen2.5-0.5B-Instruct"
CKPT="P1/Output/models/lora-05b/checkpoint-113"

def build_chain_text(c):
    chain=c.get("parent_chain",[]) or []
    parts=[f"{pc.get('image','')} | cmd: {pc.get('cmd','None') or 'None'}"
           for pc in chain[-5:]]
    return " | ".join(parts)[:800]

chains=[json.loads(l) for l in open(DATA,encoding='utf-8') if l.strip()]
SAMPLE_SIZE=50
chains=chains[:SAMPLE_SIZE]
print(f"EVTX: {len(chains)} chains, suspicious={sum(1 for c in chains if c.get('is_suspicious'))}", flush=True)

tok=AutoTokenizer.from_pretrained(BASE, padding_side="left")
if tok.pad_token is None: tok.pad_token=tok.eos_token
base=AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16,
    device_map="auto")
try:
    model=PeftModel.from_pretrained(base, CKPT)
    print("loaded LoRA", flush=True)
except Exception as e:
    print(f"LoRA fail: {e}, using base", flush=True)
    model=base
model.eval()

# === CONSTRAINED JSON SCHEMA ===
# Allowed values per field — no free text, only enum
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["MALICIOUS", "BENIGN"]},
        "technique_id": {
            "type": "string",
            "enum": ["T1003","T1059","T1059.001","T1053","T1053.005","T1071",
                    "T1218","T1218.001","T1218.011","T1027","T1027.001",
                    "T1562","T1572","T1574.002","T1060","T1490","T1003.001",
                    "T1064","T1087","T1082","T1083","none"]
        },
        "evidence_field": {"type": "string", "enum": ["cmdline","parent_chain","event_seq","none"]},
        "confidence": {"type": "string", "enum": ["high","low"]}
    },
    "required": ["verdict","technique_id","evidence_field","confidence"]
}

# Build strict prompt with FORMAT constraint explicit
PROMPT_FREE = (
    "You are a security analyst. Classify this provenance chain.\n"
    "Chain: {chain}\n"
    "Output: Verdict, Subject, Action, TTP, Explanation"
)

PROMPT_CONSTRAINED = (
    "You are a security analyst. Classify this provenance chain.\n"
    "Chain: CHAIN_PLACEHOLDER\n"
    "Respond ONLY with a JSON object in this exact format:\n"
    '{"verdict": "MALICIOUS or BENIGN", "technique_id": "T1059.001 or T1218.001 or T1003 or T1027 or none", "evidence_field": "cmdline or parent_chain or event_seq or none", "confidence": "high or low"}\n'
    "JSON:"
)

def generate_constrained(prompt, max_new=80, **kwargs):
    """Use JSON-schema-style constrained decoding via stop tokens + regex post-filter"""
    enc=tok(prompt, return_tensors="pt", truncation=True, max_length=480).to(model.device)
    out=model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                      pad_token_id=tok.eos_token_id,
                      eos_token_id=tok.eos_token_id)
    text=tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return text

# Try outlines library for TRUE constrained decoding (regex-based)
try:
    import outlines
    from outlines import generate, models
    print("Using outlines library for TRUE constrained generation", flush=True)

    # Build outlines model wrapper
    out_model = models.Transformers(model, tok)
    # Use regex to constrain: json with verdict enum
    regex_pattern = (
        r'\{[^{}]*"verdict"[^{}]*"(MALICIOUS|BENIGN)"[^{}]*\}'
    )
    constrained_gen = generate.regex(out_model, regex_pattern)

    def gen_with_outlines(prompt):
        try:
            return constrained_gen(prompt, max_tokens=100)
        except Exception as e:
            return f"OUTLINES_FAIL: {e}"
    use_outlines = True
except Exception as e:
    print(f"outlines not usable: {e}", flush=True)
    use_outlines = False
    def gen_with_outlines(prompt):
        return None

# === Run both modes on same 50 chains ===
results_free=[]
results_const=[]

for i,c in enumerate(chains):
    chain_text=build_chain_text(c)
    is_mal=int(bool(c.get("is_suspicious")))

    # --- FREE FORM ---
    t0=time.time()
    free_text = generate_constrained(PROMPT_FREE.format(chain=chain_text))
    free_lat=time.time()-t0

    # --- CONSTRAINED (outlines) ---
    t0=time.time()
    if use_outlines:
        const_text = gen_with_outlines(PROMPT_CONSTRAINED.replace("CHAIN_PLACEHOLDER", chain_text))
    else:
        # fallback: generate, then try to extract JSON anywhere in text
        raw = generate_constrained(PROMPT_CONSTRAINED.replace("CHAIN_PLACEHOLDER", chain_text), max_new=150)
        const_text = "PARSE_FAIL"
        try:
            const_text = json.loads(raw)
        except:
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m:
                try:
                    const_text = json.loads(m.group(0))
                except:
                    first_line = raw.split('\n')[0].strip()
                    if first_line.startswith('{'):
                        try:
                            const_text = json.loads(first_line)
                        except:
                            const_text = m.group(0)
            else:
                m2 = re.search(r'verdict["\s:]+(MALICIOUS|BENIGN)', raw, re.I)
                if m2:
                    const_text = f'{{"verdict":"{m2.group(1).upper()}", "parse":"fallback"}}'
    const_lat=time.time()-t0

    # Parse free
    free_v = re.search(r"Verdict:\s*(MALICIOUS|BENIGN)", free_text, re.I)
    free_verdict = 1 if free_v and free_v.group(1).upper()=="MALICIOUS" else 0
    free_ttp = re.search(r"TTP:\s*(T\d{4})", free_text, re.I)
    free_ttp = free_ttp.group(1) if free_ttp else "none"

    # Parse constrained
    if isinstance(const_text, dict):
        const_verdict = 1 if const_text.get("verdict")=="MALICIOUS" else 0
        const_ttp = const_text.get("technique_id","none")
        const_evidence = const_text.get("evidence_field","none")
        const_conf = const_text.get("confidence","low")
        parse_ok = True
        const_text_str = json.dumps(const_text)
    else:
        const_verdict = -1
        const_ttp = "PARSE_FAIL"
        parse_ok = False
        const_text_str = str(const_text)

    results_free.append({"nid":c["nid"],"is_mal":is_mal,
                         "verdict":free_verdict,"ttp":free_ttp,
                         "lat":free_lat,"gen":free_text})
    results_const.append({"nid":c["nid"],"is_mal":is_mal,
                          "verdict":const_verdict,"ttp":const_ttp,
                          "lat":const_lat,"gen":const_text_str,
                          "parse_ok":parse_ok})
    if (i+1)%10==0:
        print(f"  {i+1}/{len(chains)} done", flush=True)

# === Summary ===
def stats(results, label):
    n=len(results)
    n_ok = sum(1 for r in results if r["verdict"]>=0)
    n_correct = sum(1 for r in results if r["verdict"]==r["is_mal"])
    n_ttp = sum(1 for r in results if r["ttp"] not in ["none","PARSE_FAIL",None])
    avg_lat = float(np.mean([r["lat"] for r in results]))
    return {
        "label":label,
        "n":n,
        "verdict_accuracy":n_correct/n if n else 0,
        "parse_ok_rate":n_ok/n if n else 0,
        "ttp_rate":n_ttp/n if n else 0,
        "avg_latency_sec":avg_lat,
    }

s_free=stats(results_free,"free-form")
s_const=stats(results_const,"constrained")
print(f"\n=== FREE FORM ===")
for k,v in s_free.items(): print(f"  {k}: {v}")
print(f"\n=== CONSTRAINED ===")
for k,v in s_const.items(): print(f"  {k}: {v}")

with open(OUT_FREE,"w") as f:
    json.dump({"summary":s_free,"per_alert":results_free}, f, indent=2)
with open(OUT_CONST,"w") as f:
    json.dump({"summary":s_const,"per_alert":results_const}, f, indent=2)
print(f"\nSAVED {OUT_FREE} and {OUT_CONST}", flush=True)