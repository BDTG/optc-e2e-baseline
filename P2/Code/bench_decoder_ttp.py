import json, re, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DATA="P1/Output/data/ttp_holdout.jsonl"
MODELS=[("Qwen/Qwen2.5-0.5B-Instruct",None),("Qwen/Qwen2.5-1.5B-Instruct",None)]
PROMPT='Chain: CHAIN_PLACEHOLDER\nRespond ONLY JSON: {"verdict":"MALICIOUS or BENIGN","technique_id":"T1059.001 or T1218.001 or T1003 or T1027 or none","evidence":"cmdline or parent_chain or none"}\nJSON:'

def load_chain(a):
    return " | ".join([(c.get("msg") or "") for c in a.get("parent_chain",[])[:5]])[:600]

alerts=[json.loads(l) for l in open(DATA,encoding='utf-8') if l.strip()]
print(f"TTP holdout n={len(alerts)}", flush=True)
# keep 60 for speed (20 mal + 40 ben)
import random; random.seed(42)
random.shuffle(alerts)
sample=alerts[:60]
print(f"sample {len(sample)} susp={sum(1 for a in sample if a.get('is_suspicious') or a.get('label')==0)}", flush=True)

for model_name, ckpt in MODELS:
    print(f"\n=== {model_name} constrained ===", flush=True)
    tok=AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto", trust_remote_code=True)
    if ckpt: model=PeftModel.from_pretrained(model, ckpt)
    model.eval()
    ok=0; ttp_hit=0; halluc=0; total=len(sample)
    for a in sample:
        chain=load_chain(a)
        prompt=PROMPT.replace("CHAIN_PLACEHOLDER", chain)
        inputs=tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out=model.generate(**inputs, max_new_tokens=60, do_sample=False, pad_token_id=tok.eos_token_id)
        gen=tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        # parse
        m=re.search(r'\{.*?\}', gen, re.S)
        if m:
            try:
                j=json.loads(m.group(0))
                verdict=j.get("verdict","")
                tech=j.get("technique_id","")
                # check
                if verdict in ["MALICIOUS","BENIGN"]: ok+=1
                else: halluc+=1
                # TTP hit if suspicious and tech != none
                is_mal = a.get("is_suspicious") or str(a.get("label"))=="0" or a.get("self_label","").startswith("MAL")
                if is_mal and tech!="none" and tech!="": ttp_hit+=1
                elif not is_mal and tech=="none": ok+=0 # already counted
                else: pass
            except: halluc+=1
        else:
            halluc+=1
    print(f" ver {ok}/{total}={ok/total:.2f} ttp_hit {ttp_hit} halluc {halluc} halluc_rate={halluc/total:.2f}", flush=True)
    del model; torch.cuda.empty_cache()
print("DONE", flush=True)
