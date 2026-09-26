"""Diag: generate ra JSON chu zua tu TC LoRA - su dung dung prompt format the train."""
import json, sys, torch
sys.stdout.reconfigure(line_buffering=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

CKPT = "C:/Users/BDTG/AppData/Local/Temp/rq1b-adapter-42"
SUB = r"F:\backup\OpTC-thesis\P1\Output\data\rq1a-subset.jsonl"


def flat(ch):
    if isinstance(ch, str):
        try: ch = json.loads(ch.replace("'", '"'))
        except Exception: ch = []
    return " | ".join([(c.get("msg") or "") for c in (ch or [])][-5:])


tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
if tok.pad_token is None: tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base, CKPT, torch_dtype=torch.float32).eval()

recs = [json.loads(l) for l in open(SUB, encoding="utf-8")]
samples = []
for lab in (1, 0):
    for r in recs:
        ch = r.get("parent_chain", [])
        if isinstance(ch, list) and len(ch) >= 2 and int(r["label"]) == lab:
            samples.append(r)
            break

for r in samples:
    ch = flat(r["parent_chain"])[:600]
    p = f"Chain: {ch}\nRespond JSON: {{verdict,technique_id}}\n"
    ids = tok(p, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=40, do_sample=False,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"--- lab={r['label']} nid={r['nid']}")
    print("GEN:", gen[:180])
