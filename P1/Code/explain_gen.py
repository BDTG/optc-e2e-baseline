"""Phuong an (b): SLM/teacher TU SINH rationale tu do (free-generation) tren 40 mau pack,
de judge cham groundedness noi dung. Chay tren GPU (GPD3N9T).
Usage: python explain_gen.py --model Qwen/Qwen2.5-0.5B-Instruct --out rationale-05b.json
"""
import argparse, json, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--bench", default=r"C:\Users\BDTG\optc-bench\adgen-ttp-bench.jsonl")
ap.add_argument("--nids", default=r"C:\Users\BDTG\optc-bench\gen40-nids.json")
ap.add_argument("--out", required=True)
a = ap.parse_args()

nids = set(json.load(open(a.nids, encoding="utf-8")))
rows = [json.loads(l) for l in open(a.bench, encoding="utf-8") if json.loads(l).get("nid") in nids]
print("matched:", len(rows), flush=True)

tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    a.model, torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
model.to(dev).eval()

PROMPT = ("You are a SOC analyst. Read the alert chain and explain in 2-3 sentences "
          "why it is MALICIOUS or BENIGN, citing specific evidence (commands, processes, events).\n"
          "End with exactly two lines:\nVERDICT: MALICIOUS or BENIGN\nTTP: Txxxx (e.g. T1059) or none\n\n"
          "Example:\nChain:\ncmd: powershell -enc SQBuAHYAbwBrAGUALQByAG8AZgBpAGwAZQAgACs9IA positiv | parent: svchost.exe -> powershell.exe\n"
          "Explanation: The chain shows powershell.exe spawned with a base64-encoded (-enc) command, "
          "a classic obfuscation pattern. The parent svchost.exe spawning powershell is abnormal. "
          "This matches command-and-scripting-interpreter abuse.\nVERDICT: MALICIOUS\nTTP: T1059\n\n"
          "Now your turn.\nChain:\n{narr}\nExplanation:")

out = []
for i, o in enumerate(rows):
    ch = o.get("parent_chain", []) or []
    narr = " | ".join([(c.get("msg") or "") for c in ch[-5:]])[:1500]
    enc = tok(PROMPT.format(narr=narr), return_tensors="pt").to(dev)
    t0 = time.time()
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=220, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    txt = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    out.append({"nid": o["nid"], "label": o.get("label"),
                "rationale": txt.strip(), "latency_s": round(time.time() - t0, 2)})
    if (i + 1) % 10 == 0:
        print(f"{i+1}/{len(rows)}", flush=True)
json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
print("SAVED", a.out, flush=True)
