"""Smoke test (~2-3 phut): nap Qwen2.5-0.5B fp32, chay 5 mau dau tien end-to-end
(scoring + luu JSON). Neu pass thi moi truong OK, chay run_all duoc."""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

SAMPLE = "data/adgen-ttp-bench.jsonl"
OUT = "output/smoke-test-05b.json"
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
N = 5
sys.stdout.reconfigure(line_buffering=True)
os.makedirs("output", exist_ok=True)

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(), end="")
if torch.cuda.is_available():
    print(f" | gpu: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
else:
    print(" | (CPU mode)")

FEWSHOT = (
    'Example: Chain: powershell.exe | cmd: powershell -nop -w hidden -enc SQBFAFgAIA==\\n'
    '{"verdict":"MALICIOUS","technique_id":"T1059","evidence_field":"cmdline","confidence":"high"}\\n'
    'Example: Chain: svchost.exe | chrome.exe | cmd: none\\n'
    '{"verdict":"BENIGN","technique_id":"none","evidence_field":"none","confidence":"high"}\\n'
)
PREFIX = ("You are a security analyst. Classify this Windows process provenance chain.\\n"
          + FEWSHOT + "Chain: {chain}\\n"
          + 'Respond with one JSON object: {{"verdict":"{v}","technique_id":"{t}","evidence_field":"{e}"}}\\n'
          + "JSON:")
VERDICTS = ["MALICIOUS", "BENIGN"]


def batch_scores(model, tok, prefix, candidates):
    pre = tok(prefix, add_special_tokens=False)["input_ids"]
    cands = [tok(c, add_special_tokens=False)["input_ids"] for c in candidates]
    seqs = [pre + c for c in cands]
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    inp = torch.tensor([[pad] * (L - len(s)) + s for s in seqs]).to(model.device)
    mask = (inp != pad).long()
    out = []
    with torch.no_grad():
        logits = model(input_ids=inp, attention_mask=mask).logits
        for k in range(inp.size(0)):
            c = cands[k]
            m = len(c)
            seg = logits[k, L - m - 1:L, :].float()
            logp = F.log_softmax(seg, dim=-1)
            out.append(sum(float(logp[j, c[j]]) for j in range(m)) / m)
        del logits
    return out


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    rows = [json.loads(l) for l in open(SAMPLE, encoding="utf-8")][:N]
    print(f"data OK: {SAMPLE} ({N} mau thu)")
    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("loading model (lan dau tai ~1GB, can internet)...")
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32, device_map="auto")
    model.eval()
    print("model OK:", BASE)
    recs = []
    for o in rows:
        ch = o.get("parent_chain", []) or []
        txt = " | ".join([(c.get("msg") or "") for c in ch[-5:]])[:800]
        base = PREFIX.split('{{"verdict"')[0].replace("{chain}", txt)
        vs = batch_scores(model, tok, base + '{"verdict":"', VERDICTS)
        recs.append({"nid": o["nid"], "is_mal": int(o["label"]),
                     "verdict": int(vs[0] > vs[1]),
                     "verdict_score": round(vs[0] - vs[1], 4)})
        print(f"  {o['nid']}: verdict={recs[-1]['verdict']} score={recs[-1]['verdict_score']}")
    json.dump({"smoke": True, "n": len(recs), "records": recs}, open(OUT, "w"), indent=1)
    print(f"SMOKE PASS — saved {OUT}")
    print("Moi truong OK. Chay run_all.bat (Windows) hoac bash run_all.sh (Linux).")


if __name__ == "__main__":
    main()
