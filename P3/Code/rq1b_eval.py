"""RQ1b eval — LoRA seed 42 trên 21 dị nghi (12 TP + 9 FP): goal push 9 FP -> BENIGN, giu 12 TP.
Dùng custom loop generation thay Trainer (tránh heavy). Constrained scoring như RQ1a."""
import json, sys, torch, torch.nn.functional as F
sys.stdout.reconfigure(line_buffering=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

CKPT = "C:/Users/BDTG/AppData/Local/Temp/rq1b-ckpt-42"
SUB = r"F:\backup\OpTC-thesis\P1\Output\data\rq1a-subset.jsonl"


def flat(ch):
    if isinstance(ch, str):
        try: ch = json.loads(ch.replace("'", '"'))
        except Exception: ch = []
    return " | ".join([(c.get("msg") or "") for c in (ch or [])][-5:])


def batch_scores(model, tok, prefix, candidates, chunk=4):
    pre = tok(prefix, add_special_tokens=False)["input_ids"]
    out = []
    for ci in range(0, len(candidates), chunk):
        cands = [tok(c, add_special_tokens=False)["input_ids"] for c in candidates[ci:ci + chunk]]
        seqs = [pre + c for c in cands]
        L = max(len(s) for s in seqs); pad = tok.pad_token_id
        inp = torch.tensor([[pad] * (L - len(s)) + s for s in seqs])
        mask = torch.tensor([[1 if i - (L - len(s)) >= 0 else 0 for i in range(L)] for s in seqs])
        with torch.no_grad():
            logits = model(input_ids=inp, attention_mask=mask).logits[:, -L - 1:-1].float()
        logp = F.log_softmax(logits, dim=-1)
        for row, c in enumerate(cands):
            m = len(c)
            lp = sum(float(logp[row, j, c[j]]) for j in range(m)) / m
            out.append(lp)
        del logits, logp
    return out


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, CKPT, torch_dtype=torch.float32)
    model.eval()
    recs = [json.loads(l) for l in open(SUB, encoding="utf-8")]
    flag = []
    for r in recs:
        ch = r.get("parent_chain", [])
        if isinstance(ch, list) and len(ch) >= 2: flag.append(r)
    print(f"flagged={len(flag)}")
    VV = ["MALICIOUS", "BENIGN"]
    TP = TNc = FP = FN = 0; recs_out = []
    for r in flag:
        txt = flat(r.get("parent_chain", []))[:700]
        base_p = f"You are a security analyst. Chain: {txt}\nRespond one JSON: {{verdict...}}\n" + '{"verdict":"'
        vs = batch_scores(model, tok, base_p, VV)
        verdict = 1 if vs[0] > vs[1] else 0
        lab = int(r["label"])
        if verdict == 1 and lab == 1: TP += 1
        elif verdict == 0 and lab == 0: TNc += 1
        elif verdict == 1 and lab == 0: FP += 1
        else: FN += 1
        recs_out.append({"nid": r["nid"], "label": lab, "pred": verdict,
                         "score": round(vs[0] - vs[1], 3)})
        print(f"  {r['nid']} lab={lab} pred={verdict} s={round(vs[0]-vs[1],3)}")
    acc = (TP + TNc) / max(TP + TNc + FP + FN, 1)
    summ = {"n_flagged": len(flag), "TP": TP, "TN": TNc, "FP": FP, "FN": FN,
            "acc": round(acc, 4)}
    print("SUMMARY:", json.dumps(summ))
    json.dump({"summary": summ, "recs": recs_out},
              open(r"F:\backup\OpTC-thesis\P1\Output\results_phase2\rq1b-lora-eval.json", "w", encoding="utf-8"), indent=1)
    print("RQ1B_EVAL_DONE")


main()
