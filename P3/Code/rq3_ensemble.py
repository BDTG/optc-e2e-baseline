"""RQ3 — ensemble 3x0.5B (adapters seed 42/43/44) majority-vote trên 21 flagged.
Giả sử rq1b4-adapter-42 đã có; nếu thiếu seed nào thì train bằng lora4 logic inline (đơn giản: báo thiếu).
Mỗi adapter: generation MALICIOUS/BENIGN đúng prompt train. Vote >=2/3 → MALICIOUS."""
import json, sys, random, torch
sys.stdout.reconfigure(line_buffering=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SUB = r"F:\backup\OpTC-thesis\P1\Output\data\rq1a-subset.jsonl"
PROMPT = "Chain: {chain}\nRespond JSON: {{verdict,technique_id}}\n"
ADAPTERS = [f"C:/Users/BDTG/AppData/Local/Temp/rq1b4-adapter-{s}" for s in (42, 43, 44)]


def flat(ch):
    if isinstance(ch, str):
        try: ch = json.loads(ch.replace("'", '"'))
        except Exception: ch = []
    return " | ".join([(c.get("msg") or "") for c in (ch or [])][-5:])


def main():
    import os
    missing = [a for a in ADAPTERS if not os.path.isdir(a)]
    if missing:
        print("MISSING adapters:", missing)
        print("HINT: run rq1b_lora4.py <seed> trước cho seed 43 va 44")
        sys.exit(1)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32)
    recs = [json.loads(l) for l in open(SUB, encoding="utf-8")]
    flag = [r for r in recs if isinstance(r.get("parent_chain"), list) and len(r["parent_chain"]) >= 2]
    outs = {}  # nid -> list of preds per seed
    for a in ADAPTERS:
        model = PeftModel.from_pretrained(base, a, torch_dtype=torch.float32).eval()
        preds = []
        for r in flag:
            ids = tok(PROMPT.format(chain=flat(r["parent_chain"])[:600]), return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=25, do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            preds.append(1 if "MALICIOUS" in gen[:40] else 0)
        outs[a] = preds
        del model
        print(f"done {a}")
    TP = TN = FP = FN = 0; recs_out = []
    for i, r in enumerate(flag):
        v = [sum(1 for a in ADAPTERS if outs[a][i] == 1) >= 2 for _ in [0]]  # majority mal
        pred = 1 if sum(outs[a][i] for a in ADAPTERS) >= 2 else 0
        lab = int(r["label"])
        TP += pred == 1 and lab == 1; TN += pred == 0 and lab == 0
        FP += pred == 1 and lab == 0; FN += pred == 0 and lab == 1
        recs_out.append({"nid": r["nid"], "label": lab, "pred": pred,
                         "votes": sum(outs[a][i] for a in ADAPTERS)})
    n = max(TP + TN + FP + FN, 1)
    summ = {"ensemble": "3x0.5B LoRA seeds 42/43/44 majority-vote",
            "eval_flagged": len(flag), "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "acc": round((TP + TN) / n, 4),
            "precision": round(TP / max(TP + FP, 1), 4), "recall": round(TP / max(TP + FN, 1), 4),
            "F1": round(2 * (TP / max(TP + FP, 1)) * (TP / max(TP + FN, 1)) /
                        max((TP / max(TP + FP, 1)) + (TP / max(TP + FN, 1)), 1e-9), 4)}
    json.dump({"summary": summ, "recs": recs_out},
              open(r"F:\backup\OpTC-thesis\P1\Output\results_phase2\rq3-ensemble.json", "w", encoding="utf-8"), indent=1)
    print("SUMMARY:", json.dumps(summ))
    print("RQ3_DONE")


main()
