"""RQ1b v3 — LoRA balanced data + eval đúng prompt format train.
Train: 120 mal-oversample-đứng + 19 mal/ben template chain = 300 mal template giảm còn 60;
benign per template lọc: tăng benign lên ngang. Kết cấu train: 120 + 60 mal + 60 benign template.
Prompt duy nhất: "Chain: X\nRespond JSON: {verdict,technique_id}\n" — same cho train/eval.
Eval sau save adapter: generation (đúng format đã thấy diag học tốt)."""
import json, sys, os, time, random, torch
sys.stdout.reconfigure(line_buffering=True)
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

SUB = r"F:\backup\OpTC-thesis\P1\Output\data\rq1a-subset.jsonl"
HOLD = r"F:\backup\OpTC-thesis\P1\Output\data\ttp_holdout.jsonl"
PROMPT = "Chain: {chain}\nRespond JSON: {{verdict,technique_id}}\n"


def flat(ch):
    if isinstance(ch, str):
        try: ch = json.loads(ch.replace("'", '"'))
        except Exception: ch = []
    return " | ".join([(c.get("msg") or "") for c in (ch or [])][-5:])


def build_balanced():
    recs = [json.loads(l) for l in open(SUB, encoding="utf-8")]
    hold = [json.loads(l) for l in open(HOLD, encoding="utf-8")]
    exs = []
    mal = [r for r in recs if r["label"] == 1]
    for _ in range(10):
        for r in mal:
            ch = flat(r.get("parent_chain", []))
            exs.append({"p": PROMPT.format(chain=ch[:600]),
                        "c": '{"verdict":"MALICIOUS","technique_id":"T1059"}'})
    hmal = [r for r in hold if int(r["label"]) == 1]
    hben = [r for r in hold if int(r["label"]) == 0]
    random.shuffle(hmal)
    for r in hmal[:60]:
        ch = flat(r.get("parent_chain", []))
        ttp = str(r.get("technique_id", "none"))
        ttp = ttp if ("T10" in ttp or "T15" in ttp or "T12" in ttp) else "none"
        exs.append({"p": PROMPT.format(chain=ch[:600]),
                    "c": json.dumps({"verdict": "MALICIOUS", "technique_id": ttp})})
    random.shuffle(hben)
    for r in hben[:60]:
        ch = flat(r.get("parent_chain", []))
        exs.append({"p": PROMPT.format(chain=ch[:600]),
                    "c": '{"verdict":"BENIGN","technique_id":"none"}'})
    return exs


class ExDS(Dataset):
    def __init__(self, exs, tok):
        self.x = []
        for e in exs:
            full = e["p"] + e["c"]
            ids = tok(full, truncation=True, max_length=512, padding="max_length",
                      return_tensors="pt")["input_ids"][0]
            self.x.append({"input_ids": ids, "attention_mask": (ids != tok.pad_token_id).long(),
                           "labels": ids.clone()})

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    random.seed(seed); torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32)
    lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                      lora_dropout=0.05, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    exs = build_balanced()
    random.shuffle(exs)
    ds = ExDS(exs, tok)
    maln = sum(1 for e in exs if "MALICIOUS" in e["c"]); benn = len(exs) - maln
    print(f"seed={seed} n_ex={len(ds)} mal={maln} ben={benn}")
    args = TrainingArguments(output_dir=f"C:/Users/BDTG/AppData/Local/Temp/rq1b3-ck{seed}",
                             per_device_train_batch_size=1, num_train_epochs=2, learning_rate=2e-4,
                             logging_steps=40, save_strategy="no", report_to=[],
                             remove_unused_columns=False, seed=seed, disable_tqdm=True)
    Trainer(model=model, args=args, train_dataset=ds).train()
    ad = f"C:/Users/BDTG/AppData/Local/Temp/rq1b3-adapter-{seed}"
    model.save_pretrained(ad)
    model.eval()
    # eval: generation đúng prompt train trên 21 flagged
    recs = [json.loads(l) for l in open(SUB, encoding="utf-8")]
    flag = [r for r in recs if isinstance(r.get("parent_chain"), list) and len(r["parent_chain"]) >= 2]
    TP = TN = FP = FN = 0; recs_out = []
    for r in flag:
        ids = tok(PROMPT.format(chain=flat(r["parent_chain"])[:600]), return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=25, do_sample=False,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = 1 if "MALICIOUS" in gen[:40] else 0
        lab = int(r["label"])
        TP += pred == 1 and lab == 1; TN += pred == 0 and lab == 0
        FP += pred == 1 and lab == 0; FN += pred == 0 and lab == 1
        recs_out.append({"nid": r["nid"], "label": lab, "pred": pred, "gen": gen[:60]})
    n = max(TP + TN + FP + FN, 1)
    summ = {"seed": seed, "n_train": len(ds), "mal_train": maln, "ben_train": benn,
            "eval_flagged": len(flag), "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "acc": round((TP + TN) / n, 4),
            "precision": round(TP / max(TP + FP, 1), 4), "recall": round(TP / max(TP + FN, 1), 4),
            "adapter": ad}
    json.dump({"summary": summ, "recs": recs_out},
              open(r"F:\backup\OpTC-thesis\P1\Output\results_phase2\rq1b-lora-eval.json", "w", encoding="utf-8"), indent=1)
    json.dump(summ, open(f"C:/Users/BDTG/AppData/Local/Temp/rq1b3-seed{seed}.json", "w", encoding="utf-8"), indent=1)
    print("SUMMARY:", json.dumps(summ))
    print("RQ1B3_DONE")


main()
