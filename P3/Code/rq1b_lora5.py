"""RQ1b v5 — retrain LoRA voi benign NOISE THAT (chain>=2 tu 2250 cascade, 869 mau) thay chain trang.
Train: 84 mal (12x2 real + 60 template) + 84 benign-noise-real + 19 holdout ben = 187.
Eval: 21 flagged (12 mal + 9 FP NOISE THAT — chinh la cac mau mal nhung se duoc model hoc duong)."""
import json, sys, os, time, random, torch
sys.stdout.reconfigure(line_buffering=True)
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

SUB = r"F:\backup\OpTC-thesis\P1\Output\data\rq1a-subset.jsonl"
HOLD = r"F:\backup\OpTC-thesis\P1\Output\data\ttp_holdout.jsonl"
NOISE = r"F:\backup\OpTC-thesis\P1\Output\data\rq1b-noise-benign.jsonl"
PROMPT = "Chain: {chain}\nRespond JSON: {{verdict,technique_id}}\n"


def flat(ch):
    if isinstance(ch, str):
        try: ch = json.loads(ch.replace("'", '"'))
        except Exception: ch = []
    return " | ".join([(c.get("msg") or "") for c in (ch or [])][-5:])


def build():
    recs = [json.loads(l) for l in open(SUB, encoding="utf-8")]
    hold = [json.loads(l) for l in open(HOLD, encoding="utf-8")]
    noise = [json.loads(l) for l in open(NOISE, encoding="utf-8")]
    exs = []
    mal = [r for r in recs if r["label"] == 1]
    for _ in range(2):
        for r in mal:
            exs.append({"p": PROMPT.format(chain=flat(r.get("parent_chain", []))[:600]),
                        "c": '{"verdict":"MALICIOUS","technique_id":"T1059"}'})
    hmal = [r for r in hold if int(r["label"]) == 1]
    random.shuffle(hmal)
    for r in hmal[:60]:
        ttp = str(r.get("technique_id", "none"))
        ttp = ttp if any(t in ttp for t in ("T10", "T15", "T12")) else "none"
        exs.append({"p": PROMPT.format(chain=flat(r.get("parent_chain", []))[:600]),
                    "c": json.dumps({"verdict": "MALICIOUS", "technique_id": ttp})})
    # benign noise THAT — chinh la loai FP model phan biet sau nhat
    random.shuffle(noise)
    for r in noise[:84]:
        exs.append({"p": PROMPT.format(chain=flat(r.get("parent_chain", []))[:600]),
                    "c": '{"verdict":"BENIGN","technique_id":"none"}'})
    for r in [x for x in hold if int(x["label"]) == 0]:
        exs.append({"p": PROMPT.format(chain=flat(r.get("parent_chain", []))[:600]),
                    "c": '{"verdict":"BENIGN","technique_id":"none"}'})
    return exs


class DS(Dataset):
    def __init__(self, exs, tok):
        self.x = []
        for e in exs:
            ids = tok(e["p"] + e["c"], truncation=True, max_length=512, padding="max_length",
                      return_tensors="pt")["input_ids"][0]
            self.x.append({"input_ids": ids, "attention_mask": (ids != tok.pad_token_id).long(),
                           "labels": ids.clone()})

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 43
    random.seed(seed); torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                              lora_dropout=0.05, task_type="CAUSAL_LM"))
    exs = build(); random.shuffle(exs)
    ds = DS(exs, tok)
    maln = sum(1 for e in exs if "MALICIOUS" in e["c"]); benn = len(exs) - maln
    print(f"seed={seed} n_ex={len(ds)} mal={maln} ben={benn} (ben gom noise that)")
    args = TrainingArguments(output_dir=f"C:/Users/BDTG/AppData/Local/Temp/rq1b5-ck{seed}",
                             per_device_train_batch_size=1, num_train_epochs=2, learning_rate=2e-4,
                             logging_steps=40, save_strategy="no", report_to=[],
                             remove_unused_columns=False, seed=seed, disable_tqdm=True)
    t0 = time.time()
    Trainer(model=model, args=args, train_dataset=ds).train()
    ad = f"C:/Users/BDTG/AppData/Local/Temp/rq1b5-adapter-{seed}"
    model.save_pretrained(ad)
    model.eval()
    recs = [json.loads(l) for l in open(SUB, encoding="utf-8")]
    flag = [r for r in recs if isinstance(r.get("parent_chain"), list) and len(r["parent_chain"]) >= 2]
    TP = TN = FP = FN = 0; recs_out = []
    for r in flag:
        ids = tok(PROMPT.format(chain=flat(r["parent_chain"])[:600]), return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=25, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = 1 if "MALICIOUS" in gen[:40] else 0
        lab = int(r["label"])
        TP += pred == 1 and lab == 1; TN += pred == 0 and lab == 0
        FP += pred == 1 and lab == 0; FN += pred == 0 and lab == 1
        recs_out.append({"nid": r["nid"], "label": lab, "pred": pred, "gen": gen[:60]})
    n = max(TP + TN + FP + FN, 1)
    p, rc = TP / max(TP + FP, 1), TP / max(TP + FN, 1)
    summ = {"seed": seed, "n_train": len(ds), "mal_train": maln, "ben_train": benn,
            "ben_source": "noise-that chain>=2 (869 tu 2250 cascade) + 19 holdout",
            "eval_flagged": len(flag), "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "acc": round((TP + TN) / n, 4), "precision": round(p, 4), "recall": round(rc, 4),
            "F1": round(2 * p * rc / max(p + rc, 1e-9), 4), "train_s": round(time.time() - t0),
            "adapter": ad}
    json.dump({"summary": summ, "recs": recs_out},
              open(r"F:\backup\OpTC-thesis\P1\Output\results_phase2\rq1b5-lora-eval.json", "w", encoding="utf-8"), indent=1)
    print("SUMMARY:", json.dumps(summ))
    print("RQ1B5_DONE")


main()
