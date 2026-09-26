"""RQ1b v2 — LoRA train + SAVE adapter + eval 21 flagged, tat ca trong 1 chay (tranh train 2 lan).
seed param argv[1], default 42. Save: C:/Users/BDTG/AppData/Local/Temp/rq1b-adapter-<seed>"""
import json, sys, os, time, random, torch
sys.stdout.reconfigure(line_buffering=True)
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

SUB = r"F:\backup\OpTC-thesis\P1\Output\data\rq1a-subset.jsonl"
HOLD = r"F:\backup\OpTC-thesis\P1\Output\data\ttp_holdout.jsonl"
TTP_ENUM = ["T1003", "T1059", "T1053", "T1071", "T1218", "T1027", "T1562",
            "T1574", "T1490", "T1087", "T1082", "T1083", "none"]
VERDICTS = ["MALICIOUS", "BENIGN"]


def flat(ch):
    if isinstance(ch, str):
        try: ch = json.loads(ch.replace("'", '"'))
        except Exception: ch = []
    return " | ".join([(c.get("msg") or "") for c in (ch or [])][-5:])


def build_examples():
    recs = [json.loads(l) for l in open(SUB, encoding="utf-8")]
    hold = [json.loads(l) for l in open(HOLD, encoding="utf-8")]
    exs = []
    mal = [r for r in recs if r["label"] == 1]
    for _ in range(10):
        for r in mal:
            ch = flat(r.get("parent_chain", []))
            exs.append({"prompt": f"Chain: {ch[:600]}\nRespond JSON: {{verdict,technique_id}}\n",
                        "completion": '{"verdict":"MALICIOUS","technique_id":"T1059"}'})
    for r in hold:
        ch = flat(r.get("parent_chain", []))
        verdict = "MALICIOUS" if int(r["label"]) == 1 else "BENIGN"
        ttp = str(r.get("technique_id", "none"))
        ttp = ttp if any(t in ttp for t in TTP_ENUM) else "none"
        exs.append({"prompt": f"Chain: {ch[:600]}\nRespond JSON: {{verdict,technique_id}}\n",
                    "completion": f'{{"verdict":"{verdict}","technique_id":"{ttp}"}}'})
    return exs


class ExDS(Dataset):
    def __init__(self, exs, tok):
        self.x = []
        for e in exs:
            full = e["prompt"] + e["completion"]
            ids = tok(full, truncation=True, max_length=512, padding="max_length",
                      return_tensors="pt")["input_ids"][0]
            mask = (ids != tok.pad_token_id).long()
            self.x.append({"input_ids": ids, "attention_mask": mask, "labels": ids.clone()})

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


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
            out.append(sum(float(logp[row, j, c[j]]) for j in range(m)) / m)
        del logits, logp
    return out


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    random.seed(seed); torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32)
    lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                      lora_dropout=0.05, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    exs = build_examples(); random.shuffle(exs)
    ds = ExDS(exs[:400], tok)
    print(f"seed={seed} n_ex={len(ds)}")
    args = TrainingArguments(output_dir=f"C:/Users/BDTG/AppData/Local/Temp/rq1b-ck{seed}",
                             per_device_train_batch_size=1, num_train_epochs=1, learning_rate=2e-4,
                             logging_steps=50, save_strategy="no", report_to=[],
                             remove_unused_columns=False, seed=seed, disable_tqdm=True)
    tr = Trainer(model=model, args=args, train_dataset=ds)
    t0 = time.time(); tr.train()
    # SAVE
    ad = f"C:/Users/BDTG/AppData/Local/Temp/rq1b-adapter-{seed}"
    model.save_pretrained(ad)
    print(f"saved adapter {ad} ({time.time()-t0:.0f}s)")
    hist = json.load(open(args.output_dir + "/trainer_state.json", encoding="utf-8"))["log_history"] \
        if os.path.exists(args.output_dir + "/trainer_state.json") else []
    losses = [h["loss"] for h in hist if "loss" in h]
    # EVAL on 21 flagged
    model.eval()
    recs = [json.loads(l) for l in open(SUB, encoding="utf-8")]
    flag = [r for r in recs if isinstance(r.get("parent_chain"), list) and len(r["parent_chain"]) >= 2]
    TP = TN = FP = FN = 0; recs_out = []
    for r in flag:
        txt = flat(r["parent_chain"])[:700]
        p = f"You are a security analyst. Chain: {txt}\nRespond one JSON object.\n" + '{"verdict":"'
        vs = batch_scores(model, tok, p, VERDICTS)
        pred = 1 if vs[0] > vs[1] else 0
        lab = int(r["label"])
        TP += pred == 1 and lab == 1; TN += pred == 0 and lab == 0
        FP += pred == 1 and lab == 0; FN += pred == 0 and lab == 1
        recs_out.append({"nid": r["nid"], "label": lab, "pred": pred, "score": round(vs[0] - vs[1], 3)})
    summ = {"seed": seed, "train_loss_last": losses[-1] if losses else None,
            "train_time_s": round(time.time() - t0, 1), "adapter": ad,
            "eval_flagged": len(flag), "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "acc": round((TP + TN) / max(len(flag), 1), 4)}
    json.dump(res_dir := {"summary": summ, "recs": recs_out},
              open(r"F:\backup\OpTC-thesis\P1\Output\results_phase2\rq1b-lora-eval.json", "w",
                   encoding="utf-8"), indent=1)
    json.dump(summ, open(f"C:/Users/BDTG/AppData/Local/Temp/rq1b-seed{seed}.json", "w", encoding="utf-8"), indent=1)
    print("SUMMARY:", json.dumps(summ))
    print("RQ1B_DONE")


main()
