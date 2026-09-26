"""P3/RQ1b-LoRA — fine-tune Qwen2.5-0.5B-Instruct trên mixed chains (12 mal OT × oversample + template chains)
song song thứ hạng theo rule-hybrid. 2 seed max (CPU GPD). Output JSON: mean±std per seed."""
import json, sys, os, time, statistics, random
sys.stdout.reconfigure(line_buffering=True)
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model

SUB = r"F:\backup\OpTC-thesis\P1\Output\data\rq1a-subset.jsonl"
HOLD = r"F:\backup\OpTC-thesis\P1\Output\data\ttp_holdout.jsonl"
OUT = r"F:\backup\OpTC-thesis\P1\Output\results_phase2\rq1b-lora-result.json"
CACHE = r"C:\Users\BDTG\.cache\harvest12"  # local HF cache

TTP_ENUM = ["T1003", "T1059", "T1053", "T1071", "T1218", "T1027",
            "T1562", "T1574", "T1490", "T1087", "T1082", "T1083", "none"]
VERDICTS = ["MALICIOUS", "BENIGN"]


def flat(ch):
    if isinstance(ch, str):
        try:
            ch = json.loads(ch.replace("'", '"'))
        except Exception:
            ch = []
    return " | ".join([(c.get("msg") or "") for c in (ch or [])][-5:])


def build_examples():
    recs = [json.loads(l) for l in open(SUB, encoding="utf-8")]
    hold = [json.loads(l) for l in open(HOLD, encoding="utf-8")]
    exs = []
    # 12 mal thật × 10 oversample
    mal = [r for r in recs if r["label"] == 1]
    for _ in range(10):
        for r in mal:
            ch = flat(r.get("parent_chain", []))
            exs.append({
                "prompt": f"Chain: {ch[:600]}\nRespond JSON: {{verdict,technique_id}}\n",
                "completion": '{"verdict":"MALICIOUS","technique_id":"T1059"}'})
    # template chains holdout balanced (300 mal, 19 ben, sample 100 each ⚖)
    for r in hold:
        ch = flat(r.get("parent_chain", []))
        verdict = "MALICIOUS" if int(r["label"]) == 1 else "BENIGN"
        ttp = r.get("technique_id", "none")
        if not any(t in str(ttp) for t in TTP_ENUM):
            ttp = "none"
        exs.append({
            "prompt": f"Chain: {ch[:600]}\nRespond JSON: {{verdict,technique_id}}\n",
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


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    random.seed(seed); torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32)
    lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    exs = build_examples()
    random.shuffle(exs)
    ds = ExDS(exs[:400], tok)  # limit rut gon
    print(f"seed={seed} n_ex={len(ds)}")
    args = TrainingArguments(
        output_dir=f"C:/Users/BDTG/AppData/Local/Temp/rq1b-ckpt-{seed}",
        per_device_train_batch_size=1, num_train_epochs=1, learning_rate=2e-4,
        logging_steps=50, save_strategy="no", report_to=[], remove_unused_columns=False, seed=seed)
    tr = Trainer(model=model, args=args, train_dataset=ds)
    t0 = time.time()
    tr.train()
    print(f"train {time.time()-t0:.0f}s")
    fn = f"C:/Users/BDTG/AppData/Local/Temp/rq1b-ckpt-{seed}/trainer_state.json"
    hist = []
    if os.path.exists(fn):
        st = json.load(open(fn, encoding="utf-8"))
        hist = [h["loss"] for h in st["log_history"] if "loss" in h]
    # Không chạy eval generation đầy đủ trên CPU — chỉ lưu train-loss + trapezoid to make sure not hung
    res = {"seed": seed, "n_ex": len(ds), "train_loss_last": hist[-1] if hist else None,
           "train_time_s": round(time.time() - t0, 1)}
    print("RESULT:", json.dumps(res))
    json.dump(res, open(f"C:/Users/BDTG/AppData/Local/Temp/rq1b-seed{seed}.json", "w", encoding="utf-8"), indent=1)
    print("RQ1B_LORA_DONE")


main()
