"""baseline_encoder.py — Fine-tune FULL (khong LoRA) encoder ~150M lam classifier.

Day la doi thu THAT cua H0 (khong phai TF-IDF). Neu encoder nay da ngang SLM 8B,
ke hoach phai reframe truoc khi dau tu them tuan nao.

Khong dung LoRA/quant nhu model.py (target_modules q/k/v/o_proj la ten lop cua
kien truc decoder Qwen, khong khop BERT-style encoder). 150M full fine-tune
vua du VRAM tren GPU tieu dung, khong can PEFT.

Chay (causal split):
  python baseline_encoder.py --model answerdotai/ModernBERT-base --data "E:\\dataset\\processed\\train_data_201_refined.jsonl" --split-frac 0.7

Chay (host-holdout — train host nay, test host khac chua thay):
  python baseline_encoder.py --model answerdotai/ModernBERT-base --data "E:\\dataset\\processed\\train_data_frozen_v2.jsonl" ^
      --split-mode host_holdout --holdout-hosts SysClient0501.systemia.com
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from config import DataConfig
from data import load_split
from eval import score_rows
from metrics import all_metrics, print_metrics
from utils import (compute_class_weights, file_sha256, get_compute_dtype,
                    make_run_dir, print_gpu_info, save_run_metrics, set_seed,
                    WeightedTrainer)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-frac", type=float, default=0.6)
    ap.add_argument("--split-mode", choices=["causal", "host_holdout"], default="causal")
    ap.add_argument("--holdout-hosts", nargs="*", default=None,
                     help="danh sach host chi dung o test (split-mode=host_holdout)")
    ap.add_argument("--neg-ratio", type=float, default=None,
                     help="undersample benign trong TRAIN ve <ratio>:1 so voi malicious "
                          "(vd 20 = 20 benign/1 malicious). Test khong doi. Dung de so "
                          "sanh cong bang voi train.py khi train.py cung dung neg-ratio.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42,
                     help="seed dung chung cho undersample + training. Doi de chay nhieu "
                          "seed cung 1 cau hinh -> bao cao mean+-std thay vi 1 con so.")
    args = ap.parse_args()

    set_seed(args.seed)
    print_gpu_info()

    holdout_tag = "-".join(h.split(".")[0] for h in (args.holdout_hosts or []))
    run_dir = make_run_dir("baseline_encoder", model=args.model.split("/")[-1],
                            split=args.split_mode, holdout=holdout_tag, seed=args.seed)
    print(f"[run] artifact -> {run_dir}")
    dataset_hash = file_sha256(args.data)
    print(f"[data] sha256({args.data}) = {dataset_hash}")

    cfg = DataConfig(dataset_path=args.data, split_frac=args.split_frac,
                      split_mode=args.split_mode, holdout_hosts=args.holdout_hosts,
                      neg_pos_ratio=args.neg_ratio, seed=args.seed)
    train_rows, test_rows = load_split(cfg)

    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               TrainingArguments, DataCollatorWithPadding)
    from datasets import Dataset

    print(f"[model] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,} (~{n_params/1e6:.0f}M)")

    def tok_fn(b):
        return tok(b["text"], truncation=True, max_length=args.max_seq_length)

    ds_tr = Dataset.from_list(
        [{"text": r["text"], "label": int(r["label"])} for r in train_rows]
    ).map(tok_fn, batched=True)

    w = compute_class_weights(train_rows)
    print(f"[train] class weights = {w.tolist()}")

    use_bf16 = get_compute_dtype() == torch.bfloat16
    targs = TrainingArguments(
        output_dir=str(run_dir / "checkpoint"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.01,
        logging_steps=50, save_strategy="no", report_to=[], seed=args.seed,
        bf16=use_bf16, fp16=(not use_bf16 and torch.cuda.is_available()))

    trainer = WeightedTrainer(model=model, args=targs, train_dataset=ds_tr,
                               data_collator=DataCollatorWithPadding(tok),
                               class_weights=w)
    print("\n===== TRAIN =====")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    train_sec = time.time() - t0
    gpu_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    scores, labels, infer_ms = score_rows(model, tok, test_rows, args.max_seq_length)

    m = all_metrics(labels, scores)
    print(f"\n===== BASELINE ENCODER: {args.model} (~{n_params/1e6:.0f}M, full fine-tune) =====")
    print_metrics(m)
    print(f"  latency/mau   : {infer_ms:.2f} ms")
    print(f"  train time    : {train_sec:.0f}s")
    print(f"  peak mem      : {gpu_gb:.2f} GB")
    print(f"  test pos/neg  : {sum(labels)}/{len(labels)-sum(labels)}")

    with open(run_dir / "scores.jsonl", "w", encoding="utf-8") as f:
        for r, sc in zip(test_rows, scores):
            f.write(json.dumps({"score": sc, "label": int(r["label"]),
                                "host": r["meta"].get("host")}) + "\n")
    save_run_metrics(run_dir, model=args.model, n_params=n_params,
                      data=args.data, dataset_sha256=dataset_hash,
                      split_mode=args.split_mode,
                      holdout_hosts=args.holdout_hosts, split_frac=args.split_frac,
                      neg_pos_ratio=args.neg_ratio, seed=args.seed,
                      epochs=args.epochs, max_seq_length=args.max_seq_length,
                      train_sec=round(train_sec), latency_ms=round(infer_ms, 3),
                      peak_mem_gb=round(gpu_gb, 3),
                      test_pos=sum(labels), test_neg=len(labels) - sum(labels),
                      **m)
    print(f"[run] scores + metrics.json -> {run_dir}")


if __name__ == "__main__":
    main()
