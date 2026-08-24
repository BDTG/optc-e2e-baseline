"""train.py — Entry-point: fine-tune SLM phan loai + danh gia + do chi phi.

Chay:
  python train.py                                  # config mac dinh
  python train.py --model Qwen/Qwen2.5-1.5B --quant int8   # override cho RQ2
  python train.py --config cfg.json                # tu file
  python train.py --export-config cfg.json         # xuat config mac dinh roi thoat

Cho RQ2 (Pareto): quet --model x --quant, moi lan ghi 1 dong vao results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from config import SLMConfig
from data import load_split
from eval import score_rows
from model import load_tokenizer, load_classifier
from metrics import all_metrics, print_metrics
from utils import (compute_class_weights, configure_stdout, file_sha256,
                    get_compute_dtype, make_run_dir, print_gpu_info, print_section,
                    save_run_metrics, set_seed, WeightedTrainer)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--export-config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--quant", choices=["none", "int8", "int4"], default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument("--split-frac", type=float, default=None)
    ap.add_argument("--split-mode", choices=["causal", "host_holdout"], default=None)
    ap.add_argument("--holdout-hosts", nargs="*", default=None,
                    help="danh sach host chi dung o test (split-mode=host_holdout)")
    ap.add_argument("--neg-ratio", type=float, default=None,
                    help="undersample benign trong TRAIN ve <ratio>:1 so voi malicious "
                         "(vd 20 = 20 benign/1 malicious). Test khong doi. Cat manh so "
                         "step khi mat can bang cuc doan.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="per_device_train_batch_size. Tang neu con nhieu VRAM du "
                         "(xem peak_mem o lan chay truoc) de giam step va tang throughput.")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="gradient_accumulation_steps. Effective batch = batch_size * "
                         "grad_accum — giu gia tri nay khi tang batch_size de effective "
                         "batch khong doi (dong dynamics hoc giu nguyen).")
    ap.add_argument("--grad-checkpointing", type=lambda s: s.lower() != "false",
                    default=None, metavar="true|false",
                    help="Tat (false) neu con nhieu VRAM du de tang toc (khong phai "
                         "recompute activations trong backward), doi lay VRAM cao hon.")
    ap.add_argument("--lora-r", type=int, default=None,
                    help="LoRA rank. Tang capacity, gan nhu khong ton VRAM them, nhung "
                         "tang risk overfit voi tap train nho — nen tang --lora-alpha "
                         "cung ty le (mac dinh alpha/r=2.0) de giu toc do hoc on dinh.")
    ap.add_argument("--lora-alpha", type=int, default=None,
                    help="LoRA alpha — scale hieu dung = alpha/r. Neu tang --lora-r ma "
                         "khong tang alpha tuong ung, moi buoc cap nhat trong so se yeu di.")
    ap.add_argument("--scores-path", default=None)
    ap.add_argument("--results-path", default="results.jsonl",
                    help="append 1 dong ket qua moi lan chay (cho RQ2)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed dung chung cho undersample + training. Doi de chay nhieu "
                         "seed cung 1 cau hinh -> bao cao mean+-std thay vi 1 con so.")
    return ap.parse_args()


def build_config(args) -> SLMConfig:
    cfg = SLMConfig.from_json(args.config) if args.config else SLMConfig()
    if args.model:          cfg.model_name = args.model
    if args.quant:          cfg.quantization.quant = args.quant
    if args.data:           cfg.data.dataset_path = args.data
    if args.epochs is not None:         cfg.training.num_train_epochs = args.epochs
    if args.max_seq_length is not None: cfg.training.max_seq_length = args.max_seq_length
    if args.split_frac is not None:     cfg.data.split_frac = args.split_frac
    if args.split_mode:                 cfg.data.split_mode = args.split_mode
    if args.holdout_hosts is not None:  cfg.data.holdout_hosts = args.holdout_hosts
    if args.neg_ratio is not None:      cfg.data.neg_pos_ratio = args.neg_ratio
    if args.batch_size is not None:     cfg.training.per_device_train_batch_size = args.batch_size
    if args.grad_accum is not None:     cfg.training.gradient_accumulation_steps = args.grad_accum
    if args.grad_checkpointing is not None: cfg.training.gradient_checkpointing = args.grad_checkpointing
    if args.lora_r is not None:         cfg.lora.r = args.lora_r
    if args.lora_alpha is not None:     cfg.lora.lora_alpha = args.lora_alpha
    if args.scores_path:    cfg.scores_path = args.scores_path
    if args.seed is not None:
        cfg.training.seed = args.seed
        cfg.data.seed = args.seed
    return cfg


def main():
    configure_stdout()
    args = parse_args()

    if args.export_config:
        SLMConfig().to_json(args.export_config)
        print(f"[config] da xuat mac dinh -> {args.export_config}")
        return

    cfg = build_config(args)
    set_seed(cfg.training.seed)
    print_gpu_info()

    holdout_tag = "-".join(h.split(".")[0] for h in (cfg.data.holdout_hosts or []))
    run_dir = make_run_dir("train", model=cfg.model_name.split("/")[-1],
                            quant=cfg.quantization.quant, split=cfg.data.split_mode,
                            holdout=holdout_tag, seed=cfg.training.seed)
    print(f"[run] artifact -> {run_dir}")
    if not args.scores_path:   # neu nguoi dung khong tu chi duong dan rieng
        cfg.scores_path = str(run_dir / "scores.jsonl")
    cfg.out_dir = str(run_dir / "checkpoint")

    print_section("CONFIG")
    print(cfg)
    dataset_hash = file_sha256(cfg.data.dataset_path)
    print(f"[data] sha256({cfg.data.dataset_path}) = {dataset_hash}")

    # ---- du lieu ----
    train_rows, test_rows = load_split(cfg.data)

    from transformers import (TrainingArguments, Trainer, DataCollatorWithPadding)
    from datasets import Dataset

    tok = load_tokenizer(cfg.model_name)
    model = load_classifier(cfg, tok)

    def tok_fn(b):
        return tok(b["text"], truncation=True, max_length=cfg.training.max_seq_length)

    ds_tr = Dataset.from_list(
        [{"text": r["text"], "label": int(r["label"])} for r in train_rows]
    ).map(tok_fn, batched=True)

    # ---- class weights cho mat can bang ----
    trainer_cls = Trainer
    trainer_kwargs = {}
    if cfg.training.use_class_weights:
        w = compute_class_weights(train_rows)
        print(f"[train] class weights = {w.tolist()}")
        trainer_cls = WeightedTrainer
        trainer_kwargs["class_weights"] = w

    tc = cfg.training
    use_bf16 = get_compute_dtype() == torch.bfloat16
    targs = TrainingArguments(
        output_dir=cfg.out_dir, num_train_epochs=tc.num_train_epochs,
        max_steps=tc.max_steps, per_device_train_batch_size=tc.per_device_train_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=tc.learning_rate, lr_scheduler_type=tc.lr_scheduler_type,
        warmup_ratio=tc.warmup_ratio, optim=tc.optim, weight_decay=tc.weight_decay,
        max_grad_norm=tc.max_grad_norm, gradient_checkpointing=tc.gradient_checkpointing,
        logging_steps=tc.logging_steps, save_strategy=tc.save_strategy,
        report_to=[], seed=tc.seed,
        bf16=use_bf16, fp16=(not use_bf16 and torch.cuda.is_available()))

    trainer = trainer_cls(model=model, args=targs, train_dataset=ds_tr,
                          data_collator=DataCollatorWithPadding(tok), **trainer_kwargs)
    print_section("TRAIN")
    t_train0 = time.time()
    trainer.train()
    train_sec = time.time() - t_train0

    # ---- luu LoRA adapter + tokenizer sau khi train ----
    # Voi model quantized (int4/int8) + device_map="auto", Trainer co the khong luu duoc.
    # Dung PEFT model.save_pretrained() de chi luu adapter (nhe, ~MB) thay vi full model.
    os.makedirs(cfg.out_dir, exist_ok=True)
    model.save_pretrained(cfg.out_dir)   # chi luu LoRA weights + modules_to_save (score head)
    tok.save_pretrained(cfg.out_dir)
    print(f"[out] LoRA adapter + tokenizer -> {cfg.out_dir}")

    # ---- inference: diem rui ro lien tuc ----
    print_section("INFERENCE + METRICS")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    scores, labels, infer_ms = score_rows(model, tok, test_rows, cfg.training.max_seq_length)

    m = all_metrics(labels, scores)
    print(f"model={cfg.model_name} quant={cfg.quantization.quant}")
    print_metrics(m)
    print(f"  latency/mau   : {infer_ms:.1f} ms  (tren thiet bi hien tai)")
    gpu_gb = torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else 0.0
    print(f"  peak mem      : {gpu_gb:.2f} GB")
    print(f"  train time    : {train_sec:.0f}s")

    # ---- luu diem rui ro (tinh lai metric bat ky luc nao) ----
    with open(cfg.scores_path, "w") as f:
        for r, sc in zip(test_rows, scores):
            f.write(json.dumps({"score": sc, "label": int(r["label"]),
                                "host": r["meta"].get("host"),
                                "size": r["meta"].get("size")}) + "\n")
    print(f"[out] diem rui ro -> {cfg.scores_path}")

    # ---- append 1 dong ket qua cho RQ2 ----
    row = {"model": cfg.model_name, "quant": cfg.quantization.quant,
           "max_seq_len": cfg.training.max_seq_length,
           "latency_ms": round(infer_ms, 2), "peak_mem_gb": round(gpu_gb, 2),
           "train_sec": round(train_sec), **{k: round(v, 4) for k, v in m.items()}}
    with open(args.results_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[out] ket qua -> {args.results_path} (append, bang quet cho RQ2)")

    save_run_metrics(run_dir, model=cfg.model_name, quant=cfg.quantization.quant,
                      data=cfg.data.dataset_path, dataset_sha256=dataset_hash,
                      split_mode=cfg.data.split_mode,
                      holdout_hosts=cfg.data.holdout_hosts, split_frac=cfg.data.split_frac,
                      neg_pos_ratio=cfg.data.neg_pos_ratio, seed=cfg.training.seed,
                      epochs=cfg.training.num_train_epochs,
                      max_seq_length=cfg.training.max_seq_length,
                      train_sec=round(train_sec), latency_ms=round(infer_ms, 3),
                      peak_mem_gb=round(gpu_gb, 3), **m)
    print(f"[run] checkpoint + scores + metrics.json -> {run_dir}")
    print("[note] latency o day do tren GPU/thiet bi hien tai. RQ2 can do lai tren CPU "
          "bang measure_cpu.py.")


if __name__ == "__main__":
    main()
