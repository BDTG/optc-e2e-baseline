"""
finetune.py — Entry-point for the QLoRA fine-tuning pipeline.

Usage:
    python finetune.py                                          # default config
    python finetune.py --model Qwen/Qwen2.5-1.5B-Instruct      # override model
    python finetune.py --config my_config.json                  # load from file
    python finetune.py --export-config my_config.json           # export defaults
"""

from __future__ import annotations

import argparse
import sys

from config import FinetuneConfig
from data import load_and_format_dataset
from model import apply_lora, build_bnb_config, build_lora_config, load_base_model, load_tokenizer
from trainer import build_sft_trainer, build_training_args, run_training
from utils import configure_stdout, get_compute_dtype, print_gpu_info, print_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning pipeline for causal language models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",          type=str, default=None, metavar="PATH",
                        help="Load a JSON config file.")
    parser.add_argument("--export-config",   type=str, default=None, metavar="PATH",
                        help="Export default config to JSON and exit.")
    parser.add_argument("--model",           type=str, default=None, metavar="MODEL_NAME",
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--dataset",         type=str, default=None, metavar="PATH",
                        help="Path to the JSONL training file.")
    parser.add_argument("--output-dir",      type=str, default=None, metavar="DIR",
                        help="Directory for checkpoints and TensorBoard logs.")
    parser.add_argument("--lora-output-dir", type=str, default=None, metavar="DIR",
                        help="Directory for the saved LoRA adapter.")
    parser.add_argument("--epochs",          type=int,   default=None, metavar="N",
                        help="Number of training epochs.")
    parser.add_argument("--lr",              type=float, default=None, metavar="FLOAT",
                        help="Peak learning rate.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> FinetuneConfig:
    """Build FinetuneConfig. Priority: CLI flags > JSON file > defaults."""
    config = FinetuneConfig.from_json(args.config) if args.config else FinetuneConfig()

    if args.model:           config.model_name = args.model
    if args.dataset:         config.dataset_path = args.dataset
    if args.output_dir:      config.output_dir = args.output_dir
    if args.lora_output_dir: config.lora_output_dir = args.lora_output_dir
    if args.epochs is not None: config.training.num_train_epochs = args.epochs
    if args.lr is not None:     config.training.learning_rate = args.lr

    return config


def main() -> None:
    configure_stdout()
    args = parse_args()

    if args.export_config:
        FinetuneConfig().to_json(args.export_config)
        print(f"[Config] Exported to: {args.export_config}")
        sys.exit(0)

    config = build_config(args)
    print_section("QLoRA Fine-Tuning Pipeline")
    print(config)

    print_gpu_info()

    print_section("Step 1 / 5 — Tokenizer")
    tokenizer = load_tokenizer(model_name=config.model_name)

    print_section("Step 2 / 5 — Dataset")
    dataset = load_and_format_dataset(
        dataset_path=config.dataset_path,
        tokenizer=tokenizer,
        config=config,
    )

    print_section("Step 3 / 5 — Base Model")
    bnb_config = build_bnb_config(
        compute_dtype=get_compute_dtype(),
        quant_cfg=config.quantization,
    )
    model = load_base_model(model_name=config.model_name, bnb_config=bnb_config)

    print_section("Step 4 / 5 — LoRA Adapters")
    lora_config = build_lora_config(lora_cfg=config.lora)
    model = apply_lora(model=model, lora_config=lora_config)

    print_section("Step 5 / 5 — Training")
    training_args = build_training_args(config=config)
    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=lora_config,
        training_args=training_args,
        config=config,
    )
    run_training(trainer=trainer, lora_output_dir=config.lora_output_dir, tokenizer=tokenizer)

    print_section("Pipeline Complete")


if __name__ == "__main__":
    main()
