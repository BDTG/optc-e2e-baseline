"""trainer.py — SFTConfig, SFTTrainer construction, and training execution."""

from __future__ import annotations

import os

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

from config import FinetuneConfig
from utils import get_compute_dtype


def build_training_args(config: FinetuneConfig) -> SFTConfig:
    """Build SFTConfig from config. bf16/fp16 is resolved automatically from GPU."""
    compute_dtype = get_compute_dtype()
    use_bf16 = (compute_dtype == torch.bfloat16) and torch.cuda.is_available()
    use_fp16 = (compute_dtype == torch.float16) and torch.cuda.is_available()

    tc = config.training

    kwargs = {
        "output_dir": config.output_dir,
        "logging_dir": os.path.join(config.output_dir, "logs"),
        "logging_steps": tc.logging_steps,
        "report_to": tc.report_to,

        # Duration
        "num_train_epochs": tc.num_train_epochs,
        "max_steps": tc.max_steps,

        # Batch size
        "per_device_train_batch_size": tc.per_device_train_batch_size,
        "gradient_accumulation_steps": tc.gradient_accumulation_steps,

        # Learning rate & scheduler
        "learning_rate": tc.learning_rate,
        "lr_scheduler_type": tc.lr_scheduler_type,

        # Optimizer
        "optim": tc.optim,
        "adam_beta1": tc.adam_beta1,
        "adam_beta2": tc.adam_beta2,
        "adam_epsilon": tc.adam_epsilon,
        "weight_decay": tc.weight_decay,

        # Gradient
        "max_grad_norm": tc.max_grad_norm,
        "gradient_checkpointing": tc.gradient_checkpointing,

        # Mixed precision
        "fp16": use_fp16,
        "bf16": use_bf16,
        "tf32": tc.tf32,

        # Regularization
        "label_smoothing_factor": tc.label_smoothing_factor,

        # NEFTune noise embedding (improves instruction following; paper: 5.0–15.0).
        "neftune_noise_alpha": tc.neftune_noise_alpha,

        # Checkpointing
        "save_strategy": tc.save_strategy,
        "save_total_limit": tc.save_total_limit,
        "eval_strategy": "no",

        # SFT / Dataset settings
        "dataset_text_field": "text",
        "max_length": tc.max_seq_length,
        "packing": tc.packing,
        "dataset_num_proc": tc.dataset_num_proc,

        # Misc
        "seed": tc.seed,
        "torch_compile": tc.torch_compile,
        "dataloader_pin_memory": True,
    }

    if tc.warmup_steps > 0:
        kwargs["warmup_steps"] = tc.warmup_steps
    elif tc.warmup_ratio is not None:
        kwargs["warmup_ratio"] = tc.warmup_ratio

    return SFTConfig(**kwargs)


def build_sft_trainer(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    peft_config: LoraConfig,
    training_args: SFTConfig,
    config: FinetuneConfig,
) -> SFTTrainer:
    """Build SFTTrainer. Expects a 'text' column produced by data.load_and_format_dataset."""
    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )


def run_training(
    trainer: SFTTrainer,
    lora_output_dir: str,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Run training, then save the LoRA adapter and tokenizer to lora_output_dir."""
    print("\n[Trainer] Starting training run...")
    trainer.train()

    print(f"\n[Trainer] Training complete. Saving adapter to: {lora_output_dir}")
    trainer.model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)
    print("[Trainer] Saved. Run 'python inference.py' to test the model.")
