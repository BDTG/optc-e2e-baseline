"""data.py — Dataset loading and chat-template formatting for the fine-tuning pipeline."""

from __future__ import annotations

import os
from typing import Dict, List

from datasets import Dataset, load_dataset

from config import FinetuneConfig


# Each JSONL record must contain these keys.
REQUIRED_KEYS: List[str] = ["instruction", "output"]


def _validate_dataset_path(dataset_path: str) -> None:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Training file not found: '{dataset_path}'.\n"
            "Each line must be a JSON object with 'instruction' and 'output' keys."
        )


def _validate_dataset_schema(dataset: Dataset) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in dataset.column_names]
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {missing}.\n"
            f"Found columns: {dataset.column_names}."
        )


def _build_format_fn(tokenizer):
    """Return a batched map function that applies the tokenizer's chat template."""
    def format_batch(batch: Dict[str, List]) -> Dict[str, List[str]]:
        texts: List[str] = []
        for instruction, output in zip(batch["instruction"], batch["output"]):
            messages = [
                {"role": "user",  "content": instruction},
                {"role": "model", "content": output},
            ]
            # add_generation_prompt=False includes the completion text during training.
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}
    return format_batch


def load_and_format_dataset(
    dataset_path: str,
    tokenizer,
    config: FinetuneConfig,
) -> Dataset:
    """Load a JSONL file and produce a 'text' column formatted with the chat template."""
    print(f"\n[Data] Loading dataset from: {dataset_path}")
    _validate_dataset_path(dataset_path)

    dataset: Dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"[Data] Loaded {len(dataset):,} samples.")

    _validate_dataset_schema(dataset)

    print("[Data] Applying chat template...")
    dataset = dataset.map(_build_format_fn(tokenizer), batched=True, desc="Formatting")

    print("[Data] Sample preview:")
    print("-" * 60)
    print(dataset["text"][0])
    print("-" * 60)

    return dataset
