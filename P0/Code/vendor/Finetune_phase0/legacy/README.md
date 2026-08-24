# LLM Fine-Tuning with QLoRA (legacy — text generation)

Fine-tune any causal language model using QLoRA (4-bit quantization + LoRA adapters).
The pipeline is designed for consumer GPUs with 8-24 GB VRAM and is tested with Gemma and Qwen models.

> This is the original generation-mode template, kept separate from the OpTC anomaly-detection
> classification pipeline in [`../optc/`](../optc/README.md). Run every command below from
> inside this `legacy/` directory — relative output paths (`./gemma-finetuned`, etc.) resolve
> against the current working directory, not the script location.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Preparing the Dataset](#preparing-the-dataset)
5. [Configuration](#configuration)
6. [Running Fine-Tuning](#running-fine-tuning)
7. [Running Inference](#running-inference)
8. [CLI Reference](#cli-reference)
9. [Troubleshooting](#troubleshooting)

---

## Requirements

- Python 3.10 or later
- NVIDIA GPU with at least 8 GB VRAM (CUDA 11.8 or 12.1)
- CUDA-enabled PyTorch (see installation below)

CPU-only mode is supported but extremely slow and not recommended for training.

---

## Installation

**Step 1 — Install PyTorch with CUDA support.**

PyTorch must be installed separately before the other packages so that the correct CUDA build is selected.

For CUDA 12.1:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Step 2 — Install the remaining dependencies.**

```
pip install -r ../requirements.txt
```

**Step 3 — Authenticate with HuggingFace (required for gated models such as Gemma).**

```
huggingface-cli login
```

Paste your Access Token from https://huggingface.co/settings/tokens.
You must also accept the model's license agreement on the model's HuggingFace page before downloading.

---

## Project Structure

```
Finetune/
    requirements.txt    Python package dependencies (shared with optc/).
    legacy/
        config.py        All hyperparameters declared as typed dataclasses.
        utils.py         Shared helpers: GPU info, dtype resolution, chat template.
        data.py          Dataset loading, validation, and chat-template formatting.
        model.py         Tokenizer loading, 4-bit quantization, LoRA adapter injection.
        trainer.py       TrainingArguments, SFTTrainer construction, training execution.
        finetune.py      Entry-point: CLI parsing and pipeline orchestration.
        inference.py     Interactive multi-turn chat with the fine-tuned model.
```

Note: `legacy/config.py`, `data.py`, `model.py`, `utils.py` are separate from the
same-named files in `../optc/` — the two pipelines don't import from each other.

---

## Preparing the Dataset

Create a JSONL file where each line is a JSON object with two keys:

- `instruction` — the user prompt or question
- `output` — the expected model response

Example record:
```json
{"instruction": "What is gradient descent?", "output": "Gradient descent is an optimization algorithm..."}
```

Save this file as `dataset_prepared.jsonl` in the `legacy/` directory, or pass a custom path with `--dataset`.

---

## Configuration

All hyperparameters are declared in `config.py` as dataclasses. The most important fields are:

**Paths**

| Field | Default | Description |
|---|---|---|
| `model_name` | `google/gemma-4-E2B` | HuggingFace model ID or local path |
| `dataset_path` | `dataset_prepared.jsonl` | Path to the JSONL training file |
| `output_dir` | `./gemma-finetuned` | Directory for checkpoints and TensorBoard logs |
| `lora_output_dir` | `./gemma-lora-adapter` | Directory where the final adapter is saved |

**Training (`TrainingConfig`)**

| Field | Default | Description |
|---|---|---|
| `max_seq_length` | `512` | Maximum token length per sample. Increase for longer text, costs more VRAM. |
| `per_device_train_batch_size` | `1` | Samples per GPU per step. Set to 1 for GPUs with 8-12 GB VRAM. |
| `gradient_accumulation_steps` | `4` | Effective batch size = batch_size x this value. |
| `num_train_epochs` | `3` | Number of full passes over the dataset. |
| `learning_rate` | `2e-4` | Peak learning rate. |
| `save_strategy` | `"no"` | `"no"` saves only at the end; `"epoch"` saves after every epoch. |

**LoRA (`LoRAConfig`)**

| Field | Default | Description |
|---|---|---|
| `r` | `8` | LoRA rank. Higher values = more parameters = better capacity, more VRAM. |
| `lora_alpha` | `16` | LoRA scaling factor. Effective LoRA learning rate ≈ lora_alpha / r. |
| `lora_dropout` | `0.05` | Dropout on LoRA layers. |
| `target_modules` | all projection layers | Which linear layers to apply LoRA to. |

**Configuration file**

You can export the defaults to JSON, edit them, and reload:
```
python finetune.py --export-config my_config.json
# edit my_config.json as needed
python finetune.py --config my_config.json
```

---

## Running Fine-Tuning

Run with all defaults (model, dataset, and hyperparameters from `config.py`):
```
python finetune.py
```

Override common settings from the command line without editing any file:
```
python finetune.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 5 --lr 1e-4
```

Use a JSONL dataset at a custom path:
```
python finetune.py --dataset /path/to/my_data.jsonl
```

The pipeline prints a step-by-step log. When training completes, the LoRA adapter and tokenizer are saved to `lora_output_dir` (default `./gemma-lora-adapter`).

Monitor training with TensorBoard:
```
tensorboard --logdir ./gemma-finetuned/logs
```

---

## Running Inference

Start an interactive chat session using the saved adapter:
```
python inference.py
```

Compare the fine-tuned model against the base model without the adapter:
```
python inference.py --no-adapter
```

Override generation parameters at runtime:
```
python inference.py --temperature 0.3 --max-new-tokens 512
```

**Chat commands available during the session:**

| Command | Effect |
|---|---|
| `quit` or `exit` | End the session |
| `/clear` | Reset conversation history (start a new context) |
| `/params` | Print the current generation parameters |

The session maintains the full conversation history so the model has multi-turn context.

---

## CLI Reference

### finetune.py

```
python finetune.py [OPTIONS]

  --config PATH             Load hyperparameters from a JSON file.
  --export-config PATH      Export default config to JSON and exit.
  --model MODEL_NAME        HuggingFace model ID or local directory.
  --dataset PATH            Path to the JSONL training file.
  --output-dir DIR          Directory for checkpoints and TensorBoard logs.
  --lora-output-dir DIR     Directory for the saved LoRA adapter.
  --epochs N                Number of training epochs.
  --lr FLOAT                Peak learning rate.
```

### inference.py

```
python inference.py [OPTIONS]

  --model MODEL_NAME        HuggingFace base model ID or local directory.
  --adapter DIR             Path to the LoRA adapter directory.
  --no-adapter              Run the base model without any adapter.
  --max-new-tokens N        Maximum tokens to generate per reply (default: 256).
  --temperature FLOAT       Sampling temperature (default: 0.7).
  --top-p FLOAT             Nucleus sampling threshold (default: 0.9).
  --repetition-penalty      Repetition penalty (default: 1.1).
```

---

## Troubleshooting

**"Failed to load the base model" / authentication error**

Run `huggingface-cli login` and paste a valid token. For gated models (Gemma, LLaMA),
you must also accept the license on the model's HuggingFace page.

**Out-of-memory (CUDA OOM) error during training**

Reduce `max_seq_length` (e.g., from 512 to 256) or keep `per_device_train_batch_size` at 1.
Both settings are in `TrainingConfig` inside `config.py`.

**CUDA is not available**

Confirm that `torch.cuda.is_available()` returns `True`. If not, reinstall PyTorch with
the correct CUDA index URL for your driver version (see Installation above).

**Tokenizer has no chat template**

The pipeline automatically applies a Gemma-compatible fallback template when the tokenizer
does not include one. If your model uses a different format (e.g., ChatML), update
`GEMMA_CHAT_TEMPLATE` in `utils.py` to match the expected format.

**Training loss does not decrease**

- Verify that the dataset is correctly formatted (instruction/output pairs).
- Try increasing `lora_alpha` (e.g., from 16 to 32) or increasing `r` (e.g., from 8 to 16).
- Lower the learning rate (e.g., from 2e-4 to 1e-4).
- Check the sample preview printed at startup to confirm the chat template is applied correctly.
