"""config.py — All hyperparameters and file paths for the fine-tuning pipeline.

Parameter sources:
  - TrainingConfig  -> transformers.TrainingArguments
  - LoRAConfig      -> peft.LoraConfig
  - QuantizationConfig -> bitsandbytes.BitsAndBytesConfig
  - SFTConfig       -> trl.SFTConfig (SFTTrainer arguments)
  - GenerationConfig -> transformers.GenerationConfig
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class LoRAConfig:
    """Maps to peft.LoraConfig. Ref: https://huggingface.co/docs/peft/package_reference/lora"""

    # Rank of the update matrices. Higher rank = more parameters = more VRAM.
    r: int = 8
    # Scaling factor. Effective LoRA scale = lora_alpha / r (or / sqrt(r) with use_rslora).
    lora_alpha: int = 16
    # Dropout applied to the LoRA input; 0.0 disables it.
    lora_dropout: float = 0.05
    # Modules to inject LoRA into. Use "all-linear" to target every linear layer (QLoRA style).
    target_modules: Union[List[str], str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    # Which biases to train: "none" | "all" | "lora_only".
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # --- Advanced LoRA options ---

    # Modules to fully unfreeze and save alongside LoRA weights (e.g. classification heads).
    modules_to_save: Optional[List[str]] = None
    # Rank-Stabilized LoRA: sets scale = lora_alpha / sqrt(r) for better stability at high ranks.
    use_rslora: bool = False
    # Weight-Decomposed LoRA (DoRA): decomposes weights into magnitude + direction components.
    use_dora: bool = False
    # LoRA weight initialization: True (default, Microsoft), "gaussian", "pissa", "olora", "loftq".
    init_lora_weights: Union[bool, str] = True
    # Per-layer rank overrides: {"layer_name_regex": rank}. Layers not listed use `r`.
    rank_pattern: Dict[str, int] = field(default_factory=dict)
    # Per-layer alpha overrides: {"layer_name_regex": alpha}. Layers not listed use `lora_alpha`.
    alpha_pattern: Dict[str, int] = field(default_factory=dict)
    # Restrict LoRA to specific layer indices (e.g. [0, 1, 2] applies to first 3 layers only).
    layers_to_transform: Optional[Union[List[int], int]] = None
    # Name of the nn.ModuleList attribute; required when layers_to_transform is set.
    layers_pattern: Optional[Union[List[str], str]] = None


@dataclass
class QuantizationConfig:
    """Maps to bitsandbytes.BitsAndBytesConfig. Ref: https://huggingface.co/docs/bitsandbytes"""

    load_in_4bit: bool = True
    # "nf4" (NormalFloat4) is recommended for QLoRA. Alternative: "fp4".
    bnb_4bit_quant_type: str = "nf4"
    # Double quantization (quantizing the quantization constants) saves ~0.4 bits/param.
    bnb_4bit_use_double_quant: bool = True
    # compute_dtype is resolved at runtime via utils.get_compute_dtype().
    # Set load_in_8bit=True instead of load_in_4bit to use LLM.int8() quantization.
    load_in_8bit: bool = False


@dataclass
class TrainingConfig:
    """Maps to transformers.TrainingArguments. Ref: https://huggingface.co/docs/transformers/main_classes/trainer"""

    # --- Sequence / batch size ---
    # Maximum tokens per sample (prompt + completion). Raise for longer pairs; costs VRAM.
    max_seq_length: int = 512
    # Samples per GPU per step. Use 1 for GPUs with <=12 GB VRAM.
    per_device_train_batch_size: int = 1
    # Accumulate gradients over N steps before updating weights.
    # Effective batch = per_device_train_batch_size * num_devices * gradient_accumulation_steps.
    gradient_accumulation_steps: int = 4

    # --- Duration ---
    num_train_epochs: int = 3
    # Hard limit on total steps; overrides num_train_epochs when set to a positive integer.
    max_steps: int = -1

    # --- Learning rate & scheduler ---
    learning_rate: float = 2e-4
    # Scheduler type: "linear" | "cosine" | "cosine_with_restarts" | "polynomial" |
    # "constant" | "constant_with_warmup" | "inverse_sqrt" | "reduce_lr_on_plateau".
    lr_scheduler_type: str = "cosine"
    # Fraction of total steps used for linear warmup from 0 to learning_rate.
    warmup_ratio: float = 0.03
    # Exact number of warmup steps (takes precedence over warmup_ratio when > 0).
    warmup_steps: int = 0

    # --- Optimizer ---
    # "paged_adamw_8bit" keeps optimizer states in CPU-paged memory; reduces VRAM.
    # Other options: "adamw_torch", "adamw_8bit", "adafactor", "sgd", "adamw_torch_fused".
    optim: str = "paged_adamw_8bit"
    # Adam beta1: exponential decay rate for first moment estimates (momentum).
    adam_beta1: float = 0.9
    # Adam beta2: exponential decay rate for second moment estimates (variance).
    adam_beta2: float = 0.999
    # Epsilon for numerical stability in Adam denominator.
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.001

    # --- Gradient ---
    # Clip gradients whose global norm exceeds this value. Set 0 to disable.
    max_grad_norm: float = 0.3
    # Enable gradient checkpointing: recomputes activations on backward pass to save VRAM
    # at the cost of ~20% slower training.
    gradient_checkpointing: bool = False

    # --- Mixed precision ---
    # bf16 / fp16 are resolved automatically from GPU capabilities in trainer.py.
    # Set tf32=True to use TF32 (19-bit mantissa) matrix multiplication on Ampere+ GPUs.
    tf32: Optional[bool] = None

    # --- Regularization ---
    # Label smoothing [0.0, 0.1]: replaces hard 0/1 targets with soft values to reduce overconfidence.
    label_smoothing_factor: float = 0.0

    # --- NEFTune ---
    # Add uniform noise to embeddings during training to improve instruction following.
    # Paper recommends values in [5.0, 15.0]. Set None to disable.
    neftune_noise_alpha: Optional[float] = None

    # --- Logging & checkpointing ---
    logging_steps: int = 1
    # "no" = save only at end | "epoch" | "steps".
    save_strategy: str = "no"
    # Maximum number of checkpoints to keep. Older ones are deleted automatically.
    save_total_limit: Optional[int] = None
    report_to: str = "tensorboard"

    # --- SFT-specific (trl.SFTTrainer / SFTConfig) ---
    # Pack multiple short samples into one sequence to maximize GPU utilization.
    # Disable when sample lengths vary widely (risk of mixing contexts across samples).
    packing: bool = False
    # Number of parallel processes for dataset.map() preprocessing.
    dataset_num_proc: Optional[int] = None

    # --- Reproducibility ---
    seed: int = 42

    # --- torch.compile (PyTorch 2.0+) ---
    # Compile the model for ~20-50% speedup. Requires PyTorch >= 2.0.
    torch_compile: bool = False


@dataclass
class GenerationConfig:
    """Parameters for model.generate() used during inference."""

    max_new_tokens: int = 256
    # Sampling temperature: lower = more deterministic, higher = more random.
    temperature: float = 0.7
    # Nucleus sampling: keeps the smallest set of tokens whose cumulative probability >= top_p.
    top_p: float = 0.9
    # Top-k sampling: restricts sampling to the k most likely next tokens. 0 disables it.
    top_k: int = 50
    # Penalizes tokens that have already appeared. Values > 1.0 reduce repetition.
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass
class FinetuneConfig:
    """Master config that bundles all sub-configs and path settings."""

    # HuggingFace model ID or local path. Gated models require `huggingface-cli login`.
    model_name: str = "google/gemma-4-E2B"
    # JSONL file where each line has "instruction" and "output" keys.
    dataset_path: str = "dataset_prepared.jsonl"
    output_dir: str = "./gemma-finetuned"
    lora_output_dir: str = "./gemma-lora-adapter"

    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @classmethod
    def from_json(cls, json_path: str) -> "FinetuneConfig":
        """Load config from a JSON file. Missing keys fall back to defaults."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")
        with path.open("r", encoding="utf-8") as f:
            data: dict = json.load(f)
        return cls(
            model_name=data.get("model_name", cls.__dataclass_fields__["model_name"].default),
            dataset_path=data.get("dataset_path", cls.__dataclass_fields__["dataset_path"].default),
            output_dir=data.get("output_dir", cls.__dataclass_fields__["output_dir"].default),
            lora_output_dir=data.get("lora_output_dir", cls.__dataclass_fields__["lora_output_dir"].default),
            lora=LoRAConfig(**data.get("lora", {})),
            quantization=QuantizationConfig(**data.get("quantization", {})),
            training=TrainingConfig(**data.get("training", {})),
            generation=GenerationConfig(**data.get("generation", {})),
        )

    def to_json(self, json_path: str) -> None:
        """Save the current config to a JSON file."""
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    def __str__(self) -> str:
        lines = ["FinetuneConfig:"]
        lines.append(f"  model_name           : {self.model_name}")
        lines.append(f"  dataset_path         : {self.dataset_path}")
        lines.append(f"  output_dir           : {self.output_dir}")
        lines.append(f"  lora_output_dir      : {self.lora_output_dir}")
        lines.append(f"  training.epochs      : {self.training.num_train_epochs}")
        lines.append(f"  training.max_steps   : {self.training.max_steps}")
        lines.append(f"  training.lr          : {self.training.learning_rate}")
        lines.append(f"  training.scheduler   : {self.training.lr_scheduler_type}")
        lines.append(f"  training.optim       : {self.training.optim}")
        lines.append(f"  training.grad_ckpt   : {self.training.gradient_checkpointing}")
        lines.append(f"  training.neftune     : {self.training.neftune_noise_alpha}")
        lines.append(f"  lora.r               : {self.lora.r}")
        lines.append(f"  lora.alpha           : {self.lora.lora_alpha}")
        lines.append(f"  lora.use_rslora      : {self.lora.use_rslora}")
        lines.append(f"  lora.use_dora        : {self.lora.use_dora}")
        lines.append(f"  lora.init_weights    : {self.lora.init_lora_weights}")
        return "\n".join(lines)
