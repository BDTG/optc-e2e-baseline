"""model.py — Tokenizer loading, 4-bit quantization, base model loading, and LoRA setup."""

from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from config import LoRAConfig, QuantizationConfig
from utils import ensure_chat_template


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
) -> PreTrainedTokenizerBase:
    """Load tokenizer, set pad_token if missing, apply fallback chat template if needed."""
    print(f"\n[Model] Loading tokenizer: {model_name}")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )

    # Decoder-only models often have no pad token; EOS is the standard substitute.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  [INFO] pad_token was None — set to eos_token.")

    ensure_chat_template(tokenizer)

    print(f"  Vocabulary size  : {tokenizer.vocab_size:,}")
    print(f"  Max model length : {tokenizer.model_max_length:,}")
    return tokenizer


def build_bnb_config(
    compute_dtype: torch.dtype,
    quant_cfg: QuantizationConfig,
) -> BitsAndBytesConfig:
    """Build a BitsAndBytesConfig for 4-bit NF4 quantization (QLoRA)."""
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.load_in_4bit,
        bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
    )


def load_base_model(
    model_name: str,
    bnb_config: BitsAndBytesConfig,
    trust_remote_code: bool = True,
) -> PreTrainedModel:
    """
    Load a causal LM with 4-bit quantization and prepare it for k-bit training.

    Raises RuntimeError with actionable steps if the download fails
    (missing HuggingFace auth, unaccepted license, wrong model ID).
    """
    print(f"\n[Model] Loading base model in 4-bit: {model_name}")

    try:
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        raise RuntimeError(
            "\n[ERROR] Failed to load the base model. Common causes:\n\n"
            "  1. Not authenticated with HuggingFace (required for gated models).\n"
            "     Run:  huggingface-cli login\n\n"
            "  2. Model license not accepted on the HuggingFace model page.\n\n"
            "  3. Incorrect model ID.\n"
            f"     Model ID used: '{model_name}'\n\n"
            f"  Original error: {exc}"
        ) from exc

    # prepare_model_for_kbit_training upcasts normalization layers to float32
    # and enables gradient checkpointing for memory efficiency.
    model = prepare_model_for_kbit_training(model)

    # KV-cache is incompatible with gradient checkpointing.
    model.config.use_cache = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters : {total_params / 1e6:.1f} M")
    return model


def build_lora_config(lora_cfg: LoRAConfig) -> LoraConfig:
    """Translate LoRAConfig dataclass into a PEFT LoraConfig object."""
    return LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
        # Advanced options
        modules_to_save=lora_cfg.modules_to_save,
        use_rslora=lora_cfg.use_rslora,
        use_dora=lora_cfg.use_dora,
        init_lora_weights=lora_cfg.init_lora_weights,
        rank_pattern=lora_cfg.rank_pattern or {},
        alpha_pattern=lora_cfg.alpha_pattern or {},
        layers_to_transform=lora_cfg.layers_to_transform,
        layers_pattern=lora_cfg.layers_pattern,
    )


def apply_lora(model: PreTrainedModel, lora_config: LoraConfig) -> PeftModel:
    """Inject LoRA adapters into the model. Only adapter weights will be trained."""
    print("\n[Model] Applying LoRA adapters...")
    peft_model: PeftModel = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model
