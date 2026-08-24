"""
inference.py — Interactive multi-turn chat with a fine-tuned QLoRA model.

Usage:
    python inference.py                           # use saved LoRA adapter
    python inference.py --no-adapter              # run the base model only
    python inference.py --temperature 0.5         # override generation params

Chat commands:
    quit / exit   Exit the session.
    /clear        Reset conversation history.
    /params       Print current generation parameters.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from config import FinetuneConfig, GenerationConfig, QuantizationConfig
from model import build_bnb_config, load_tokenizer
from utils import configure_stdout, get_compute_dtype, print_section


def parse_args() -> argparse.Namespace:
    defaults = FinetuneConfig()
    gen = defaults.generation

    parser = argparse.ArgumentParser(
        description="Interactive chat with a fine-tuned QLoRA model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",              type=str,   default=defaults.model_name,
                        help="HuggingFace base model ID or local path.")
    parser.add_argument("--adapter",            type=str,   default=defaults.lora_output_dir,
                        help="Path to the LoRA adapter directory.")
    parser.add_argument("--no-adapter",         action="store_true",
                        help="Run the base model without any LoRA adapter.")
    parser.add_argument("--max-new-tokens",     type=int,   default=gen.max_new_tokens)
    parser.add_argument("--temperature",        type=float, default=gen.temperature)
    parser.add_argument("--top-p",              type=float, default=gen.top_p)
    parser.add_argument("--top-k",              type=int,   default=gen.top_k)
    parser.add_argument("--repetition-penalty", type=float, default=gen.repetition_penalty)
    return parser.parse_args()


def load_model_for_inference(
    base_model_name: str,
    adapter_dir: str,
    use_adapter: bool,
) -> PreTrainedModel:
    """Load the quantized base model and optionally attach a LoRA adapter."""
    bnb_config = build_bnb_config(
        compute_dtype=get_compute_dtype(),
        quant_cfg=QuantizationConfig(),
    )

    print(f"[Inference] Loading base model: {base_model_name}")
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if use_adapter:
        if not os.path.isdir(adapter_dir):
            print(f"[WARNING] Adapter directory not found: '{adapter_dir}'. Running base model only.")
        else:
            print(f"[Inference] Attaching LoRA adapter from: {adapter_dir}")
            model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()
    return model


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    conversation_history: List[Dict[str, str]],
    gen_config: GenerationConfig,
) -> str:
    """Generate one assistant reply for the current conversation history."""
    prompt: str = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True,
    )

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length: int = inputs.input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            top_k=gen_config.top_k,
            repetition_penalty=gen_config.repetition_penalty,
            do_sample=gen_config.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens, skipping the input prompt.
    new_tokens = output_ids[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    configure_stdout()
    args = parse_args()

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
    )

    print_section("Inference — QLoRA Fine-Tuned Model")

    # Prefer loading the tokenizer from the adapter dir because it may contain
    # a custom chat template saved during fine-tuning.
    tokenizer_source = (
        args.adapter
        if (not args.no_adapter and os.path.isdir(args.adapter))
        else args.model
    )
    tokenizer = load_tokenizer(model_name=tokenizer_source)

    model = load_model_for_inference(
        base_model_name=args.model,
        adapter_dir=args.adapter,
        use_adapter=not args.no_adapter,
    )

    print_section("Chat Session  —  type 'quit' to exit")
    print("  Commands: quit | exit | /clear | /params")

    conversation_history: List[Dict[str, str]] = []

    while True:
        try:
            user_input: str = input("\nYou: ")
        except (KeyboardInterrupt, EOFError):
            print("\n[Inference] Session ended.")
            break

        stripped = user_input.strip()

        if stripped.lower() in ("quit", "exit"):
            print("[Inference] Exiting.")
            break

        if stripped == "/clear":
            conversation_history.clear()
            print("[Inference] Conversation history cleared.")
            continue

        if stripped == "/params":
            print(
                f"  max_new_tokens    : {gen_config.max_new_tokens}\n"
                f"  temperature       : {gen_config.temperature}\n"
                f"  top_p             : {gen_config.top_p}\n"
                f"  repetition_penalty: {gen_config.repetition_penalty}\n"
                f"  do_sample         : {gen_config.do_sample}"
            )
            continue

        if not stripped:
            continue

        conversation_history.append({"role": "user", "content": stripped})

        try:
            reply = generate_response(
                model=model,
                tokenizer=tokenizer,
                conversation_history=conversation_history,
                gen_config=gen_config,
            )
        except Exception as exc:
            print(f"[ERROR] Generation failed: {exc}")
            conversation_history.pop()
            continue

        conversation_history.append({"role": "model", "content": reply})
        print(f"\nModel: {reply}")


if __name__ == "__main__":
    main()
