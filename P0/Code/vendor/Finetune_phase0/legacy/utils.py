"""utils.py — Shared helpers for GPU diagnostics, dtype resolution, and chat templates."""

from __future__ import annotations

import sys

import torch


def configure_stdout() -> None:
    """Set stdout to UTF-8. Call once at the start of main(), not at import time."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def print_section(title: str, width: int = 60) -> None:
    bar = "=" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def get_compute_dtype() -> torch.dtype:
    """Return the best float dtype for the current GPU (bfloat16, float16, or float32)."""
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def print_gpu_info() -> None:
    """Print a CUDA device summary. Warns if no GPU is found."""
    print_section("GPU / CUDA Diagnostic")

    if not torch.cuda.is_available():
        print("  [WARNING] CUDA is NOT available.")
        print("  Training on CPU will be extremely slow.")
        print("  Install a CUDA-enabled PyTorch build:")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return

    device_count = torch.cuda.device_count()
    print(f"  CUDA version   : {torch.version.cuda}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  Device count   : {device_count}")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_vram_gb = props.total_memory / 1_073_741_824
        print(f"\n  --- Device {i} ---")
        print(f"  Name           : {props.name}")
        print(f"  Total VRAM     : {total_vram_gb:.2f} GiB")
        print(f"  Compute cap.   : {props.major}.{props.minor}")
        print(f"  Compute dtype  : {get_compute_dtype()}")
        print(f"  bfloat16 OK    : {torch.cuda.is_bf16_supported()}")


# Gemma-compatible Jinja2 chat template used when the tokenizer has no built-in one.
GEMMA_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<start_of_turn>' + message['role'] + '\\n' + message['content'] + '<end_of_turn>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}"
)


def ensure_chat_template(tokenizer) -> None:
    """Apply the Gemma fallback chat template if the tokenizer has none."""
    if tokenizer.chat_template is None:
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE
        print("  [INFO] No built-in chat template found. Applied Gemma-compatible fallback.")
