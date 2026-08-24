"""utils.py — Helper GPU/dtype/seed/JSONL/class-weight. Giu tu ban goc, bo phan
chat-template (khong dung cho phan loai)."""
from __future__ import annotations

import random
import sys
from typing import List, Optional

import numpy as np
import torch
from transformers import Trainer

# re-export tu run_utils (thuan thu vien chuan) de cac script can torch/transformers
# van goi duoc qua "from utils import ..." nhu truoc, khong doi cach dung.
from run_utils import file_sha256, make_run_dir, read_jsonl, save_run_metrics  # noqa: F401


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def print_section(title: str, width: int = 60) -> None:
    bar = "=" * width
    print(f"\n{bar}\n  {title}\n{bar}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_compute_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def print_gpu_info() -> None:
    print_section("GPU / CUDA Diagnostic")
    if not torch.cuda.is_available():
        print("  [WARNING] CUDA khong san sang. Train tren CPU se rat cham.")
        return
    n = torch.cuda.device_count()
    print(f"  CUDA {torch.version.cuda} | PyTorch {torch.__version__} | {n} device(s)")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  VRAM={p.total_memory/1_073_741_824:.1f} GiB  "
              f"cc={p.major}.{p.minor}  dtype={get_compute_dtype()}")


def compute_class_weights(train_rows: List[dict]) -> torch.Tensor:
    """[1.0, n_neg/n_pos] cho cross-entropy co trong so — endpoint mat can bang cuc doan
    (malicious thuong <5% mau)."""
    n_pos = sum(r["label"] for r in train_rows)
    n_neg = len(train_rows) - n_pos
    return torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float)


class WeightedTrainer(Trainer):
    """transformers.Trainer voi cross-entropy co trong so lop, thay vi lay trung binh
    khong trong so (se bi lop malicious hiem ap dao boi lop benign da so)."""

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            out.logits, labels, weight=self.class_weights.to(out.logits.device))
        return (loss, out) if return_outputs else loss
