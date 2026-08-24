"""config.py — Hyperparameters & paths cho pipeline SLM phan loai bat thuong endpoint.

Khac ban goc (sinh text): task_type=SEQ_CLS, them num_labels/split_frac/quant,
bo phan GenerationConfig/SFT/NEFTune (khong dung cho phan loai).
He config nay giu de tien QUET nhieu cau hinh cho RQ2 (Pareto phan cung).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # q/k/v/o_proj (attention) + gate/up/down_proj (MLP) — LoRA chi tren attention
    # khong du capacity de ghi de tri thuc pretrain cho cac ky thuat tan cong khong co
    # dau hieu tu vung ro rang (vd loi dung netstat/plink) khi chi co vai tram mau train
    # (da kiem chung: SLM bo lo 4/18 dong malicious ma TF-IDF/ModernBERT (full fine-tune)
    # van bat duoc). Them MLP de tang capacity thich nghi context (khong chi attention).
    target_modules: Union[List[str], str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )
    bias: str = "none"
    task_type: str = "SEQ_CLS"            # <-- phan loai, khong phai CAUSAL_LM
    # classification head phai duoc train day du + luu cung LoRA
    modules_to_save: Optional[List[str]] = field(default_factory=lambda: ["score"])
    use_rslora: bool = False
    use_dora: bool = False
    init_lora_weights: Union[bool, str] = True


@dataclass
class QuantizationConfig:
    """quant='none'|'int8'|'int4' — bien QUET chinh cho RQ2 Pareto."""
    quant: str = "int4"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingConfig:
    max_seq_length: int = 1024           # ngan sach token; cay dai bi cat -> du lieu cho luan diem on-device
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 3
    max_steps: int = -1
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    optim: str = "paged_adamw_8bit"
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    gradient_checkpointing: bool = True
    tf32: Optional[bool] = None
    logging_steps: int = 20
    save_strategy: str = "no"
    report_to: str = "none"
    seed: int = 42
    # phan loai: xu ly mat can bang (endpoint ~3-8% doc)
    use_class_weights: bool = True


@dataclass
class DataConfig:
    dataset_path: str = "train_data.jsonl"   # JSONL tu parser/: {text,label,meta:{host,ts,size,...}}
    split_frac: float = 0.6                  # chia CAUSAL theo thoi gian: train=qua khu, test=tuong lai
    split_mode: str = "causal"               # 'causal' | 'host_holdout' (cho RQ5)
    holdout_hosts: Optional[List[str]] = None  # cho RQ5: cac host chi dung o test
    neg_pos_ratio: Optional[float] = None    # neu dat: undersample benign trong TRAIN ve
                                              # ty le nay (vd 20.0 = 20 benign : 1 malicious).
                                              # CHI anh huong train, test giu nguyen 100%.
                                              # Muc dich: cat so step training khi mat can
                                              # bang cuc doan (malicious <1% mau).
    seed: int = 42                           # seed cho undersample benign (deterministic
                                              # theo seed). Doi de chay nhieu seed -> bao
                                              # cao mean+-std thay vi 1 con so.


@dataclass
class SLMConfig:
    model_name: str = "Qwen/Qwen2.5-3B"
    num_labels: int = 2
    out_dir: str = "./slm-ckpt"
    scores_path: str = "scores.jsonl"        # diem rui ro moi mau test -> tinh lai metric bat ky luc nao

    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_json(cls, json_path: str) -> "SLMConfig":
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {json_path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            model_name=data.get("model_name", cls.__dataclass_fields__["model_name"].default),
            num_labels=data.get("num_labels", 2),
            out_dir=data.get("out_dir", "./slm-ckpt"),
            scores_path=data.get("scores_path", "scores.jsonl"),
            lora=LoRAConfig(**data.get("lora", {})),
            quantization=QuantizationConfig(**data.get("quantization", {})),
            training=TrainingConfig(**data.get("training", {})),
            data=DataConfig(**data.get("data", {})),
        )

    def to_json(self, json_path: str) -> None:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def __str__(self) -> str:
        return (f"SLMConfig:\n"
                f"  model      : {self.model_name}\n"
                f"  quant      : {self.quantization.quant}\n"
                f"  num_labels : {self.num_labels}\n"
                f"  max_seq_len: {self.training.max_seq_length}\n"
                f"  epochs     : {self.training.num_train_epochs}\n"
                f"  lr         : {self.training.learning_rate}\n"
                f"  lora.r     : {self.lora.r}\n"
                f"  split      : {self.data.split_mode} @ {self.data.split_frac}\n"
                f"  data       : {self.data.dataset_path}")
