"""model.py — Tai SLM lam CLASSIFIER (khong phai sinh text).

Khac ban goc: AutoModelForSequenceClassification thay AutoModelForCausalLM.
Ho tro quant none/int8/int4 (bien QUET cho RQ2). Giu xu ly loi tai model.
"""
from __future__ import annotations

import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from config import SLMConfig
from utils import get_compute_dtype

# Token mask IP/username do parser/utils.py::mask_ips()/mask_usernames() sinh ra luc
# export du lieu — PHAI KHOP CHINH XAC voi chuoi that trong text. Doi 1 ben phai doi ca 2.
MASK_TOKENS = ["<IP_LOCAL>", "<IP_PRIVATE>", "<IP_PUBLIC>", "<USER>"]


def load_tokenizer(model_name: str):
    print(f"[model] tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        print("  [info] pad_token=None -> dat = eos_token")
    n_added = tok.add_special_tokens({"additional_special_tokens": MASK_TOKENS})
    print(f"  [info] them {n_added} token mask IP/username vao tu vung: {MASK_TOKENS}")
    return tok


def _bnb_config(quant: str, qcfg, compute_dtype):
    if quant == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=qcfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=qcfg.bnb_4bit_use_double_quant)
    if quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_classifier(cfg: SLMConfig, tokenizer):
    quant = cfg.quantization.quant
    compute_dtype = get_compute_dtype()
    print(f"[model] classifier: {cfg.model_name}  quant={quant}")
    bnb = _bnb_config(quant, cfg.quantization, compute_dtype)
    kwargs = dict(num_labels=cfg.num_labels, trust_remote_code=True,
                  torch_dtype=compute_dtype)
    if bnb is not None:
        kwargs["quantization_config"] = bnb
        kwargs["device_map"] = "auto"
    try:
        model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            "\n[ERROR] Tai model that bai. Nguyen nhan thuong gap:\n"
            "  1. Chua dang nhap HuggingFace (model gated): huggingface-cli login\n"
            "  2. Chua chap nhan license tren trang model.\n"
            "  3. Sai model ID: " + cfg.model_name + "\n"
            f"  Loi goc: {exc}") from exc

    # dam bao co pad token id (classifier can no de tim token cuoi, khong phai pad)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        # Qwen doi khi khong co pad lan eos id -> them tuong minh
        tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
        model.resize_token_embeddings(len(tokenizer))
        pad_id = tokenizer.pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    model.config.pad_token_id = pad_id

    # resize embedding cho MOI token vua them vao tokenizer (mask token trong
    # load_tokenizer() + pad token neu vua them o tren). mean_resizing=True (mac dinh
    # transformers) khoi tao hang moi bang trung binh embedding hien co thay vi random
    # thuan — on dinh hon cho token chua tung thay trong pretraining.
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        print(f"  [info] resize embedding -> {len(tokenizer):,} token")

    if quant in ("int4", "int8"):
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # chi train rieng cac hang embedding cua token mask moi (khong dung modules_to_save
    # cho embed_tokens vi se train lai toan bo embedding da pretrain). PEFT match
    # "embed_tokens" theo ten suffix, tu tim dung lop du kien truc long nhau bao nhieu
    # lop (vd Qwen3.5: model.language_model.embed_tokens).
    mask_token_ids = tokenizer.convert_tokens_to_ids(MASK_TOKENS)

    lc = cfg.lora
    lora = LoraConfig(
        r=lc.r, lora_alpha=lc.lora_alpha, lora_dropout=lc.lora_dropout,
        target_modules=lc.target_modules, bias=lc.bias, task_type=lc.task_type,
        modules_to_save=lc.modules_to_save, use_rslora=lc.use_rslora,
        use_dora=lc.use_dora, init_lora_weights=lc.init_lora_weights,
        trainable_token_indices={"embed_tokens": mask_token_ids})
    model = get_peft_model(model, lora)
    # ep pad_token_id vao MOI module con co config (chac chan cham lop Qwen ben trong)
    for module in model.modules():
        if hasattr(module, "config") and hasattr(module.config, "pad_token_id"):
            module.config.pad_token_id = pad_id
    model.config.pad_token_id = pad_id
    model.print_trainable_parameters()
    return model
