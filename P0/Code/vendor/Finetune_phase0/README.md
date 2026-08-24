# LLM Fine-Tuning

Two independent pipelines sharing one Python environment (`.venv/` + `requirements.txt` here):

| Directory | Purpose |
|---|---|
| [`optc/`](optc/README.md) | Endpoint anomaly detection on the DARPA OpTC dataset (LoRA classifier + TF-IDF/encoder baselines) — **active research** |
| [`legacy/`](legacy/README.md) | Original generic QLoRA text-generation fine-tuning template (chat/instruction-following), unrelated to OpTC |

The two pipelines **share no code** — each directory has its own `config.py`/`data.py`/
`model.py`/`utils.py` (same names, different contents). Always `cd` into the target
subdirectory before running a script inside it: default output paths (`./slm-ckpt`,
`./gemma-finetuned`, ...) are relative to the current working directory.

## Setup (shared)

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`legacy/` additionally needs `trl` + `bitsandbytes` (listed in `requirements.txt`) to run
`trainer.py`/`finetune.py` — confirm they're installed before using it. `optc/` needs
neither.
