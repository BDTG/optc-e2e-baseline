# OpTC SLM Anomaly-Detection Pipeline

Endpoint anomaly detection on the DARPA OpTC dataset. Goal: rule out H0 (does a ~150M
fine-tuned encoder already match an SLM?) before investing in later phases. See the
research `Note.md` for full context.

Separate from [`../legacy/`](../legacy/README.md) (text-generation fine-tuning) — the two
pipelines don't share `config.py`/`data.py`/`model.py`/`utils.py` despite the same file
names. **Run every command below from inside this `optc/` directory** — per-run artifacts
(`train.py`, `baseline_encoder.py`, `baseline_tfidf.py`, `measure_cpu.py`) are written to
`runs/`, relative to the current working directory (see "Per-run artifacts" below).

---

## Pipeline order

```
1. parser/export_optc_ecar.py   raw eCAR NDJSON (E:\dataset\raw\...)  -> train_data_<bundle>.jsonl
   (or parser/export_dataset.py for a different Sysmon/Elastic source, not OpTC)
2. parser/refine_labels.py      train_data_<bundle>.jsonl             -> train_data_<bundle>_refined.jsonl
3. baseline_tfidf.py     cheapest baseline — run BEFORE the encoder/SLM to rule out H0
4. baseline_encoder.py   ModernBERT/DeBERTa ~150M baseline, the REAL opponent for H0
5. train.py              SLM (LoRA + quant), compared against (3)/(4) on the same split
6. measure_cpu.py        real CPU latency/RAM (RQ2/RQ3), independent of (5)
```

`parser/` only holds raw-data-to-JSONL scripts, self-contained — it doesn't import
`config`/`data`/`model` from the rest of the pipeline. `baseline_*.py`/`train.py`/
`measure_cpu.py` live at the `optc/` root because they need `config.py`/`data.py`/
`model.py`/`metrics.py`/`utils.py`.

### 1–2. Export + refine labels

```
python parser/export_optc_ecar.py ^
    --data "E:\dataset\raw\AIA-201-225\AIA-201-225.ecar-2019-12-08T11-05-10.046.json" ^
    --data "E:\dataset\raw\AIA-201-225\AIA-201-225.ecar.json" ^
    --out "E:\dataset\processed\train_data_201.jsonl"

python parser/refine_labels.py --in "E:\dataset\processed\train_data_201.jsonl" --out "E:\dataset\processed\train_data_201_refined.jsonl"
```

`export_optc_ecar.py` assigns coarse labels from a (host, time-window) list read from
`OpTCRedTeamGroundTruth.pdf` (`ATTACK_WINDOWS`) — this over-labels benign background
processes that happen to run in the same window. `refine_labels.py` narrows it down to the
true attacker process lineage (tree traversal over (host,pid)->(host,ppid), with Windows
PID-reuse handling — see the file's docstring). **Always use the `_refined.jsonl` file for
baselines/training, never the raw one.**

### 3. Cheapest baseline (TF-IDF + GBM) — run first

```
python baseline_tfidf.py --data "E:\dataset\processed\train_data_201_refined.jsonl"

# host-holdout: train on one host, test on an unseen one (checks generalization across
# hosts instead of memorization within one)
python baseline_tfidf.py --data "E:\dataset\processed\train_data_combined.jsonl" ^
    --split-mode host_holdout --holdout-hosts SysClient0501.systemia.com
```

### 4. ~150M encoder baseline (the real opponent for H0)

```
python baseline_encoder.py --model answerdotai/ModernBERT-base ^
    --data "E:\dataset\processed\train_data_201_refined.jsonl" --split-frac 0.7
```

### 5. SLM (LoRA + quant)

```
python train.py --model Qwen/Qwen2.5-3B --data "E:\dataset\processed\train_data_201_refined.jsonl"
python train.py --model Qwen/Qwen2.5-1.5B --quant int8   # config sweep for RQ2, appended to results.jsonl
```

### 6. Real CPU latency (RQ2/RQ3)

```
CUDA_VISIBLE_DEVICES="" python measure_cpu.py --adapter <checkpoint-path-from-runs> --base Qwen/Qwen2.5-3B --data train_data.jsonl --n 200 --quant int4
```

---

## Per-run artifacts (`runs/`)

`train.py`, `baseline_encoder.py`, `baseline_tfidf.py`, and `measure_cpu.py` each call
`make_run_dir(script, **tags)` (from `run_utils.py`) at the start of `main()`, creating one
directory per run instead of overwriting a shared `./slm-ckpt`/`./encoder-ckpt`/
`scores.jsonl`:

```
runs/<script>__<tag1>_<tag2>..._<YYYYMMDD-HHMMSS>/
    checkpoint/       # LoRA adapter (train.py) or full weights (baseline_encoder.py)
    scores.jsonl        # per-row test-set risk scores (if applicable)
    metrics.json          # final metrics + the args used for this run (model/split/quant/...)
```

e.g. `runs/baseline_encoder__ModernBERT-base_host_holdout_20260808-014500/`.
`runs/` is gitignored (can get large, never committed). `train.py`'s `results.jsonl` is a
deliberate exception — a table appended across many runs (the RQ2 sweep), lives outside
`runs/`, and is never overwritten.

`run_utils.py` (pure stdlib, no torch) holds `make_run_dir`/`save_run_metrics`/
`read_jsonl`; `utils.py` re-exports them so torch-dependent scripts can still
`from utils import ...` as usual. Kept separate so `baseline_tfidf.py`/`data.py`/
`config.py` don't pull in torch/transformers just to read a JSONL file.

## Metrics

`metrics.py` — `recall@FPR1%`/`recall@FPR0.1%` are the primary metrics (extreme class
imbalance makes AUC-ROC inflate — see `all_metrics()` + `print_metrics()`).

## Shared internal modules (train.py / baseline_encoder.py)

- `eval.py::score_rows()` — batched inference on the test set (not one row at a time),
  returns (scores, labels, mean latency_ms).
- `utils.py::WeightedTrainer` + `compute_class_weights()` — a Trainer with weighted
  cross-entropy loss, shared by both fine-tuning scripts.

`parser/utils.py` is a SEPARATE module (unrelated to `utils.py`/`run_utils.py` here) —
used only internally among the three `parser/` scripts; see that file's docstring.

## Notes

- `config.py::SLMConfig` defaults to `Qwen/Qwen2.5-3B`, `int4` quant, `causal` split. Set
  `split_mode="host_holdout"` + `holdout_hosts` for RQ5a (unseen-host).
- `data.py::load_split` requires at least one positive sample in both train and test — it
  raises a clear error if the split empties one class out (usually when the attack window
  is too narrow relative to `split_frac`).
- Source data (`E:\dataset\raw\`, `processed\`) lives outside this repo — see
  `E:\dataset\docs\` for ground truth/errata.
