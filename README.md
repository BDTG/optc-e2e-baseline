# optc-e2e-baseline

End-to-end baseline evaluation on the **DARPA OpTC** dataset: reproducible PIDSMAKER
pipelines (Flash · MAGIC · Velox · ORTHRUS) plus the Phase-0 evidence comparing
TF-IDF vs ModernBERT vs a small LoRA-tuned language model.

This repository is the coordination point between the two machines in the project:

- **Machine A (light)** — 16 GB RAM: builds databases, runs pilot subsets, reproduces
  PIDSMAKER end-to-end, maintains docs/reports.
- **Machine B (GPU)** — RTX 5070 Ti + Ryzen 9 9950X / 64 GB: runs the full-graph
  (L6) baseline that Machine A cannot fit in memory.
- **Machine L6** — Win10 22H2, 32 GB, WSL2 Ubuntu: chạy Flash/MAGIC full
  📌 **Toàn bộ data lưu tại `E:\Data\Thai` (WSL `/mnt/e/Data/Thai`)** — dump/raw/artifacts
  đều để đó, không để `/home/vung2`.

---

## Repository layout

```
patches/       Source patches applied to the PIDSMAKER harness (8 code files + JSON-skip fix)
config/        PIDSMAKER config.py — database=optc_h051_full, canonical 19–21/22/23–25 split
gt/            Ground truth: uuid_index_map (113 nodes), gt_1hop/gt_2hop fragments,
               original ORTHRUS node_h051_0925.csv
scripts/       split_frac (matches original load_split), precision_1hop_2hop,
               calib_threshold, reproduce_tfidf_identity
runbook/       Step-by-step RUNBOOK for the L6 baseline on Machine B (commands + pitfalls)
setup_all.sh   ONE-COMMAND auto-setup for a fresh WSL2 Ubuntu: packages → Miniconda →
               Python 3.9 → deps → PIDSMAKER clone + patches → Postgres (optionally restore dump)
YEU_CAU_CAI_THIEN_PHASE0.md — protocol requirements for reproducible Phase-0 re-runs
                             (Vietnamese; key points: seed parametrization, AUC-PR as
                             primary metric, VRAM logging, multi-seed/multi-fold)
DATA_TRANSFER.md — large files (>100 MB) are distributed out-of-band; see the checksums here
```

> `vendor/` (Phase-0 code from the private repo `imHaruuu/Finetune`) was REMOVED from this
> public repo on purpose — it is not needed to run the L6 baseline. Only setup_all.sh +
> patches + config + GT are required.

## Pipeline

```
parser/export_optc_ecar.py        raw eCAR NDJSON -> train_data_<bundle>.jsonl
parser/refine_labels.py           coarse window labels -> true process lineage (_refined)
baseline_tfidf.py                 cheapest baseline (histogram gradient boosting)
baseline_encoder.py               ModernBERT/DeBERTa ~150M — the real opponent for H0
train.py                          SLM (Qwen3.5-4B LoRA + int4)
measure_cpu.py                    real on-device CPU latency/RAM (RQ2/RQ3)
```

## Running the L6 baseline (Machine B)

1. Read [`runbook/RUNBOOK_remote_L6.md`](runbook/RUNBOOK_remote_L6.md) — requires ≥32 GB RAM.
2. Apply patches (chỉ 1 file — nó đã bao gồm cả skip-JSON deviation):
   ```bash
   git apply patches/all_patches.diff
   ```
   > `pidsmaker_patch.diff` cũ đã bỏ — nó bị hỏng (corrupt) và trùng nội dung đã có
   > trong `all_patches.diff`. Các thay đổi của nó (skip malformed JSON) nằm sẵn trong all_patches.
3. Replace config: copy `config/config.py` into `pidsmaker/config/`.
4. Restore the database from `optc_h051_full.dump` (see `DATA_TRANSFER.md`).
5. Run the 4 systems (Flash / MAGIC / Velox / ORTHRUS) with the commands in the runbook.

## Why the patches are required

The two patches under `patches/` encode ~15 pipeline-stopping bugs discovered over 30+
runs on Machine A (P1-14 → P1-45) while making the upstream PIDSMAKER harness work on
our OpTC data. **Do not skip them** — running the pristine upstream code re-hits the
same failure chain and costs days.

### `all_patches.diff` (8 source files, ~10 fixes)

| File | Original bug fixed | Impact |
|---|---|---|
| `create_database_optc.py` | `isinstance` guard, `ON CONFLICT (node_uuid)`, utf-8 encoding, dropped invalid `tqdm(errors=…)` | DB creation crashed mid-run |
| `create_database_optc.py` | INSERT column mismatch → string `event_uuid` landed in `edge_label INTEGER` | malformed event table |
| `create_database_optc.py` + `build_default_graphs.py` | 3-way schema sync: INSERT stores `path/cmd/index_id`, SELECT reads exact columns, schema gains columns | nodes had no index_id → `KeyError: '0'` → whole graph construction failed |
| `build_default_graphs.py` | `split2nodes` fallback for single-day data (only `test` key exists) | graphs could not build from 1-day bundles |
| `featurization_utils.py` | `get_corpus` skips missing splits | word2vec crashed with `KeyError: 'val'` |
| `feat_inference.py` | zero-vector fallback + dimension fix for nodes outside corpus | `KeyError` and `torch.vstack` size mismatch |
| `training_loop.py` | fallback train→test for `data_sample` | empty `train_data` (single-day) crashed training |
| `data_utils.py` | `load_all_datasets` skips splits with no files | `FileNotFoundError …/edge_embeds/train` |
| `utils.py` | netflow query `src_addr/dst_addr` → `local_ip` | missing column in current schema |

### `pidsmaker_patch.diff` (1 file — JSON-skip deviation)

Skips malformed JSON events during DB creation (previously fatal). Kept **separate**
on purpose: it is a deviation from upstream PIDSMAKER behavior and must be called out
individually in the reproducibility checklist rather than buried in the generic fixes.

## Key evidence (Phase 0, external machine — different dataset, trends only)

| Data regime | Model | SEC | MCC | F1 | recall@1% |
|---|---|---|---|---|---|
| Abundant (~250 mal/train) | TF-IDF | 0.911±0.012 | 0.833±0.077 | 0.830±0.083 | 0.867±0.050 |
| | ModernBERT | 0.879±0.030 | 0.887±0.032 | 0.888±0.028 | 0.856±0.050 |
| | SLM 4B LoRA | 0.841±0.025 | 0.828±0.022 | 0.827±0.029 | 0.833±0.000 |
| Scarce (23 mal/train) | TF-IDF | 0.999±0.001 | 0.993±0.003 | 0.994±0.003 | 0.993±0.002 |
| | ModernBERT | 0.933±0.137 | 0.809±0.402 | 0.800±0.427 | 0.797±0.433 |
| | SLM 4B LoRA | 0.599±0.145 | 0.150±0.319 | 0.173±0.305 | 0.023±0.003 |

> 28 runs, host-holdout, 2 folds. Detail: `multi_seed_robustness_report.md`
> in the `phase1/results/` tree of the working repo.

## Contributing / results round-trip

After Machine B finishes, push back: run logs + `results.pth` + precision@1hop/2hop +
`metrics.json`. Machine A updates `baseline_comparison.md` and the research notes.
