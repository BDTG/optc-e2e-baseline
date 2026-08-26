# OpTC E2E Baseline - Research Repository

## Cấu trúc
```
optc-e2e-baseline/
├── P0/                    # Phase 0: Flash + Magic baselines (tầng 1)
│   ├── Code/             # setup_all.sh, config, scripts, vendor, runbook
│   ├── Output/            # logs_flash.log, logs_magic.log, flash_magic_report.md
│   └── gt/                # ground truth OpTC
├── P1/                    # Phase 1: ORTHRUS + VELOX + SLM tier-2 (tầng 2)
│   ├── Code/
│   │   ├── orthrus_src/   # ORTHRUS source (10+ patches OpTC H051)
│   │   ├── velox_src/     # Velox source (patches OpTC H051)
│   │   ├── pidsmaker_config/
│   │   ├── patches/        # orthrus_opc_h051.patch, velox_worktree.patch
│   │   ├── launch_scripts/
│   │   └── orthrus.yml
│   └── Output/            # METRICS_SUMMARY.txt, FINAL_METRICS_full_test.json,
│                          # FM_SWEEP_full.json, VELOX_OPTIMIZATION_RESULTS.md
└── README.md
```

## Kết quả chính — FULL COVERAGE (không còn pilot)

Temporal split: train 19-21/09 | val 22/09 | **test 23-25/09**.
GT = 114 malicious nodes, trong đó **chỉ ~34-37 hoạt động trong test
period** → recall ceiling thực tế ≈ 30%. Số dưới đây là Recall@k tính
trên active-GT ceiling.

| Model | AUC | R@10K | R@20K | R@50K | Ghi chú |
|-------|-----|-------|-------|-------|---------|
| **Velox** | **0.9170** | **57%** | **81%** | **86%** | full test 2940 TWs |
| **ORTHRUS** | 0.8879 | 38% | 79% | 82% | full test 145 TWs |
| Flash | 0.5645 | 5% | 32% | 32% | full DB 7 ngày |
| Magic | 0.7513 | 3% | 8% | 8% | full DB 7 ngày |

Chi tiết đầy đủ: [`P1/Output/METRICS_SUMMARY.txt`](P1/Output/METRICS_SUMMARY.txt),
[`P1/Output/FINAL_METRICS_full_test.json`](P1/Output/FINAL_METRICS_full_test.json),
[`P1/Output/FM_SWEEP_full.json`](P1/Output/FM_SWEEP_full.json).

### Phát hiện chính
- **Velox vượt ORTHRUS trên full test** ở mọi ngưỡng budget — đảo ngược
  kết luận pilot trước đây (ORTHRUS pilot @10K = 88.6% là do test scope
  chỉ 1 ngày tấn công).
- **Recall ceiling 30%**: 80/114 GT nodes không xuất hiện trong edges
  23-25/09. Mọi số recall /114 phải đọc kèm ceiling này.
- Flash/Magic ranking gần random (@50K ≤ 32%) dù recall thô cao.
- Bug `gen_nodeid2msg` (positional index sau DB migration) đã sửa →
  precision ceiling path+cmd: 33.3% @2K, 8.46% @10K → **P1 gate PASSED**
  cho SLM tier-2.
- TF-IDF baseline: cắt 99.35% FP giữ 100% TP @k=2000 → SLM zero-shot
  phải vượt mốc này (RQ1a / H0).

### Velox performance optimization (T1-T3)
Epoch 66s → 40s (-39%), RAM bounded ~3.5GB không swap.
Chi tiết: [`P1/Output/VELOX_OPTIMIZATION_RESULTS.md`](P1/Output/VELOX_OPTIMIZATION_RESULTS.md)

## Restore từ patches
```bash
cd P1/Code/orthrus_src && git apply ../patches/orthrus_opc_h051.patch
cd ../velox_src && git apply ../patches/velox_worktree.patch
```
