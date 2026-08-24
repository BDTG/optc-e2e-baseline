# OpTC E2E Baseline - Research Repository

## Cấu trúc
```
optc-e2e-baseline/
├── P0/                    # Phase 0: Flash + Magic baselines (tầng 1)
│   ├── Code/             # setup_all.sh, config, scripts, vendor, runbook
│   ├── Output/            # logs_flash.log, logs_magic.log, flash_magic_report.md
│   └── gt/                # ground truth OpTC
├── P1/                    # Phase 1: ORTHRUS + VELOX (tầng 2)
│   ├── Code/
│   │   ├── orthrus_src/   # ORTHRUS source (10+ patches OpTC H051)
│   │   ├── velox_src/     # Velox source (patches OpTC H051)
│   │   ├── pidsmaker_config/
│   │   ├── patches/        # orthrus_opc_h051.patch, velox_worktree.patch
│   │   ├── launch_scripts/
│   │   └── orthrus.yml
│   └── Output/            # orthrus_run.log, logs_velox.log
└── README.md
```

## Kết quả đã đạt (OpTC H051, temporal split)
| Baseline | AUC | Recall | FP | TP | Note |
|----------|-----|--------|----|----|------|
| Flash    | 0.56 | 76% | 919K | 87 | full DB |
| Magic    | 0.75 | 54% | 186K | 62 | full DB |
| ORTHRUS  | 0.94 | 7.9% | 3,406 | 9 | pilot (test = ngày 25/09) |

Velox: đang chạy (self-train, GPU RTX 5060 Ti).

## Restore từ patches
```bash
cd P1/Code/orthrus_src && git apply ../patches/orthrus_opc_h051.patch
cd ../velox_src && git apply ../patches/velox_worktree.patch
```
