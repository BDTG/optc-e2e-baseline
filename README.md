# OpTC E2E Baseline — Research Repository

Luận văn: *Phát hiện bất thường endpoint trên provenance graph + SLM tier-2* (OpTC H051).

---

## 1. Cấu trúc repo

```
optc-e2e-baseline/
├── README.md                    ← file duy nhất này (consolidated)
├── .gitignore                   (data/, logs/, *.pkl/pth, model weights)
│
├── P0/                          # Phase 0: Flash + Magic baselines (tầng 1, end-to-end)
│   ├── Code/                    # setup_all.sh, config, scripts, vendor, runbook
│   ├── Output/                  # logs_flash.log, logs_magic.log, flash_magic_report.md
│   └── gt/                      # ground truth OpTC
│
├── P1/                          # Phase 1: ORTHRUS + Velox + SLM tier-2 (tầng 1 GNN + tầng 2 SLM)
│   ├── Code/                    # orthrus.yml, velox worktree, patches OpTC H051,
│   │                            # tier2_lora_train.py, tier2_bert_train.py, tier2_distill.py,
│   │                            # tier3_ttp_build.py, benchmark_hw_grid.py, tier2_*.bat
│   └── Output/                  # xem §5 chi tiết bên dưới
│
└── data/                        # (gitignored) raw data OpTC + WSL vhdx 16GB
    ├── orthrus/velox/           # ground truth + edge tables
    ├── nid2msg_cache.pkl        # 2.9M node-id → msg mapping
    └── atomic-red-team/         # Atomic Red Team repo (TTP unseen test)
```

---

## 2. Tóm tắt kết quả (3 phase)

### Phase 0 — Detector layer trên OpTC H051

**Temporal split:** train 19-21/09 | val 22/09 | **test 23-25/09**.  
**GT** = 114 malicious nodes, **chỉ ~34-37 active trong test period** → recall ceiling thực tế ≈ 30%.  
Số dưới đây là Recall@k tính trên **active-GT ceiling** (không chia cho 114).

| Model  | AUC | R@10K | R@20K | R@50K | Scope |
|--------|-----|-------|-------|-------|-------|
| **Velox** | **0.9170** | **57%** | **81%** | **86%** | full test 2940 TWs |
| **ORTHRUS** | 0.8879 | 38% | 79% | 82% | full test 145 TWs |
| Flash | 0.5645 | 5% | 32% | 32% | full DB 7 ngày |
| Magic | 0.7513 | 3% | 8% | 8% | full DB 7 ngày |

Default threshold (max_val_loss):

| Model | Scope | ROC-AUC | AP | Recall | Precision | TP | FP |
|-------|-------|---------|----|--------|-----------|----|----|
| Velox | test 23-25/09 | 0.9170 | 0.00268 | 0% | 0% | 0 | 0 |
| ORTHRUS | test 23-25/09 | 0.8879 | 0.00373 | 0% | 0% | 0 | 6 |
| Flash | full DB 7 ngày | 0.5645 | 0.0005 | 76.3% | 0.0001 | 87 | 919,526 |
| Magic | full DB 7 ngày | 0.7513 | - | 54.4% | 0.0003 | 62 | 185,668 |

⚠️ **Không so sánh trực tiếp**: Flash/Magic đánh giá trên toàn bộ 7 ngày (gồm cả train/val, recall 76%/54% đếm cả attack đã train nên cao giả tạo); ORTHRUS/Velox chỉ test 23-25/09. Default threshold của ORTHRUS/Velox quá nghiêm → dùng bảng Recall@k ở trên.

### Phát hiện chính Phase 0

- **Velox vượt ORTHRUS** trên full test ở mọi ngân sách alert — đảo ngược kết luận pilot (ORTHRUS pilot @10K = 88.6% là do test scope chỉ 1 ngày tấn công).
- **Recall ceiling 30%**: 80/114 GT nodes không xuất hiện trong edges 23-25/09. Mọi số recall /114 phải đọc kèm ceiling.
- Bug `gen_nodeid2msg` (positional index sau DB migration) đã sửa → precision ceiling path+cmd: **33.3% @2K, 8.46% @10K** → P1 gate PASSED cho SLM tier-2.
- TF-IDF baseline: cắt **99.35% FP giữ 100% TP** @k=2000 → SLM zero-shot phải vượt mốc này (RQ1a / H0).

### Phase 1 — Velox optimization

Epoch 66s → 40s (-39%), RAM bounded ~3.5GB không swap trên laptop i5-10300H 16GB.

### Phase 2 — Tier-2 SLM experiments (Note.md RQ1a, RQ1b)

**Mục tiêu:** Test (a) H0 encoder ≥ SLM (Note.md:89), (b) generalization gap trên TTP unseen.

| Tier | Model | Train | AP (V2 single) | AP (V2 5-fold CV) | AP (TTP unseen) | Verdict |
|------|-------|-------|----------------|-------------------|------------------|---------|
| 0 | Random | — | 0.0044 | — | 0.060 | baseline |
| 1 | TF-IDF char + LR | V2 | 0.89 (subset) | 0.254 (OOF global) | — | **winner thực sự** |
| 2 | SLM 0.5B fewshot | — | 0.29 | — | — | thua |
| 2 | SLM 1.5B fewshot | — | 0.57 | — | — | thua |
| 2 | LoRA Qwen 0.5B 1ep | V2 | 0.0044 | — | 0.1174 | ≈ random |
| 3 | ModernBERT 150M | V2 | **0.9999 (artifact!)** | **0.0049 ± 0.0009** | 0.6603 | single holdout artifact |
| 3 | TinyBERT 4M distilled | V2 (KD từ BERT) | 0.0038 | — | — | distill fail |

**🚨 PHÁT HIỆN QUAN TRỌNG — BERT 150M artifact bị bác bỏ:**
- Single 450-holdout AP=0.9999 chỉ có **2 positives** → ranking spurious do shortcut
- 5-fold CV (preserves 12 pos / 2238 neg ratio per fold): **AP=0.0049 ± 0.0009** ≈ random 12/2250
- AUC 5-fold=0.25 (dưới random 0.5) → score ngược
- → **BERT 150M không có lợi thế so với TF-IDF** khi đánh giá đúng phương pháp

**Phát hiện tổng quát Phase 2 (đã sửa sau CV):**
1. **Note.md:89 reframe (sau CV):** TF-IDF > encoder deep learning (BERT 150M CV-AP 0.005 ≈ random) ≈ SLM zero/few/LoRA > distilled student. SLM tier-2 độc lập **không khả thi**.
2. **TF-IDF + LR mới là winner thực sự** cho tier-2 V2: AP 0.89 subset, OOF 0.254 global, latency 1.5ms.
3. **Generalization gap có thật** (nếu tin single holdout): BERT 0.9999 (V2) → 0.6603 (TTP), nhưng cả hai đều nghi ngờ artifact vì imbalance.
4. **Distill xuống 4M thất bại**: loss giảm mạnh nhưng val AP không cải thiện.
5. **Caveats nghiêm trọng**: V2 chỉ 12 positives toàn dataset → AP single holdout unreliable. TTP holdout 19 chain từ YAML (không phải event chain thật). Cần sysmon + real provenance để verify.

**Phát biểu lại Note.md:**
- Note.md:7 "SLM tầng hai" vẫn đúng về kiến trúc nhưng **không phải primary detector**.
- Note.md:89 "loại trừ H0" → không loại trừ được với V2. Cần reframe: **TF-IDF + LR là tier-2 đủ tốt, không cần deep learning**.

---

## 3. Setup & reproduction

### Yêu cầu
- Python 3.11, torch 2.11+cu128, transformers 5.16
- Dataset OpTC H051 (≈25GB) — torrent `thai.rar` đã có, extract vào `data/orthrus/`
- Hardware đo: i5-10300H 16GB + GTX 1650 Ti 4GB (driver 616.56), CUDA 12.x

### Cài env
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers peft bitsandbytes accelerate datasets pyyaml scikit-learn
```

### Chạy end-to-end
```bash
# Phase 0: detectors
cd P0/Code && bash setup_all.sh
bash runbook/run_flash.sh
bash runbook/run_magic.sh

# Phase 1: Velox full test
cd P1/Code
# Apply patches (Velox + ORTHRUS for OpTC H051)
git apply patches/velox_worktree.patch -C ../velox_src
git apply patches/orthrus_opc_h051.patch -C ../orthrus_src
# Train + test (đã có artifacts trong Output/models/)

# Phase 2: tier-2 SLM experiments
cd P1/Code
python tier2_lora_train.py        # LoRA 0.5B 1 epoch (GPU 28 phút)
python tier2_lora_eval.py         # eval on 450 holdout
python tier2_bert_train.py        # BERT 150M 3 epoch bf16 (GPU 28 phút)
python tier2_distill.py           # KD BERT -> TinyBERT 4M
python tier3_ttp_build.py         # parse atomic-red-team YAML
python tier3_bert_test.py         # BERT on TTP unseen
python tier3_lora_test.py         # LoRA on TTP unseen
```

---

## 4. Bằng chứng thực nghiệm (Phase 0)

| File | Nội dung |
|------|---------|
| `P1/Output/results_phase0/final-metrics-full-test.json` | ORTHRUS full test 145 TWs metrics |
| `P1/Output/results_phase0/fm-sweep-full.json` | Flash/Magic recall@k sweep |
| `P1/Output/results_phase0/orthrus-ceiling-corrected.txt` | Precision ceiling cho path+cmd features |
| `P1/Output/results_phase0/orthrus-precision-ceiling-k10k.txt` | Ceiling @10K |
| `P1/Output/results_phase0/orthrus-threshold-sweep.txt` | ORTHRUS threshold sweep |
| `P1/Output/results_phase0/velox-ceiling-corrected.txt` | Velox ceiling |
| `P1/Output/results_phase0/velox-threshold-sweep.txt` | Velox threshold sweep |
| `P1/Output/results_phase0/result_model_epoch_1.pth` | ORTHRUS model checkpoint |

## 5. Bằng chứng thực nghiệm (Phase 1)

| File | Nội dung |
|------|---------|
| `P1/Output/results_phase1/velox-optimization-results.md` | Velox T1-T3 (chunked lazy-load, GPU bf16) |
| `P1/Output/results_phase1/metrics-summary.txt` | Tổng hợp đầy đủ phase 0 + 1 |
| `P1/Output/results_phase1/pseudocode-tier2.md` | Mã giả tier-2 cho paper |

## 6. Bằng chứng thực nghiệm (Phase 2)

| File | Nội dung |
|------|---------|
| `P1/Output/results_phase2/go-nogo-decision.md` | Phân tích NO-GO đầy đủ 8 sections |
| `P1/Output/results_phase2/report-for-advisor.txt` | Báo cáo cho thầy (Tiếng Việt) |
| `P1/Output/results_phase2/report-phase2-wsab.md` | Working-scratch báo cáo phase 2 |
| `P1/Output/results_phase2/context-phase2-execution.md` | Context file cho agent |
| `P1/Output/results_phase2/slm-tier2-v2-cv.json` | TF-IDF V2 CV (encoder rẻ baseline AP 0.254) |
| `P1/Output/results_phase2/slm-go-nogo-qwen05b-n40.json` | SLM 0.5B go-nogo n=40 |
| `P1/Output/results_phase2/tfidf-cv-results.json` | TF-IDF CV results |
| `P1/Output/results_phase2/tier2-filter-corrected.json` | Tier-2 filter experiment |
| `P1/Output/results_phase2/lora-05b-epoch1-result.json` | LoRA 0.5B AP=0.0044 |
| `P1/Output/results_phase2/bert-ttp-result.json` | BERT TTP AP=0.6603 |
| `P1/Output/results_phase2/lora-ttp-result.json` | LoRA TTP AP=0.1174 |

## 7. Data + models

| Path | Nội dung |
|------|---------|
| `P1/Output/data/alerts-enriched-v2.jsonl` | 2250 alerts enriched (training data cho tier-2) |
| `P1/Output/data/alerts-enriched-partial.jsonl` | Phiên bản partial (pre-enrich fix) |
| `P1/Output/data/gt_and_scores.json` | 114 GT nids + score_map (322679 entries) |
| `P1/Output/data/ttp_holdout.jsonl` | 319 alerts (10 TTP × ~2 chain + 300 benign) |
| `P1/Output/models/lora-05b/` | LoRA Qwen 0.5B adapter (8.7MB) + checkpoint-113/ |
| `P1/Output/models/bert-150m/` | ModernBERT 150M fine-tuned (300MB, gitignored) |
| `P1/Output/models/tinybert-4m-distilled/` | TinyBERT 4M distilled (AP=0.0038, fail) |
| `P1/Output/logs/` | Training/inference logs (gitignored) |
| `P1/Output/benchmarks/` | HW latency sweeps (hw-grid-partial) |
| `P1/Code/` | Code: orthrus_src/, velox_src/, patches/, tier2_*.py, tier3_*.py |

---

## 8. Restore từ patches
```bash
cd P1/Code/orthrus_src && git apply ../patches/orthrus_opc_h051.patch
cd ../velox_src && git apply ../patches/velox_worktree.patch
```

---

## 9. Đóng góp

| Commit mới nhất | Phase | |
|---|---|---|
| `79e5ea1` refactor(P1/Output) | phase-grouped layout | |
| `298cc65` distill BERT→TinyBERT 4M | AP=0.0038 (fail) | |
| `ad3d746` TTP unseen holdout | BERT 0.66, LoRA 0.12 | |
| `980a9a8` LoRA 0.5B + BERT 150M GPU | GPU 28 phút | |
| `4bac510` quy chuẩn tên file | dọn dẹp (A) | |
| `68baa34` context file Phase 2 | seed cho agent | |

Tất cả commit push lên GitHub: https://github.com/BDTG/optc-e2e-baseline