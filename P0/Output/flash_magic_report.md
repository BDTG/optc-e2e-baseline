# BÁO CÁO KẾT QUẢ — FLASH & MAGIC BASELINE (L6 Full)

**Dataset:** DARPA OpTC H051 — DB full 7 ngày (19.8M events)  
**Split:** Temporal train=19-21 / val=22 / test=23-25 (theo config chuẩn)  
**Ground truth:** ORTHRUS (114 attacked nodes)  
**Máy chạy:** Thầy (Win10 22H2, WSL2 Ubuntu, 28GB RAM, qua Tailscale SSH)  
**Ngày chạy:** 2026-08-20 → 2026-08-21  
**Lưu trữ:** `E:\Data\Thai\` (artifacts_flashmagic_full, artifacts_flash_backup, artifacts_magic_backup, logs)

---

## 1. FLASH (full)

| Metric | Giá trị |
|--------|---------|
| Nodes evaluated | 927,476 |
| TP (phát hiện đúng) | 87 |
| FP (báo sai) | 919,526 |
| FN (bỏ sót) | 27 |
| TN | 7,836 |
| **Precision** | 0.0001 (0.01%) |
| **Recall** | 0.7632 (76.3%) |
| **F1** | 0.0002 |
| **ROC-AUC** | 0.5645 |
| **PR-AUC** | 0.0005 |
| GT attacked nodes | 114 |

**Thời gian chạy:** ~3h (bắt đầu 20/08 22:39, xong 21/08 02:22)  
**Artifacts:** `artifacts_flashmagic_full/evaluation/evaluation/abd2bffb.../optc_h051/results/results.pth` (64.8MB)  
**Scores:** `scores_model_epoch_11.pkl`

---

## 2. MAGIC (full, featurization=only_type)

| Metric | Giá trị |
|--------|---------|
| Nodes evaluated | 927,476 |
| TP | 62 |
| FP | 185,668 |
| FN | 52 |
| TN | 741,694 |
| **Precision** | 0.0003 (0.03%) |
| **Recall** | 0.5439 (54.4%) |
| **F1** | 0.0007 |
| **ROC-AUC** | 0.7513 |
| **PR-AUC** | (tính sau) |
| GT attacked nodes | 114 |

**Thời gian chạy:** ~1h (02:32 → 03:33)  
**Artifacts:** `artifacts_flashmagic_full/evaluation/evaluation/3b2acf17.../optc_h051/results/results.pth`  
**Scores:** `scores_model_epoch_11.pkl`

---

## 3. SO SÁNH & ĐÁNH GIÁ

| | Flash | Magic |
|---|---|---|
| TP / FN | 87 / 27 | 62 / 52 |
| FP | 919,526 | 185,668 |
| Recall | **76.3%** | 54.4% |
| **ROC-AUC** | 0.5645 | **0.7513** |
| Precision | 0.01% | 0.03% |

**Nhận xét:**
- MAGIC có **AUC cao hơn hẳn** Flash (0.75 vs 0.56) → ranking node tốt hơn
- Flash bắt được **nhiều node thật hơn** (recall 76% vs 54%) nhưng **báo sai nhiều gấp 5 lần** (919K vs 186K FP)
- **Cả 2 đều có Precision cực thấp (~0.01-0.03%)** → báo động quá nhiều node vô hại

→ **Khẳng định RQ1a (Note.md):** PIDSMAKER detectors (Flash/Magic) **không thể dùng trực tiếp** làm alert — cần:
  1. FP analysis (tại sao báo sai nhiều?)
  2. SLM tầng 2 (Phase 1) để lọc FP

**So với subset pilot (trước đó):**
- Flash subset: tp=0, fn=114, AUC=0.678 → full: tp=87, fn=27, AUC=0.56 (tiến bộ do full data + temporal split)
- Magic subset: tp=69, fp=116K, recall=0.605, AUC=0.727 → full: tp=62, fp=186K, recall=0.54, AUC=0.75 (khớp xu hướng)

---

## 4. TRẠNG THÁI LƯU TRỮ

```
E:\Data\Thai\
├── optc_h051_full.dump              (DB dump 1.1GB)
├── artifacts_flashmagic_full\        (14GB - TOÀN BỘ pipeline Flash+Magic)
│   ├── evaluation/  → results.pth + scores_*.pkl  ⭐
│   ├── training/    → model epoch 0-11
│   ├── feat_inference/ transformation/ construction/  (intermediate)
│   └── featurization/ batching/ postprocessing/ triage/  (nhẹ/rỗng)
├── artifacts_flash_backup\           (609MB - evaluation Flash)
├── artifacts_magic_backup\           (1.1GB - evaluation Magic)
├── logs\                             (logs_flash.log + logs_magic.log)  ⭐ MỚI
├── artifacts\                        (Velox đang chạy)
└── logs_velox.log                   (0 byte - đang chạy)
```

**Backup đã verify:** results.pth Flash + Magic đều có mặt trên ổ E.

---

## 5. TIẾP THEO
- [ ] Velox full đang chạy (28GB RAM, artifact_dir đúng ổ E)
- [ ] ORTHRUS full (sau Velox)
- [ ] Cập nhật `baseline_comparison.md` + `LOG.md` chính thức
- [ ] FP analysis (RQ1a) — tại sao 919K/186K FP?

---
*Báo cáo tạm thời — viết 2026-08-21, sẽ cập nhật khi Velox/ORTHRUS xong.*
