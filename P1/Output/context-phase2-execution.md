# CONTEXT FOR PHASE 2 EXECUTION — SLM Tier-2 Research
# (File này dùng để seed context cho model coding agent)

## TRẠNG THÁI HIỆN TẠI (so với Note.md)

### Đã hoàn thành
- **Phase 0 (Go/no-go):** CHƯA QUAY LẠI
  - OpTC H051 full DB loaded: 19,815,600 events, 7 ngày (19-25/09/2019)
  - Train 19-21/09 (230 TWs), val 22/09 (94 TWs), test 23-25/09 (145 TWs)
  - Ground truth: 114 malicious nodes (chỉ 34-37 active trong test period)
  - Flash/Magic/ORTHRUS/Velox: FULL TEST hoàn thành, AUC 0.56-0.92
  - Recall ceiling ~30% (80/114 GT nodes "im" trong test period)
  - **TF-IDF baseline chưa chạy:** encoded alerts 2250/10000 (chưa đủ, enrich paused)
  - **SLM 3B/8B CHƯA ĐÁNH GIÁ:** `slm_tier2.py` đã viết, chưa run

- **Phase 1 (Baselines):**
  - 4 models full test ✅
  - FP sweep ✅ (ORTHRUS/Velox/Flash/Magic)
  - Precision ceiling corrected ✅ (path+cmd: 33% @2K, 8.46% @10K)
  - Enrichment: 2250/10000 alerts ✅ (paused - cần nodeid2msg query)
  - **Host 201/501 CHƯA TẢI** (cần Google Drive rclone)
  - **SHIELD CHƯA CHẠY**

### Chưa bắt đầu
- **Phase 2 (Hardware grid RQ3):** 0% — CRITICAL BLOCK cho mọi thứ sau
- **Phase 3 (Core experiments RQ1a/RQ1b/RQ4b):** 0%
- **Phase 4 (Generalization RQ5):** 0%
- **Phase 5 (Writing):** 0%

### BOTTLENECKS
1. **WSL máy thầy không ổn định** (teardown 4-10 phút) → không chạy được enrichment hay SLM trên đó
2. **Laptop không có transformers/bitsandbytes** (chỉ torch CPU)
3. **Dataset host 201/501 chưa tải** (cần Google Drive rclone)
4. **Enrichment paused** ở 2250/10000 alerts

---

## CÁI GÌ CÓ THỂ LÀM NGAY TẠI ĐÂY (không cần thầy)

### 1. Install transformers + bitsandbytes trên laptop
```
pip install transformers accelerate bitsandbytes
# bitsandbytes cần CUDA adapter nhưng mình CPU-only → cần workaround
# Hoặc: dùng llama.cpp GGUF format cho Qwen2.5-3B int4 (chạy được CPU)
```

### 2. Chạy SLM 3B int4 trên 2250 alerts đã enrich
```
python slm_tier2.py \
  --alerts_jsonl P1/Output/alerts_enriched_partial.jsonl \
  --gt_csv D:\orthrus_laptop\optec-l6\orthrus\Ground_Truth\OPTC_H051\node_h051_0925.csv \
  --alert_k 2000 \
  --skip_slm  # chạy baseline TF-IDF trước
```
→ Kết quả RQ1a preliminary: FP reduction ở k=2000 trên 2250 alerts

### 3. Hardware grid baseline (RQ3 partial)
Đo latency torch CPU inference trên laptop i5-10300H:
- ORTHRUS inference time per graph (~2.5s/graph)
- Velox inference time per TW (~0.3s/TW)
- Ghi lại: i5-10300H 4C/8T, 16GB DDR4, Windows 11

---

## CÁI GÌ CẦN THẦY/MÁY THẦY

1. **Tải host 201 + 501 bundles** từ Google Drive
   - rclone config với service account, hoặc wget trực tiếp
   - Cần bundle: `201_evaluation`, `201_benign`, `501_evaluation`, `501_benign`
2. **Fix WSL stability** (hoặc dùng Docker trên Windows)
3. **SGD/conda environment** với transformers + bitsandbytes cho GPU inference

---

## HƯỚNG DẪN CHO MODEL TIẾP THEO

### Priority 1: Chạy RQ1a trên data hiện có
File: `P1/Code/slm_tier2.py`
Input: `P1/Output/alerts_enriched_partial.jsonl` (2250 alerts)
Tính:
1. `run_baseline_tfidf()` trên 2250 alerts → FP reduction @k={500,1000,2000}
2. Nếu có transformers: `classify_with_slm()` trên top-2000 alerts
3. So sánh TF-IDF vs SLM: FP reduction difference = contribution RQ1a

### Priority 2: Hoàn thành enrichment
Script: cần viết enrichment script query DB
- DB optc_h051_full trên localhost:5432 (không password cho localhost peer auth)
- Query parent chain (3 hops) + 20 events gần nhất cho mỗi malicious node
- Output: enriched JSONL, 10000 alerts

### Priority 3: Hardware grid (RQ3)
Trên laptop i5-10300H, measure:
- Model sizes: Qwen2.5 0.5B, 1.5B, 3B (8B cần 28GB+ RAM → skip)
- Quantization: int4 (GGUF), int8, fp16 (nếu có)
- Token length: 128, 512, 2048
- Đo: latency p50/p95, weight footprint, KV cache, CPU-second/host/ngày

### Priority 4: Download host 201/501
Cần Google Drive access + rclone để tải bundles từ FiveDirections/OpTC-data

---

## CONSTRAINTS TỪ NOTE.MD
1. **Kerckhoffs:** KHÔNG đặt logic phát hiện trong prompt SLM. Prompt chỉ hướng dẫn format.
2. **H0 elimination:** Phải chứng minh SLM hơn encoder 150M (TF-IDF + LogReg hoặc BERT)
3. **Metric chính:** recall@{1,10,100} alert/host/ngày, KHÔNG dùng AUC-ROC
4. **Tầng 2 = secondary detector:** Trên ~10²-10³ candidate/host/ngày từ tier-1
5. **Zero-shot cần few-shot/LoRA:** "Nếu paper chỉ có zero-shot thì kết luận là model nhỏ không làm được"
6. **Evasion/injection:** Đo trung thực, đánh số cụ thể, không giấu trong appendix

---

## FILE PATHS
- Repo: `C:\Users\BDTG\OpTC-data\phase1\gh_repo`
- SLM code: `P1/Code/slm_tier2.py`
- Enriched alerts: `P1/Output/alerts_enriched_partial.jsonl` (2250 alerts)
- Laptop ORTHRUS artifacts: `D:\orthrus_laptop\optec-l6\`
- GT CSV: `D:\orthrus_laptop\optec-l6\orthrus\Ground_Truth\OPTC_H051\node_h051_0925.csv`
- ORTHRUS results pth: `D:\orthrus_laptop\optec-l6\orthrus\artifacts\detection\evaluation\*\OPTC_H051\precision_recall_dir\result_model_epoch_1.pth`
- DB: localhost:5432, db=optc_h051_full, user=vung2, password=[orthrus/src/config.py]
- Note.md: `C:\Users\BDTG\Downloads\Note.md`
