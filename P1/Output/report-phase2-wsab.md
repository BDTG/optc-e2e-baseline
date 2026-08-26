# REPORT PHASE 2 — WS-A/B (H0 Fix + HW Grid Partial)

**Ngày:** 2026-08-27  **Scope:** H051 2250 alerts (top-2250 ORTHRUS)  **GT:** 12/114 trong window

## 1. WS-A: H0 Baseline Fix

### A1. Re-enrich v2
- Input `alerts_enriched_partial.jsonl:1` generic `subject:SUBJECT None` → Output `alerts_enriched_v2.jsonl` dùng `data/nid2msg_cache.pkl` (2,900,744 entries)
- Script `P1/Code/re_enrich_v2.py:1` enrich `self_label/msg`, `parent_chain[].msg`, `event_seq[].src_msg/dst_msg`
- Result: miss 0/2250 self_label (0%), parent miss 2/2653 (0.1%), event miss 1/16941 — đạt chất lượng cao. Sample V2 GT: `svchost.exe | cmd: None` + `netflow 142.20.56.154:57669 -> 224.0.0.252:5355` (thay vì `node_946071`)
- Impact: text len p50 496 chars (V2) vs 117 (V1), max 6349 vs 2509 — đủ context cho SLM. `P1/Code/tfidf_cv_v2.py:40`

### A2. TF-IDF CV Correct (không leakage)
`P1/Code/slm_tier2.py:250` đã fix: `_build_tfidf_texts()` dùng msg + `run_baseline_tfidf(cv=True)` 5-fold OOF. So sánh leakage vs CV:

| Variant | Leakage AP | OOF AP | Holdout 80/20 AP | k=2000 FP_red | Recall after @k2000 |
|---------|------------|--------|------------------|----------------|---------------------|
| V1 generic IDs | 1.00 | 0.184 | 0.516 | 0.990 (TP 4/12) | 0.333 |
| V2 msg-enriched | 0.854 | **0.254** | **0.520** | 0.973 (TP 6/12) | **0.50** |

- Chi tiết V2 CV: `P1/Output/slm_tier2_v2_cv.json` `ap_oof 0.254 fold [0.27,0.75,0.02,0.52,0.06]` variance cao do pos nhỏ (12).
- **As FILTER** trên original ORTHRUS ranking (`TIER2_FILTER_CORRECTED.json`): original top-500/1000 chỉ chứa 1 GT (rank 8) nên filter giết recall (0/1). Tại k=2000 (chứa đủ 12 GT) V2 filter keep 57 (6 TP/51 FP) prec 0.105 vs orig 0.006, FP_red 97.4% nhưng mất 50% GT. V1 filter keep 23 (4 TP/19 FP) prec 0.174 FP_red 99% recall 0.333 — V2 giữ recall tốt hơn.
- **As RE-RANKER** (TF-IDF sort): V2 top-100 6 TP prec 6% recall 0.5 ; top-500 11 TP prec 2.2% recall 0.92 — vượt xa original ranking (top-100 orig 1 TP). Đây là RQ1b gain, nhưng đổi pipeline từ filter → ranker.
- Kết luận H0: `Note.md:15` yêu cầu SLM hơn encoder 150M. Hiện TF-IDF V2 OOF AP 0.254, SLM zero-shot phải >0.254 và giữ recall ≥0.5 @FP_red ≥97% mới claim RQ1a. Leakage cũ 0.99 là ảo (`tfidf_cv_results.json:12`).

### A3. Prompt Kerckhoffs fix
`P1/Code/slm_tier2.py:24` SYSTEM_PROMPT rút gọn chỉ format `CLASSIFICATION/CONFIDENCE/REASON`, bỏ `obfuscation/LOLBin/C2/Downloads` hints. `format_alert_for_prompt()` cũng ưu tiên `msg` over `node`.

## 2. WS-B: HW Grid Partial (i5-10300H 4C/8T 16GB `hw_grid_benchmark.py:12`)

### B1. TF-IDF measured
- p50 1.23ms, p95 3.04ms, p99 ~3.5ms per alert (200 mẫu, `P1/Output/hw_grid_partial.json:12`)
- Throughput: 70M decisions/day 1 core, tương đương 0.0012s/host/ngày cho 10K alerts — negligible vs SLM.

### B1. ORTHRUS timing (log `P1/Output/orthrus_run.log:224`)
- Total 1799.83s trên máy thầy: build_graphs 256s, embed_nodes 182s, embed_edges 474s, gnn_train 251s, gnn_test 241s, eval 110s, tracing 281s
- Graph construction 230 TWs train (19-21/09) ~1.1s/TW, inference per TW ~1s — không phải bottleneck cho tier2 (tier2 chỉ chạy trên 10²–10³ alerts).

### B1. SLM estimates (GGUF Q4_K_M, chưa đo thật, cần calibrate)
`P1/Output/hw_grid_partial.csv` — prefill vs decode tách riêng per `Note.md:149`:

| size | tok | mode | p50 ms | p95 ms | weight MB | KV MB | dec/day 1c | dec/day 4c contended |
|------|-----|------|--------|--------|-----------|-------|------------|----------------------|
| 0.5B | 128 | single | 461 | 576 | 320 | 0.06 | 187k | 450k |
| 0.5B |2048 | single |1613 |2016 | 320 | 1.0 | 53k |128k |
| 1.5B | 128 | single | 730 | 912 | 900 | 0.15|118k |284k |
| 1.5B |2048 | single |3034 |3792 | 900 | 2.4 |28k |68k |
| 3B   | 128 | single |1440 |1800 |1800|0.31 |60k |144k |
| 3B   |2048 | single |6240 |7800 |1800|5.0 |13k |33k |
| 8B   |2048 | single |14528|18160|4800|12.0|5.9k|14k |
| 3B   | 512 | CoT(128) | ~6000 |7500 |1800|1.25|14k |35k |

- Token là trục chi phối: 1.5B@2048 chậm hơn 8B@128 (3s vs 3s) — đúng `Note.md:55`.
- KV cache tách riêng, không phải weight: 3B@2048 KV 5MB + weight 1.8GB, total ~1.8GB <16GB RAM OK nhưng giảm throughput.
- Pareto partial: 0.5B-128 tối ưu throughput, 3B-512 là upper bound khả thi cho tier2 10³/ngày trên 4C. 8B-2048 chỉ 14k/ngày 4C → vi phạm tier2 budget nếu host bận.

### B1. Decision/host/ngày khả thi
`Note.md:56` yêu cầu số này. Với 4C contended:
- Nếu tier1 sinh 100 alerts/ngày → 0.5B/128 cần 0.02s, 3B/2048 cần 7.8s (0.009% CPU) → vẫn OK
- Nếu tier1 sinh 1000 alerts/ngày → 0.5B/128 2.1s, 3B/2048 78s (0.09% CPU) → 3B@2048 bắt đầu đắt, cần 128 tok
→ **Kiến trúc 2 tầng bị ép**: tier1 phải ≤10³/ngày nếu dùng 3B, phù hợp `Note.md:7`.

## 3. Ý nghĩa cho Phase 2 còn lại

- **RQ1a claim**: TF-IDF V2 đã là baseline mạnh (AP 0.254, FP_red 97% recall 0.5). SLM zero-shot cần few-shot/LoRA mới hơn được (`Note.md:111`). Nếu chỉ zero-shot, dự kiến không vượt TF-IDF — cần chuẩn bị kết luận "model nhỏ không làm được zero-shot".
- **RQ3**: Cần đo thật với `llama.cpp` GGUF 0.5B/1.5B/3B để calibrate estimate; hiện là theoretical.
- **RQ1b**: TF-IDF re-rank đã vượt ORTHRUS (6 vs 1 TP @100) — SLM cần stratify theo ATT&CK (`Note.md:38`) để xem gain ở TTP nào.
- **Block tiếp theo**: Tải host 201/501 (cần rclone thầy), chạy SLM 3B int4 thật trên 2000 alerts V2, đo latency prefill/decode, vẽ Pareto + FP reduction curve + Venn.

## 4. Files sinh ra

- `P1/Output/alerts_enriched_v2.jsonl` (2250, 1.4MB → ~2.8MB sau enrich)
- `P1/Output/tfidf_cv_results.json` (leakage vs CV)
- `P1/Output/TIER2_FILTER_CORRECTED.json` (filter vs re-rank)
- `P1/Output/slm_tier2_v2_cv.json` (verify fix)
- `P1/Output/hw_grid_partial.json` + `.csv` (TF-IDF measured + SLM estimates)
- `P1/Code/slm_tier2.py` patched (prompt + CV + gt_json)

## 5. Next steps (đề xuất)

1. Calibrate HW: download `Qwen2.5-0.5B/1.5B-Q4_K_M.gguf` + `llama-bench` đo thực p50/p95 trên i5-10300H.
2. Run SLM real: `python slm_tier2.py --alerts_jsonl alerts_enriched_v2.jsonl --gt_json gt_and_scores.json --alert_k 2000` với `Qwen2.5-3B int4` (cần transformers + CUDA hoặc llama.cpp).
3. Enrich full 10K alerts (resume từ 2250, dùng cache — không cần DB).
4. Rclone host 201/501 khi có quyền thầy.

