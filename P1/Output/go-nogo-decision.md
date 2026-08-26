# GO / NO-GO DECISION — Phase 0 Gate

**Date:** 2026-08-27 **Gate:** `Note.md:81-89` Phase 0 go/no-go: SLM phải hơn encoder 150M (TF-IDF) trên RQ1. Nếu không → dừng reframe.

**Baseline H0 (encoder 150M):** `P1/Output/slm_tier2_v2_cv.json:5` và `P1/Code/tfidf_cv_v2.py:40`
- V2 msg-enriched, 2250 alerts (12 pos / 2238 neg, prevalence 0.53%)
- TF-IDF char_wb(2,5) 50k + LogReg balanced, **5-fold OOF (không leakage)**
- **Global AP 0.254**, holdout 0.52, fold AP mean 0.32±0.27
- Đây là trần H0 phải vượt. Leak cũ 1.00 là ảo.

---

## 1. Thử nghiệm SLM (1) HW calibrate + (2) AP so sánh

### (1) HW calibrate — đo thật trên i5-10300H 4C/8T 16GB (không phải estimate)

| Model | Tok in | Lat p50 | Lat p95 | Mean | Dec/day 1c | Dec/day 4c contended | Đo bằng |
|-------|--------|---------|---------|------|------------|----------------------|---------|
| TF-IDF (H0) | ~500c | 1.23 ms | 3.04 ms | 1.5ms | 70M | 168M | `hw_grid_benchmark.py:12` measured 200 alerts |
| 0.5B-Instruct fewshot | 906 (GT) / 200 (benign) avg 365 | 10.7s | 13.9s | 11.0s | 8.6k | 20k | `slm_go_nogo_fewshot.py:40` 40 alerts |
| 1.5B-Instruct fewshot | 1092 | 28.8s | 40.2s | 30.1s | 3.0k | 7k | same 40 alerts |
| 3B-Instruct | — | timeout 30m (download 6GB) | — | est 60s | ~1.4k | 3.5k | not measured, extrapolated |

- **Estimate cũ `hw_grid_partial.json` sai 15-20×**: 0.5B est 0.46s vs real 10.7s, 1.5B est 0.73s vs real 28.8s. Nguyên nhân: CPU float32 không có FlashAttention, KV cache lớn, swap.
- **Kết luận RQ3:** Token là trục chi phối nhưng latency thực gấp 10× dự đoán. Tier2 với 1.5B@2048 chỉ làm được **~3k decisions/day 1c** (≈7k 4c contended). Nếu tier1 sinh 100 alerts/ngày → 50 phút/ngày (3.5% CPU) cho 1.5B, 1000 alerts → 8.3h (35% CPU) → **không khả thi**. TF-IDF làm 1000 alerts trong 1.2s.

### (2) SLM vs TF-IDF AP — fair comparison cùng subset

**Sai lầm ban đầu:** So SLM subset AP (0.57) vs TF-IDF global AP (0.254) → GO ảo.

**Fair (cùng 40 alerts, 12 GT / 28 benign, prevalence 30%, seed 42, fewshot 2048, 10 events):** `C:\Users\BDTG\AppData\Local\Temp\opencode\tfidf_subset_ap.py`

| Model | AP (same 40) | Training | Note |
|-------|--------------|----------|------|
| TF-IDF global OOF scores (trained on 2250) | **0.8905** | 5-fold OOF trên 2250, lấy score cho 40 | `tfidf_subset_ap.py:30` |
| TF-IDF subset 3-fold OOF (trained on 40) | **0.9155** | 3-fold trên 40 | overfit nhẹ nhưng vẫn cao |
| SLM 0.5B fewshot | 0.2904 | zero-shot+fewshot 2 ex, 2048 tok | `slm_fewshot_40.log:12` |
| SLM 1.5B fewshot | **0.5708** | same | `log:45` |

**Ngay cả model mạnh nhất 1.5B (0.57) thua TF-IDF 0.89 với margin 0.32.** 0.5B thua xa (0.29 vs 0.89).

**Zero-shot 0.5B** trước đó (`slm_05b_n200.log`) cho AP 0.30 nhưng tất cả `uncertain` (score 0.5) → AP = prevalence, không học gì. Fewshot cải thiện nhưng vẫn thua.

**Confusion @threshold 0.5 (40 subset):**
- 0.5B: TP 11 FP27 FN1 prec 0.289 rec 0.917 → FP reduction 3.6% (27/28 FP vẫn giữ)
- 1.5B: TP10 FP27 FN2 prec 0.27 rec 0.833 → FP reduction 3.6%
- TF-IDF @global threshold 0.5 trên 2250: TP6 FP51 prec 0.10 rec 0.5 FP_red 97% — TF-IDF loại FP tốt hơn SLM rất nhiều dù cùng giữ ~50% GT.

**3B không đo được** (download timeout 30m, `test_3b.py:1`), nhưng với trend 0.5B→1.5B AP tăng 0.29→0.57, 3B có thể ~0.65-0.70, vẫn dưới 0.89. Latency sẽ ~60s → càng không khả thi.

---

## 2. Verdict: NO-GO (theo `Note.md:89`)

> "Nếu không loại trừ được H0, toàn bộ kế hoạch phải viết lại."

- **H0 KHÔNG bị loại:** Encoder rẻ TF-IDF vượt SLM 0.5B/1.5B fewshot trên cùng phân phối (0.89 vs 0.57). `Note.md:15` yêu cầu SLM hơn encoder **ở một trong hai** chỗ (zero/few-shot unseen TTP hoặc disambiguation FP khó). Hiện tại zero/few-shot thua, và FP disambiguation cũng thua (FP_red 3% vs 97%).
- **Lý do gốc:** V2 msg vẫn là `svchost.exe | cmd: None` (phần lớn cmd là None), không có obfuscation/LOLBin thực để SLM phát huy `Note.md:9` "ngữ nghĩa tổ hợp". Dataset OpTC H051 2250 top alerts chủ yếu là Idle/System svchost, không phải encoded PowerShell.
- **HW blocker:** Latency thực 10-30s/alert → Pareto frontier sụp đổ. `Note.md:55` đúng nhưng underestimate 20×.

**Đây không phải lỗi SLM tuyệt đối, mà Phase 0 gate nói đúng: SLM ≤8B zero/few-shot không cạnh tranh được với encoder 150M trên data này.**

---

## 3. Reframe đề xuất (thay vì dừng hẳn)

Per `Note.md:7` "Sau (đứng được)": SLM là tầng 2 trên 10²–10³ candidate, mua (a) FP reduction nhờ hiểu ngữ nghĩa, (b) TTP unseen.

Với NO-GO hiện tại, 3 hướng reframe:

1. **Đổi H0 sang BERT 150M fine-tune** (thay TF-IDF). TF-IDF char n-gram đã rất mạnh trên OpTC do node IDs và path lặp lại; cần test ModernBERT fine-tune 150M xem có sập không. Nếu BERT ~0.6 AP thì H0 còn mạnh hơn.

2. **Tập trung RQ1a với LoRA, không zero-shot.** `Note.md:111` "Nếu paper chỉ có zero-shot thì kết luận là model nhỏ không làm được — yếu." Thử LoRA fine-tune 0.5B/1.5B trên 1800 alerts, test 450 holdout. Đây là adaptation trục `Note.md:51` `{zero, few, LoRA}`. Dự đoán LoRA có thể vượt TF-IDF.

3. **Đổi dataset sang TTP holdout thực sự:** OpTC 2250 không có obfuscation; cần Atomic Red Team / Splunk Attack Range tạo T1218/T1059.001/T1027 (`Note.md:38`) để test RQ1b unseen TTP — nơi encoder sẽ fail, SLM có prior.

4. **Giảm token + distill:** Dùng 128 tok (thay 2048) bằng cách chỉ lấy `self_label + 5 events` thay vì 10, latency giảm ~50% (0.5B từ 10s → ~5s). Hoặc distill SLM thành encoder nhỏ chuyên biệt.

**Không nên làm tiếp:** Enrich 10K alerts (`CONTEXT:85`) hay tải host 201/501 cho RQ5 trước khi qua gate — vô nghĩa per yêu cầu của bạn.

---

## 4. Evidence files

- `P1/Output/alerts_enriched_v2.jsonl` (V2)
- `P1/Output/slm_tier2_v2_cv.json` (TF-IDF OOF 0.254)
- `P1/Output/slm_fewshot_40.log` (0.5B 0.29, 1.5B 0.57, lat 10s/30s)
- `C:\Users\BDTG\AppData\Local\Temp\opencode\tfidf_subset_ap.py` (fair 0.89 vs 0.57)
- `P1/Output/hw_grid_partial.json` (estimate) + real latencies trên
- `P1/Code/slm_tier2.py:24,67,250` patched

**Action cần quyết định:** Bạn muốn (a) thử LoRA 0.5B trên 2250 V2 (few hours train CPU? cần GPU) hay (b) chuyển sang BERT baseline để confirm H0 mạnh, hay (c) reframe paper sang "SLM không khả thi zero-shot trên OpTC, cần LoRA + token pruning" như `Note.md:111` gợi ý?
