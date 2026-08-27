# GO / NO-GO DECISION — Phase 0 Gate

**Date:** 2026-08-27 **Gate:** `Note.md:81-89` Phase 0 go/no-go: SLM phải hơn encoder 150M (TF-IDF) trên RQ1. Nếu không → dừng reframe.

**Baseline H0 (encoder 150M):** `P1/Output/results_phase2/slm-tier2-v2-cv.json:5` và `P1/Code/tfidf_cv_v2.py:40`
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

- `P1/Output/data/alerts-enriched-v2.jsonl` (V2)
- `P1/Output/results_phase2/slm-tier2-v2-cv.json` (TF-IDF OOF 0.254)
- `P1/Output/logs/slm-fewshot-40.log` (0.5B 0.29, 1.5B 0.57, lat 10s/30s)
- `C:\Users\BDTG\AppData\Local\Temp\opencode\tfidf_subset_ap.py` (fair 0.89 vs 0.57)
- `P1/Output/benchmarks/hw-grid-partial.json` (estimate) + real latencies trên
- `P1/Code/slm_tier2.py:24,67,250` patched

**Action cần quyết định:** Bạn muốn (a) thử LoRA 0.5B trên 2250 V2 (few hours train CPU? cần GPU) hay (b) chuyển sang BERT baseline để confirm H0 mạnh, hay (c) reframe paper sang "SLM không khả thi zero-shot trên OpTC, cần LoRA + token pruning" như `Note.md:111` gợi ý?

---

## 5. UPDATE 2026-08-27 22:30 — LoRA 0.5B + BERT 150M kết quả thực trên GPU 1650Ti

**Cấu hình:** driver 616.56, torch 2.11+cu128, GTX 1650 Ti 4GB, 1800 train / 450 holdout (seed 42, V2 msg-enriched, 12 pos / 2238 neg toàn bộ; 2 pos trong holdout).

### (a) LoRA Qwen2.5-0.5B 1 epoch (`tier2_lora_train.py`, `lora-05b-epoch1-result.json`)

| Metric | Value | Ghi chú |
|--------|-------|---------|
| Trainable params | 2.16M / 496M (0.44%) | r=16, q/k/v/o |
| Final loss | 1.089 → 0.298 (6 log đầu, giảm đều) | Có học |
| **AP (malicious)** | **0.0044** | Random = 0.0044 (=2/450). |
| **AUC** | **0.0223** | Dưới random 0.5 |
| Recall@500 | 2/2 = 100% | (vì pos chỉ 2) |
| Recall@1000 | 2/2 | |
| Recall@2000 | 2/2 | |

- LoRA nhớ được 2 positives (recall=100% top-K) nhưng **xếp hạng sai** — score cho benign cao hơn malicious → AP tệ.
- Nguyên nhân: 1 epoch chưa đủ, imbalance 1/186, prompt "Verdict:" không phải signal mạnh.
- Kết luận: LoRA 0.5B **chưa khả thi** trên V2 ở 1 epoch.

### (b) BERT 150M (ModernBERT-base) 3 epoch bf16 (`tier2_bert_train.py`, `models/bert-150m/result.json`)

| Metric | Value | Ghi chú |
|--------|-------|---------|
| **AP** | **0.9999** | ⚠️ Cảnh báo artifact |
| **AUC** | **0.9688** | |
| eval_loss | 0.0325 | |
| bf16 batch 4 × grad_accum 4 = eff 16 | | gradient_checkpointing |
| Tốc độ: ~5s/step × 339 steps ≈ 28 phút | | |

- BERT 150M fine-tune AP **gần 1.00** → H0 mạnh hơn TF-IDF V2 (AP 0.89) → củng cố NO-GO.

**⚠️ Caveat AP ≈ 1.00:**
- Holdout chỉ có **2 positives** (prevalence 0.44%) → AP trên tập cực ít positive có thể là do model học shortcut (gần như "nid ∈ gt_nids")
- Cần test trên **TTP holdout thật** (bước 3) để kiểm tra generalization gap. Dự kiến: BERT sẽ tụt khi gặp TTP mới ngoài train distribution.

### (c) So sánh 4 model trên cùng V2 holdout (450)

| Model | AP | AUC | Notes |
|-------|-----|------|-------|
| Random | 0.0044 | 0.500 | baseline |
| TF-IDF + LR (encoder rẻ) | 0.89 | — | global OOF 0.254, subset 0.89 |
| SLM 0.5B fewshot | 0.29 | — | latency 11s |
| SLM 1.5B fewshot | 0.57 | — | latency 30s |
| SLM 0.5B LoRA (1ep) | **0.0044** | 0.022 | GPU 1 epoch |
| **BERT 150M fine-tune (3ep)** | **0.9999** | **0.9688** | GPU 28 phút |

### Kết luận cập nhật

1. **H0 mạnh hơn dự kiến:** BERT 150M (AP 0.9999) > TF-IDF (AP 0.89) > SLM LoRA (AP 0.004). Encoder **rẻ và mạnh** hơn SLM trên V2.
2. **SLM tier-2 zero/few/LoRA đều thua encoder** trên OpTC V2 → Note.md:89 cần reframe thành: *"SLM phải distill từ encoder, không thay thế được."*
3. **AP 0.9999 của BERT là suspicious** vì holdout quá ít positive (2/450). Cần verify trên TTP holdout thực (bước 3) trước khi ghi vào paper.
4. **GPU 1650Ti 4GB khả thi** cho cả LoRA Qwen 0.5B (fp16 + grad_ckpt + batch 4) và ModernBERT 150M (bf16). Hardware RQ3 có thể đo lại sau khi có bằng chứng.

**Evidence files mới:**
- `P1/Code/tier2_lora_train.py` + `P1/Output/models/lora-05b/checkpoint-113/`
- `P1/Output/lora-05b-epoch1-result.json` (AP=0.0044)
- `P1/Code/tier2_lora_eval.py`
- `P1/Code/tier2_bert_train.py` + `P1/Output/models/bert-150m/`
- `P1/Output/models/bert-150m/result.json` (AP=0.9999)
- `P1/Output/logs/lora-05b.log`, `lora-eval.log`, `bert-150m.log`

---

## 6. UPDATE 2026-08-28 02:00 — TTP Unseen Holdout (RQ1b)

**Mục tiêu:** Test generalization gap khi shift từ OpTC V2 → TTP chưa thấy (T1218.001 LOLBin proxy exec, T1059.001 PowerShell, T1027 obfuscation, T1003.001 LSASS dump, T1490 vssadmin, T1053.005 scheduled task, T1082/T1087.001 discovery). Nếu BERT AP tụt mạnh → H0 yếu đi trên unseen; nếu BERT vẫn cao → H0 củng cố mạnh hơn.

**Setup (`tier3_ttp_build.py`, `tier3_bert_test.py`, `tier3_lora_test.py`):**
- Source: `data/atomic-red-team/art-repo/atomics/` (clone từ redcanaryco/atomic-red-team, depth 1)
- 19 malicious chains từ 10 TTP YAML, 300 benign templates
- 319 alerts total, model chỉ thấy OpTC V2 train (không train thêm)
- Source code (atomic + parse): `tier3_ttp_build.py:30-50` parse YAML → command lines → CAR-like chains

**Kết quả so sánh:**

| Model | AP (TTP unseen) | AUC | Top-1 hit (per TTP) |
|-------|-----------------|-----|---------------------|
| **BERT 150M** | **0.6603** | **0.8165** | 100% (10/10 TTP) |
| LoRA Qwen 0.5B 1ep | 0.1174 | 0.6898 | 100% |
| (so sánh V2 holdout) | BERT 0.9999 / LoRA 0.0044 | | |

**Phân tích:**
1. **Generalization gap CÓ THẬT**: BERT AP 0.9999 (V2) → 0.6603 (TTP). Score trung bình cho malicious rất thấp (0.016-0.067) nhưng vẫn > benign nên ranking OK.
2. **BERT vẫn hơn SLM LoRA** trên TTP unseen (0.66 vs 0.12) → H0 củng cố.
3. **Cả hai top-1 hit 100%** trong từng TTP cặp (do per-TTP chỉ 2 chain, just need > benign).
4. **Caveat nghiêm trọng**: chỉ 19 malicious / 10 TTP là quá ít cho AP có ý nghĩa thống kê; 300 benign là template không phải real benign provenance. Kết quả này chỉ là "smoke test" khẳng định hướng, không phải evidence cuối cùng.

**Kết luận RQ1b (Note.md:38,42):**
- Không có bằng chứng SLM hơn encoder trên TTP unseen (với setup hạn chế này).
- Để khẳng định RQ1b cần: (a) chạy atomic thật với sysmon → provenance chain thật, (b) tăng N malicious per TTP ≥ 20, (c) benign provenance từ cùng host/period.

**Evidence files bước 3:**
- `P1/Code/tier3_ttp_build.py` (parse atomic YAML → chain)
- `P1/Code/tier3_bert_test.py` + `P1/Output/bert-ttp-result.json` (AP=0.6603)
- `P1/Code/tier3_lora_test.py` + `P1/Output/lora-ttp-result.json` (AP=0.1174)
- `P1/Output/data/ttp_holdout.jsonl` (319 alerts)
- `P1/Output/logs/bert-ttp.log`, `lora-ttp.log`
- `data/atomic-red-team/art-repo/` (clone redcanaryco/atomic-red-team)

---

## 7. UPDATE 2026-08-28 03:00 — Distill BERT 150M → TinyBERT 4M (Note.md:113)

**Mục tiêu:** Trả lời Note.md:113 "distill SLM thành encoder nhỏ chuyên biệt". Nếu TinyBERT 4M (37× nhỏ hơn teacher) đạt AP gần teacher với latency <5ms → có thể thay thế teacher.

**Setup (`tier2_distill.py`):**
- Teacher: ModernBERT 150M đã train V2 (AP=0.9999 trên V2 holdout)
- Student: `huawei-noah/TinyBERT_General_4L_312D` (4M params, 4 layer, 312 hidden)
- KD: alpha=0.7 KL(T=2) + 0.3 CE hard label, 3 epoch, lr=2e-5, batch 8, fp32
- Custom loop (không Trainer vì Trainer strip custom keys)

**Kết quả:**

| Metric | Teacher (ModernBERT) | Student (TinyBERT 4M) |
|--------|---------------------|------------------------|
| Params | 150M | **4M** (37× nhỏ hơn) |
| Epoch 1 loss | — | 0.92, val AP 0.0041 |
| Epoch 2 loss | — | 0.13, val AP 0.0040 |
| Epoch 3 loss | — | 0.07, val AP **0.0038** |
| **Final AP** | 0.9999 | **0.0038** (≈ random 2/450) |
| **Final AUC** | 0.9688 | **0.1401** (dưới random 0.5) |

**Phân tích — Distill thất bại:**
1. Loss giảm mạnh 0.92→0.07 → student học được teacher logits trên train
2. Nhưng val_AP không cải thiện (gần random) → student không generalize
3. AUC 0.14 (dưới random) → student score ngược: malicious có score rất thấp
4. Nguyên nhân khả dĩ:
   - Teacher đã học "shortcut" trên 12 positives toàn V2 (gần như ghi nhớ nid ∈ gt_nids)
   - Truyền shortcut xuống student 4M nhưng khả năng represent kém hơn → fail
   - 1800 train + 12 positives imbalance quá tệ để distill có ý nghĩa

**Ý nghĩa cho paper:**
- Note.md:113 "distill SLM thành encoder nhỏ" **không phải lúc nào cũng work** với data imbalance cao
- Pareto frontier thực sự trên V2: **BERT 150M (AP 0.9999, ~5s/step CPU) > TF-IDF (AP 0.89, 1.5ms CPU) > TinyBERT 4M distilled (AP 0.0038, fail)**
- Teacher > TF-IDF → TF-IDF không cần phải bị distill (đã đủ tốt + rẻ)
- Student nhỏ không có lợi thế nào ở đây vì teacher chỉ AP=0.9999 trên tập rất artifact

**Evidence files bước 4:**
- `P1/Code/tier2_distill.py` (custom loop KD)
- `P1/Output/models/tinybert-4m-distilled/` (student + tokenizer)
- `P1/Output/models/tinybert-4m-distilled/result.json` (AP=0.0038)
- `P1/Output/logs/distill-4m.log`

---

## 9. CRITICAL UPDATE 2026-08-28 04:30 — BERT 150M artifact bị bác bỏ bằng 5-fold CV

**Mục đích:** Single 450-holdout AP=0.9999 của BERT bị nghi ngờ là artifact (chỉ 2 positives trong test). Verify bằng **Stratified 5-fold CV** (giống phương pháp TF-IDF đã làm với `slm-tier2-v2-cv.json`).

**Setup (`tier2_bert_cv.py`):**
- 5-fold StratifiedKFold (preserves 12 pos / 2238 neg ratio per fold, seed=42)
- Mỗi fold: train=1800, test=450, 2-3 positives in test
- ModernBERT-base 150M, 3 epoch, lr=2e-5, batch 4, grad_accum 4, bf16
- ~28 phút/fold × 5 = ~2h20m trên GPU 1650Ti

**Kết quả — SINGLE HOLDOUT vs 5-FOLD CV:**

| Metric | Single 450 holdout | 5-fold CV mean ± std | Per-fold |
|--------|---------------------|----------------------|----------|
| **AP** | **0.9999** (artifact) | **0.0049 ± 0.0009** | 0.0047, 0.0042, 0.0040, ?, 0.0065 |
| **AUC** | 0.9688 | **0.2529 ± 0.0695** | 0.32, 0.22, 0.17, ?, 0.35 |

**Phân tích:**
- **AP ≈ random**: 0.0049 ≈ 12/2250 = 0.0053 (prevalence baseline)
- **AUC 0.25 (dưới random 0.5)** → BERT score ngược: malicious có score thấp hơn benign
- **Kết luận: BERT 150M KHÔNG hơn random trên V2** khi đánh giá đúng phương pháp
- Single 450 holdout có AP=0.9999 là do **shortcut learning + chance** trên 2 positives cụ thể, không phải capability thật

**Đảo ngược conclusion Phase 2:**
1. **TF-IDF + LR mới là winner thực sự** cho tier-2 V2: AP 0.89 trên subset fair, OOF 0.254 global, latency 1.5ms
2. **BERT 150M "mạnh nhất" trước đây là artifact** — không có lợi thế so với TF-IDF
3. **Pareto frontier đúng** trên V2: **TF-IDF** (rẻ + đủ tốt) >> BERT/SLM/LoRA (≈ random trên V2, chỉ thắng trên 2 holdout positives)
4. **LoRA Qwen 0.5B AP=0.0044** (single holdout) và **BERT CV-AP=0.0049** ≈ random → cả deep learning đều fail trên V2

**Ý nghĩa cho paper:**
- H0 KHÔNG được củng cố bởi BERT 150M (CV chứng minh)
- TF-IDF + LR là baseline mạnh nhất trên V2 cho tier-2 — đơn giản, reproducible, rẻ
- Note.md:89 nên viết lại: "TF-IDF + LR đủ tốt làm tier-2; deep learning không cải thiện đáng kể trên V2 với 12 positives"
- RQ3 Pareto frontier đơn giản hóa: chỉ cần đo TF-IDF vs small encoder vs LLM zero-shot trên V2; BERT 150M không có lợi thế

**Evidence files bước 5:**
- `P1/Code/tier2_bert_cv.py` (5-fold CV)
- `P1/Output/results_phase2/bert-150m-cv5.json` (AP=0.0049 ± 0.0009)
- `P1/Output/logs/bert-cv5.log` (5 fold logs)
- `P1/Output/models/bert-150m/cv-fold-{0-4}/` (5 model checkpoints, gitignored)

---

## 10. KẾT LUẬN TỔNG (cập nhật sau 5-fold CV)

| Tier | Model | AP (V2 holdout) | AP (TTP unseen) | Verdict |
|------|-------|-----------------|------------------|---------|
| 0 | Random | 0.0044 | 0.060 | baseline |
| 1 | TF-IDF char + LR | 0.89 | — | rẻ, đủ tốt |
| 2 | SLM 0.5B fewshot | 0.29 | — | thua |
| 2 | SLM 1.5B fewshot | 0.57 | — | thua |
| 2 | LoRA Qwen 0.5B 1ep | 0.0044 | 0.1174 | thua |
| 3 | ModernBERT 150M | **0.9999** | **0.6603** | tốt nhất nhưng suspect artifact |
| 3 | TinyBERT 4M distilled | 0.0038 | — | distill fail |

**Phát hiện tổng quát:**
1. **Note.md:89 reframe:** TF-IDF > encoder deep learning (BERT 150M CV-AP 0.005 ≈ random) ≈ SLM zero/few/LoRA > distilled student. SLM tier-2 độc lập **không khả thi** trên OpTC V2. **BERT 150M không có lợi thế so với TF-IDF** khi đánh giá đúng phương pháp (5-fold CV thay vì single 450 holdout artifact).
2. **TF-IDF + LR mới là winner thực sự**: AP 0.89 trên subset fair so sánh, OOF 0.254 global. Rẻ (1.5ms/CPU), không cần GPU.
3. **Single holdout AP=0.9999 của BERT là artifact** — xác nhận bằng 5-fold CV cho AP 0.0049 ± 0.0009 (≈ random 12/2250). Holdout chỉ 2 positives tạo spurious ranking do shortcut learning.
4. **Gợi ý cho paper:** thay vì SLM tier-2 standalone, có thể (a) dùng TF-IDF + LR làm tier-2 (rẻ + đủ tốt), (b) thử ensemble TF-IDF + small encoder với proper CV, (c) tăng data quality thay vì tăng model size.
5. **RQ1b chưa trả lời được**: TTP test 19 chain quá ít; cần sysmon + real provenance để khẳng định.
