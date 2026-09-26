# Chương P3 — Đánh giá phương pháp SLM tier-2 trên cascade thật (bản draft báo thầy)

*Bản draft tiếp nối chuong-p2-draft.md. Chương P2 đóng ở trạng thái "Âm toàn diện trên AD-GEN + verdict xval 0.775". Chương P3 trả lời câu hỏi: khi đưa SLM đặt đúng operating point (cascade output thật), tier-2 chịu được compose tốt đến đâu — và cách nào làm nó tốt hơn/thực tế hơn.*

## 1. Định vị + 5 câu hỏi đánh giá (RQ)

## 1. Định vị + 5 câu hỏi đánh giá (RQ)

- Input chính của chương này là **dữ liệu cascade thật** — 2250 alerts enriched của OpTC (detector tầng 1 output thật) với 12 mal GT + 192 benign; AD-GEN (bench 704) giữ vai trò **đối chứng thống nhất** cho phần quantization (RQ4/RQ4b), để so sánh cùng script với các định dạng.
- Minh bạch nguồn: nhãn vẫn là **synthetic AD-GEN** (Đ.P.Nam, UIT) cho bench quantization; 12 mal verdict là GT OpTC thật.
- 5 RQ khép vòng đánh giá: **RQ1a** rule cấu trúc vs SLM zero-shot · **RQ1b** hybrid + LoRA · **RQ2** ngân sách train (multi-seed) · **RQ3** ensemble cụm nhỏ · **RQ4** quantization go/no-go (kèm **RQ4b** ternary Bonsai-2 27B).

## 2. RQ1a — Rule cấu trúc thắng SLM zero-shot trên cascade thật

Subset 12 mal GT + 192 benign cascade (alerts-enriched-v2, chain thật cho từng nid).

| Hệ | Recall (mal 12) | FP (benign 192) | Precision | F1 |
|---|---|---|---|---|
| SLM 0.5B zero-shot (constrained) | **0/12** | 0/192 | — | — (constant-BENIGN, score −2.27 đồng nhất, TTP dự đoán đồng nhất T1082) |
| Rule "parent_chain ≥ 2 nodes" | **12/12** | 9/192 (4.7%) | 0.571 | 0.727 |

Phân tích: mal chain trung bình 2.9 node; 180/192 benign chỉ có 1 node ("Idle | cmd: None") — cascade FP về cơ bản là chain rỗng. SLM 0.5B zero-shot nói "BENIGN" an toàn, không đọc được tín hiệu cấu trúc.
**Đọc method:** trên cascade thật, tín hiệu cấu trúc (chain length) đi trước tri thức ngôn ngữ của SLM ≤1B. Rule được đưa làm **pre-filter** ở RQ1b — không phải là kết quả cuối, mà là tầng 0 của hybrid.

## 3. RQ1b — Hybrid pre-filter + LoRA fine-tune 0.5B

**Hybrid (rule → SLM soi dị nghi):** chỉ 21/204 mẫu cần SLM (12 + 9 FP), auto-clean 183 → **tiết kiệm 89.7% compute**; rule-only F1 0.727.

**LoRA r8 trên Qwen 0.5B** (train + save_pretrained + eval trong 1 lần chạy; eval prompt == train prompt từng ký tự):

| Config | n_train | TP/FP | Precision | Recall | F1 |
|---|---|---|---|---|---|
| v2 (400 ex, imbalance 19 ben) | 400 | 0/0 | — | 0 | 0 — constant-BENIGN (score −2.942) |
| v3 (199 ex, imbalance 19 ben) | 199 | 12/9 | 0.571 | 1.0 | 0.727 = rule-only |
| **v4 (balanced 84 mal / 76 benign: chain-trắng + holdout)** | 160 | **11–12/5–6** | 0.706 | 1.0 | **0.759 (chung) / 0.828 (seed 43)** |
| v5 (benign nhiễu THẬT 869 chain≥2 + holdout) | 187 | 3/3 | 0.5 | 0.25 | 0.333 |

Quy tắc rút ra (có chẩn đoán): imbalance >70/30 → constant-verdict (mất cảnh báo là lỗi lầm); LoRA cực nhạy prompt — eval prompt khác train 1 ký tự là scoring gãy. v2/v3 là các lần thất bại định hướng (bảng bên trên); v4 là config tối ưu.

**v5 = bằng chứng delimitation quan trọng nhất của chương:** khi benign thực sự cũng chain≥2 (svchost/WMI/services thật), feature chain≥2 mất ý nghĩa phân biệt và recall sụp 1.0 → 0.25. **0.5B không đủ capacity tách benign-noise-thật khỏi mal-chain — ceiling của tier-2 là thuộc tính phân bố dữ liệu, không phải thiếu data.**

## 4. RQ2 — Ngân sách train quyết định dấu kết luận (multi-seed A2)

Cùng data/split domain-adapt (LAB + REAL-sớm → REAL-muộn), chỉ budget thay đổi:

| Budget | Kết quả (5 seeds) | vs TF-IDF 0.4591 |
|---|---|---|
| Reduced: 1 epoch × 256 tok | **0.3976 ± 0.0425** (seed 44 outlier 0.3269) | Thua |
| Full: 2 epochs × 512 tok (seed 42) | **0.4733** | **Thắng** |

Đọc method: hình ảnh dấu (+/−) của A2 chạy **theo ngân sách train**, không phải theo "SLM tốt hơn n-gram". Report cả 2 vế, cấm trộn budget trong 1 bảng. Rủi ro vận hành đã ghi nhận: torch-ROCm fallback CPU im lặng (đã fix bằng TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL + fail-fast — tránh 3.2h/seed CPU ảo).

## 5. RQ3 — Ensemble 3×0.5B (majority-vote) thua single best

3 adapter (seed 42/43/44), majority-vote: **F1 0.750** — thấp hơn seed 43 đơn lẻ (0.828).
Voter degenerate (seed 44 constant-MALICIOUS) kéo cụm về MALICIOUS.
**Đọc method:** khi voter degenerate (constant-verdict), đa dạng mô hình không còn lợi thế — ensemble chỉ có lợi khi các voter mắc lỗi độc lập (variance); ở đây lỗi là bias hệ thống — cộng thêm mô hình chỉ nhân thêm cùng một bias.

## 6. RQ4/RQ4b — Quantization go/no-go (output trục quant của chương)

Cùng script + cùng bench 704 + cùng endpoint chat (thinking off qua chat_template_kwargs) trên 4 định lượng:

| Định dạng | Verdict acc | TTP acc | s/mẫu |
|---|---|---|---|
| fp32 (transformers, control) | **0.939** | 0.288 | 21.4 (CPU) |
| INT8 torch dynamic | 0.092 | 0.144 | 2.3 |
| GGUF Q4_K_M 4B | 0.259 | 0.047 | 0.37 (GPU) |
| GGUF Q8_0 4B | 0.256 | 0.060 | 0.49 (GPU) |
| **Bonsai-2 27B ternary PTQ1_0** (PrismML fork, Vulkan) | **0.2926** | 0.081 | 1.37 (GPU) |

- RQ4 (Q4/Q8/INT8): tăng bit-width không cứu — mọi định dạng dưới fp32 sụp về 0.09–0.26.
- **RQ4b chốt câu hỏi "định dạng tốt hơn có cứu được không": KHÔNG.** Bonsai-2 27B là **format ternary hiện đại nhất hiện có** (trit −1/0/+1 + Hadamard g128, claim 98.2% fp16 trên benchmark chuẩn, 5.95GB). Chỉ đạt **0.2926** — hơn chút Q4/Q8 4B và khác hẳn fp32 0.939. Constant-MALICIOUS vẫn còn (TN = 0).
- **Kết luận NO-GO: quant collapse là thuộc tính của quantization trên scoring fine-grained, không phải RTN thô** (ternary recomposed + Hadamard không cứu). fp32 CPU 0.5B là con đường duy nhất của tier-2.
- Công nghệ setup (bằng chứng tái hiện): fork PrismML branch `prism` (9a4dd89) — release prebuilt b10660 không đọc được ternary type 143; PQ2_0 segfault; chạy trên máy mới **sus** (CachyOS, RX 9060 XT Vulkan, 7800X3D): decode 29 t/s GPU, 1.7 t/s CPU-only.
- Limitation cần nêu rõ: prompt bench GGUF (chat + thinking-off trong 1 prompt) KHÔNG giống hệt cấu hình pipeline fp32 (constrained logprob) — có thể kết luận "quant lộ kém hơn" nhưng mức chênh tuyệt đối vs fp32 phải đọc với điều kiện này.
- 1 câu cho review: "n-gram offload kiểu Qwen3.8 Flash-Next 177B không phải giải pháp tier-2" — kiến trúc đó là architecture-level (core 125B + 51B lookup-table stream từ NVMe), core weights 45.8GB vẫn không fit CPU on-device; citing trong limitation, không phải future work của mình.

## 7. Bảng đối chiếu khung thầy (3 mức Dương/Âm)

| Trục | Kết quả P3 | Mức theo thầy |
|---|---|---|
| Recall rule pre-filter (cascade thật) | 12/12 (100%) FP 4.7% | **Dương mạnh** (≥0.70) |
| F1 hybrid (v4 seed 43) | 0.828 | **Dương mạnh** |
| F1 hybrid khi benign thật (v5) | 0.333 | **Âm có phân tích** — ceiling tier-2 |
| A2 full-budget | 0.4733 &gt; TF 0.4591 | Dương (margin mỏng) |
| A2 reduced-budget (5 seeds) | 0.3976 ± 0.0425 &lt; TF 0.4591 | Âm |
| Ensemble | 0.750 &lt; 0.828 single | Âm |
| Quantized (INT8→ternary) | 0.09–0.29 ≪ fp32 0.939 | **Âm — NO-GO chốt** |

## 8. Limitations & phản biện (tường minh)

1. N=12 mal thật (chỉ đủ chứng minh recall, chưa đủ ước lượng precision) — benign 192 chỉ theo cascade thật, chưa phủ đa dạng host.
2. Bench quant dùng AD-GEN template (không phải cascade thật), và prompt format khác pipeline fp32 — so sánh quan trọng là **tương đối giữa các định dạng cùng script**, không phải so tuyệt đối vs fp32. (Đa dạng benign chưa phủ đa dạng host.)
3. Bonsai-2: chỉ checkpoint PTQ1_0 chạy ổn; PQ2_0 segfault — chưa quét được toàn bộ không gian ternary (per-format speed trade-offs bỏ ngỏ).
4. Tóm lại nguyên tắc rút ra rõ nhất: SLM/chip nhỏ chỉ ngang khi đúng prompt + đủ compute — chương chốt protocol đánh giá 3 điều kiện (fp32 control + benign thật + đúng prompt). Đây là **protocol để đánh giá mọi SLM tier-2** — đóng góp chính của chương.

## 9. Kết luận P3 + khuyến nghị deployment

- **Deployment (chốt):** fp32 0.5B CPU constrained (1 nhân 25.5s/alert; RAM đỉnh 6.5–7GB; dưới 0.3% máy 8 nhân/ngày). Rule chain≥2 = tầng 0 pre-filter, không dùng model quantized cho scoring.
- **Đóng góp chương:** (1) protocol đánh giá 3 điều kiện; (2) delimitation v5 — benign thật chốt roofline tier-2; (3) NO-GO quant toàn thế hệ hiện có (RTN→ternary); (4) tách bạch "budget train vs model quality".
- **Còn mở (→P4/P5):** burst threshold transfer + evasion (B1/B2 nhãn FP), RQ5 unseen-host (72 benign miễn phí) — trong Note.md phase P4.

---
*File data → số: results_phase3/ (rq1a-subset-result, rq1b-final, rq2-final, rq3-final, rq4-final, rq4b-bonsai-ptq10). Full bảng: full-benchmark-table.html mục 18a–18g.*
