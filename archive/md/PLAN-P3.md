# PLAN — P3 tiếp theo (sau RQ1a)

## 1) Mặt bằng hiện tại (tổng hợp P0 → P3)

- **P0/P1 đóng gói xong:** Velox recall@10K 88.6% (101/114), cascade 2250 → 39, gate PASSED.
- **P2 đóng gói xong:** SLM ≤1.5B constrained/logprob, JSON 4 trường `{verdict, mitre_technique, evidence, recommended_action}`, parse_fail = 0. SLM 0.5B hòa teacher 7B ở trục giải thích (55 vs 59 wins, 5 judges blind, 200 ratings).
- **P3/RQ1a (mới nhất — chi tiết mới):**
  - SLM 0.5B zero-shot trên cascade output thật (12 mal GT OpTC + 192 benign): verdict acc 0.9412 nhưng **TP = 0/12 — model trả BENIGN gần như constant** (scores thống nhất −2.27, TTP dự đoán đồng nhất T1082).
  - **Phát hiện chính:** rule cấu trúc đơn giản **"parent_chain ≥ 2 nodes"** đạt **100% recall (12/12)**, FP chỉ 9/192 (4.7%). Mal chain trung bình 2.9 nodes; benign 180/192 chỉ có 1 node ("Idle | cmd: None").
  - Nghĩa là: trên dữ liệu cascade thật, **tín hiệu cấu trúc (chain length) mạnh hơn tri thức ngôn ngữ của SLM 1B**.
- **Kết quả nền khác đã đo:** SecBERT verdict 0.936 (thắng TF-IDF 0.826); INT8 / GGUF-Q4 sụp chất lượng (0.09–0.26);

## 2) RQ1b — phương án finetune SLM trên cascade thật (kế tiếp)

Mục tiêu: biến SLM từ "chỉ kể chuyện" thành "kể chuyện ĐÚNG trên phân bố cascade thật".

- **Data:** 12 mal GT OpTC (quá ít) + 319 ttp_holdout ( chains template, 300 mal / 19 ben) để tăng số lượng
  - Augmentation: 12 mal thật × chuỗi paraphrase + oversampling balanced
  - Split: temporal fix (KHÔNG random), giữ sạch domain shift
- **Cách fine-tune (2 hướng song song):**
  1. **Rule-hybrid:** rule chain ≥ 2 làm tin hiệu cấu trúc, SLM chỉ xử lý phần còn lại (trả lời: "giải thích TTP gì, từ đâu trong chain") — cheap nhất, khớp phát hiện RQ1a
  2. **LoRA full chain-level:** fine-tune 0.5B trên mixed {12 mal thật × oversample + template chains} — cần GPU (GPD 9060XT hoặc máy thầy 5060Ti), ước lượng ~30–60 phút/run cho 3 seed
- **Metric cần đáp ứng (khung thầy đã ±):** Recall ≥ 0.70 (Dương) *hoặc* publishable nếu ≥ 0.60 kèm phân tích hal ≤ 10–15%. N≥20/TTP, temporal split, AUC + Precision@thr + F1, nội suy hal-content (không dùng hal-rule circular).

## 3) RQ2 — Multi-seed & khoảng tin cậy

- A2 domain-adapt 0.47 ≫ TF 0.46 → **chạy 5 seeds** để có mean±std thay vì 1 seed duy nhất (nếu ổn thì tăng sức thuyết phục khoảng "thắng").
- Làm lần lượt trên venv-cpu Nitro (không cần GPU, chi phí thấp), mỗi seed ~25 phút.

## 4) RQ3 — Ensemble cụm nhỏ

- Câu hỏi mở (đã nêu với thầy): nhiều cụm 0.5B hỗ trợ nhau để tăng verdict accuracy thay vì 1 model tự suy luận.
- Lộ trình: 3 instance 0.5B independent → majority-vote logprob → so với 1 instance đơn lẻ + so với teacher 7B. Chi phí ~1 giờ trên GPD.

## 5) RQ4 — Quantization chính thức

- INT8/GGUF-Q4 **sụp chất lượng** (0.96→0.09) → lên án quantized cho scoring product
- Nhưng GGUF-Q8/Q6 chưa đo — làm *backend thủ công tốt hơn* (cam kết giữ acc ≥ 0.90) trước khi tuyên bố GO/NO-GO về quantized. Nếu đo xong vẫn sụp → chốt: fp32 CPU là con đường duy nhất của tier-2.

## 6) Vuông trống — lỗ hổng P3 chưa lấp

- N=40 compliance/groundedness chỉ có 3 hệ (0.5B / 1.1B / 1.5B) — **thiếu teacher-as-judge N=40 nội bộ** (đang chờ router kết nối lại)
- GGUF quality Q4_K_M và Q8_0 chưa so pairwise cùng 1 dataset (chỉ có Q4 = 0.259, Q8 chưa chạy)
- Multi-seed A2 là số — chưa đi vào bảng.

## 7) Bảng việc cụ thể (thứ tự ưu tiên)

| # | Việc | Nơi chạy | Thời lượng | Output kỳ vọng |
|---|---|---|---|---|
| 1 | **RQ1b-hybrid** (rule-chain + SLM) | GPD CPU | ~30p | verdict acc dự kiến >95% |
| 2 | **RQ1b-LoRA** 3 seeds | GPD 9060XT (hoặc 5060Ti thầy) | ~1h | mean±std acc + TTP |
| 3 | RQ2 multi-seed A2 | Nitro CPU | ~2h | 0.47 ± std |
| 4 | RQ3 ensemble 3×0.5B | GPD CPU | ~1h | majority-vote acc vs 1×0.5B |
| 5 | RQ4 GGUF Q8_0 | GPD GPU | ~30m | go/no-go quantized |
| 6 | Word/tổng hợp cuối | Nitro | ~1h | bảng tổng + note thầy |

## 8) Kỳ vọng đóng Paper 2

- Sau bước 1–5: **P3 đủ cấu trúc RQ1a → RQ1b → RQ2 → RQ3 → RQ4** khép trọn phương pháp đánh giá.
- Phần viết: methodology (P2 + P3), bảng so sánh rule vs SLM vs teacher, phân tích hal-content, khuyến nghị deployment (fp32 CPU trước — quantized có rủi ro sụp).
- Deliverable cho thầy: 1 Word tổng hợp + 1 HTML kết luận 1 trang (theme xanh) + git tag.
