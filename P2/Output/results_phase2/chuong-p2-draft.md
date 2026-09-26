# Chương P2 — SLM tier-2 trên trục giải thích (bản draft báo thầy)

## 1. Định vị
- Tier-2 chạy **sau detector đồ thị tầng 1** (cascade 2250→39 alerts, prev 23.1%), không phải primary detector.
- Chỉ đo SLM trên **trục giải thích: verdict_acc + ttp_acc + hal**. Không baseline TF (theo ý thầy).
- Dữ liệu: AD-GEN (235,723 narratives Sysmon; 10,878 attack, 6,522 có GT TTP per-chain; synthetic-validated, tác giả Đ.P.Nam/UIT — khai báo nguồn, không claim GT forensic).

## 2. Bốn yêu cầu kỹ thuật của thầy — đã đáp ứng
| # | Yêu cầu | Thực hiện |
|---|---|---|
| 1 | Nhãn ATT&CK chuẩn thay EVTX keyword-giả | AD-GEN, ttp_scored 0 → 704 (12 TTP × 43–209 GT) |
| 2 | GT TTP per-chain | Adapter narrative→chain, bench 709 chain |
| 3 | Metric hallucination phạt né | Rule 2 chiều: né evidence khi chain có bằng chứng = hallucinated |
| 4 | Constrained decoding (hết parse_fail) | Vượt yêu cầu: logprob scoring (teacher forcing) → parse_fail = 0 theo cấu trúc, không cần outlines |

## 3. Kết quả trục giải thích (zero/few-shot) — MỨC ÂM cả 3 size

| Setup | verdict_acc | ttp_acc | hal | Ghi chú |
|---|---|---|---|---|
| 0.5B / 704 pos-only | 0.939 (TP661/FN43) | 0.288 | 0.0 | TTP = đáp án example T1059 |
| 1.1B (TinyLlama-Chat, thay Llama-1B gated) / 704 | 1.000 (toàn MAL) | 0.296 | 0.0 | Prior MAL tuyệt đối |
| 1.5B / 704 | 0.263 (FN519) | 0.124 | 0.0 | Prior BENIGN mạnh |
| 0.5B / balanced-200 | **0.51** (31/71/29/69) | 0.185 | 0.0 | Số pos-only là prior artifact |
| 1.1B / balanced-200 | **0.50** (100/0/100/0) | 0.185 | 0.0 | Score sai (0.64) > đúng (0.61) — tune ngưỡng vô ích |
| 1.5B / balanced-200 | **0.515** (16/87/13/84) | 0.077 | 0.0 | 2 lớp score chồng lấn — yếu thật |
| 0.5B / 3-shot (T1059+T1003+BEN) / 704 | **0.101** | 0.040 (T1003×633) | 0.0 | Thêm example phá luôn verdict |

Đối chiếu khung thầy (TTP ≥0.70 Dương mạnh / 0.60–0.70 Dương / <0.60 Âm): **Âm toàn diện**.

## 4. Phân tích mức Âm (phần publishable)
1. **Few-shot = chép đáp án.** Đổi example T1218→T1059: dự đoán đổi theo 100% (T1218×704 → T1059×656). Loại bias: TTP thật 0/575 trên 11 TTP không-example. 3-shot còn phá luôn verdict (0.94→0.10). → Prompt hết đường, không phải do thiếu shots.
2. **Balanced phơi bày prior.** Pos-only 0.94/1.0 về 0.50–0.52 khi có benign. Score 2 lớp không tách (1.1B đảo ngược) → vấn đề là không đọc nội dung, không phải lệch ngưỡng.
3. **Size không cứu được.** 0.5B→1.1B: TTP 0.29→0.30 (đứng yên). 1.5B tệ nhất (0.12). FT 1.5B trên stack ROCm: full-FT fp16 NaN, LoRA xung đột lib, partial-FT 0.264 < 0.5B full-FT 0.352.
4. **Domain shift mới là gốc (bằng chứng từ FT head).** Random-split 0.70 → LAB→REAL 0.32 (−0.37); TF-IDF rơi gần hệt (−0.33) → gap thuộc dữ liệu. Domain-adapt (train +REAL-sớm) lên 0.47; ensemble SLM+TF-IDF 0.36. Trajectory 0.32→0.36→0.47, vẫn Âm.

## 5. Giá trị còn lại của SLM (kể cả khi Âm)
- **Trục explanation độc quyền:** constrained scoring valid 100%, hal 0.0, 0.2–0.5s/mẫu, chạy cả NVIDIA lẫn AMD ROCm — TF/encoder không sinh giải thích được.
- **Verdict tier-2 OpTC (operating point thật 39 alerts): xval 0.775 > 75%** kỳ vọng thầy.
- **Artifact:** adapter AD-GEN→chain (239,815 chains), protocol bench 709 GT tái dùng được, toàn bộ code + số đã push GitHub.

## 6. Limitations (ghi rõ, không giấu)
- Bench verdict pos-only 704 chưa đo FP cùng split (balanced-200 bù lại, n=200 nhỏ).
- Metric hal hiện đo rule-based evidence, chưa đo năng lực sinh evidence của model.
- Llama-1B gated → thay TinyLlama-1.1B-Chat (khai báo rõ).
- "1b" của thầy nếu chỉ model khác (Qwen3-1.7B cần transformers mới, xung đột ROCm pin) — cần máy CUDA để chạy sạch.
- AD-GEN labels = synthetic LLM-validated, không phải forensic GT.

## 7. Con đường tiếp (xin ý kiến thầy)
1. Domain-adapt sâu hơn + ensemble trên A2 (GPU sẵn, ~3h).
2. Full-FT 1.5B/3B sạch trên máy thầy 5060Ti (CUDA, tách bạch size lần cuối).
3. Chốt Âm, chuyển toàn lực P4 (temporal + threshold transfer + evasion) và B1/B2 (nhãn FP, tập evasion).
