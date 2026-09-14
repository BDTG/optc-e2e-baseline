# NOTE — P2 trục giải thích SLM (12/09, gồm mọi yêu cầu thầy)

## 1. Yêu cầu thầy → trạng thái
| Yêu cầu | Trạng thái |
|---|---|
| Nhãn ATT&CK chuẩn thay EVTX (gốc) | ✅ AD-GEN: 10,878 attack, 6,522 GT TTP per-chain |
| GT TTP từng chain (ttp_scored=0 trước đây) | ✅ 704 (12 TTP × 43–209) |
| Metric hallucination phạt né | ✅ rule 2 chiều, hal đo được |
| Constrained decoding (hết parse_fail) | ✅ vượt: logprob scoring, parse_fail = 0 theo cấu trúc |
| 3 mức đặt trước (0.70 / 0.60 / <0.60) | ✅ **rơi mức Âm** (TTP 0.29/0.30/0.12) |
| Không random split, N đủ (≥vài chục/TTP) | ✅ split LAB→REAL; TTP <20 GT thì skip |
| Kỳ vọng acc >75% | ✅ verdict tier-2 OpTC **0.775** (xval); ❌ TTP xa 0.60 |
| P2 (không phải P3), chỉ SLM acc+hal, bỏ TF | ✅ output `slm-explain-*.json` chỉ acc + hal |
| Chạy 0.5B + 1B + 1.5B, phải có số | ✅ cả 3 (1B = TinyLlama-Chat, Llama-1B gated) |
| Code lên git + README (cách chạy, link model) | ✅ đã push; README mục P2 generic |

## 2. Link (model + data)
- 0.5B: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- 1.1B: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
- 1.5B: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Data AD-GEN: https://huggingface.co/datasets/namhop88/AD-GEN → đặt vào `P1/Output/data/adgen/LAB/LAB.jsonl` + `REAL/REAL.jsonl`, chạy `python P1/Code/convert_adgen.py`

## 3. Cách chạy (Python 3.12, transformers==4.41.2; AMD thêm `set TORCH_BLAS_PREFER_HIPBLASLT=0`)
```bash
python P1/Code/explain_05b.py            # bench 704 (lặp lại cho _11b, _15b)
python P1/Code/explain_bal_05b.py        # balanced-200 (lặp lại cho _11b, _15b)
python P1/Code/summarize_explain.py && python P1/Code/summarize_bal.py
```
Output: `P1/Output/results_phase2/slm-explain-{05b,11b,15b,all}.json` và `slm-explain-bal-*.json`
(ví dụ: `{"model": "...0.5B...", "n": 704, "verdict_acc": 0.9389, "ttp_acc": 0.2884, "hal": 0.0}`).
FT head: script trong `P1/Code/p3b/` (`a1_full.py`, `a2.py`).

## 4. Số (chỉ acc + hal)
| Setup | verdict_acc | ttp_acc | hal |
|---|---|---|---|
| 0.5B / 704 | 0.939 | 0.288 | 0.0 |
| 1.1B / 704 | 1.000 | 0.296 | 0.0 |
| 1.5B / 704 | 0.263 | 0.124 | 0.0 |
| 0.5B / balanced-200 | 0.51 | 0.185 | 0.0 |
| 1.1B / balanced-200 | 0.50 | 0.185 | 0.0 |
| 1.5B / balanced-200 | 0.515 | 0.077 | 0.0 |
| 0.5B / 3-shot | 0.101 | 0.040 | 0.0 |

## 5. Kết luận
- Mức Âm cả 3 size. Số pos-only là prior artifact (balanced về ~0.5).
- Thêm example phá luôn verdict (0.94→0.10) → prompt hết đường. Size không cứu (1.5B tệ nhất).
- Gốc là domain shift: random 0.70 → LAB→REAL 0.32; FT head + adapt lên 0.47 (A1/A2). A2 multi-seed: 0.4633 ± 0.0106 (3 seeds) vs TF 0.4591 — mean nhỉnh +0.0042, CI95 [0.437, 0.490] chứa TF, 2/3 seed thắng → không khác biệt có ý nghĩa.
- LoRA-113 vô dụng (base cho số y hệt) — đã bỏ.

## 6. Teacher + audit mới (12-13/09)
- Teacher Qwen2.5-7B scoring XONG: vs GT verdict 0.04/TTP 0.10 (tệ hơn cả student) — to hơn không dạy được; agreement cao nhất ở 15B (0.73 verdict) chỉ vì cùng thiên BENIGN.
- Giám khảo Muse Spark 40 mẫu: vs GT 0.30/0.225; student-vs-judge cao nhất 15B verdict 0.60 (cùng thiên BENIGN), TTP agree 0.05–0.15.
- Audit: narrative leak tag `[hint:*]`; student input sạch 0 hint; dual-use (Deep Freeze/bcdedit) chặn trên mọi acc-vs-GT.
- Human-eval pack 40 blind + key + teacher reference: sẵn sàng gửi người chấm.

## 7. Việc 2 xong (LLM-judge blind, 200 ratings)
- 5 judges (spark/gpt/grok/deepseek/qwen3.8-flash): slm05 mean 1.365 vs t7b 1.317; wins 55 vs 59 — HÒA.
- SLM 0.5B giữ được năng lực giải thích ở 1/14 kích thước. Chi phí ~$0.35.

## 8. Còn lại
- Chờ duyệt: full-FT 1.5B/3B trên CUDA, hoặc chốt Âm chuyển P4 (+B1/B2).
- Chạy lại: xem README mục P2. Code + số: đã push GitHub.

## 9. Tài liệu liên quan trong repo (không chép lại để NOTE ngắn)
- Cấu trúc repo, setup end-to-end P0/P1, restore từ patches: xem README mục 1, 3, 8.
- Kết quả P0 (Flash/Magic) + P1 (Velox): README mục 2, 4, 5.
- Chi tiết P2 đầy đủ (bảng section 8/9/10/10b, mindmap, draft chương): `P1/Output/results_phase2/`.

## hal-that (phuong an b, N=40, 3 judges blind, 3-shot)
- Compliance 3-shot (pack 40): 0.5B 38/40 (95%) | 1.1B 40/40 | 1.5B 40/40 | 7B 40/40. (1-shot: 0.5B 21/40 = 52.5% — vi du BENIGN cuu format)
- Groundedness (G:0-2): 0.5B 0.258 | 1.1B 0.075 | 1.5B 0.425 | 7B **0.807** → hal-content 0.871 / 0.963 / 0.788 / **0.597**.
- So v1 (1-shot, 21 cap): 0.5B 0.079→0.96, 7B 0.574→0.71. Teacher dan moi judge; hal-rule 0.0 BO (vong tron).
- Ket luan deploy: free-gen bia ca 4 he → ship constrained only (evidence trich xuat that, parse_fail 0).

## Viec 3 — CPU on-device (i5-10300H, fp32, 14/09)
- p50/p95/p99 (may roi): 0.5B 21.4/24.7/28.4s | 1.1B 59.6/78.4/78.4s | 1.5B 131.5/170.8/170.8s. May ban (burn 4): 0.5B p50 96.1s.
- Burn 1.1B/1.5B: khong do duoc tren may 4-core (>5-10 phut/mau khi co tai nen — CPU inference bandwidth-bound). Ket qua: duoi tai that chi 0.5B tru duoc.
- Throughput/ngay (roi): 4,045 / 1,449 / 657. Ban (0.5B): 899. 39 alert/ngay x 21.4s ≈ 14 phut (roi) / 62 phut (ban).
- RAM: weight 2.1/4.5/6.2 GB; peak process 6.1/6.1/9.7 GB; KV@512 12.6/23.1/29.4 MB (@2048: 50.3/92.3/117.4 MB).
- Ket luan: chi 0.5B kha thi endpoint; 1.5B weight 6.2GB + peak 9.7GB vuot endpoint pho thong.
