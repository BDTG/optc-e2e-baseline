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
- Gốc là domain shift: random 0.70 → LAB→REAL 0.32; FT head + adapt lên 0.47 (A1/A2).
- LoRA-113 vô dụng (base cho số y hệt) — đã bỏ.

## 6. Còn lại
- Chờ duyệt: full-FT 1.5B/3B trên CUDA, hoặc chốt Âm chuyển P4 (+B1/B2).
- Chạy lại: xem README mục P2. Code + số: đã push GitHub.
