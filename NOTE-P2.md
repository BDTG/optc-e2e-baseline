# NOTE — P2 trục giải thích SLM (12/09)

## Làm gì
- Bench AD-GEN 704 chain GT TTP + balanced-200. harness: constrained scoring (logprob), metric phạt né.
- Chạy 3 size base-sạch: Qwen2.5-0.5B, TinyLlama-1.1B-Chat (thay Llama-1B gated), Qwen2.5-1.5B.

## Số (chỉ acc + hal, không TF)
| Setup | verdict_acc | ttp_acc | hal |
|---|---|---|---|
| 0.5B / 704 | 0.939 | 0.288 | 0.0 |
| 1.1B / 704 | 1.000 | 0.296 | 0.0 |
| 1.5B / 704 | 0.263 | 0.124 | 0.0 |
| 0.5B / balanced-200 | 0.51 | 0.185 | 0.0 |
| 1.1B / balanced-200 | 0.50 | 0.185 | 0.0 |
| 1.5B / balanced-200 | 0.515 | 0.077 | 0.0 |
| 0.5B / 3-shot | 0.101 | 0.040 | 0.0 |

## Kết luận
- Mức Âm cả 3 size (TTP < 0.60), hal 0.0.
- Số pos-only là prior artifact (balanced về ~0.5). Thêm example phá luôn verdict → prompt hết đường.
- Size không cứu được (1.5B tệ nhất). Gốc là domain shift (FT head LAB→REAL: 0.32; +adapt: 0.47).

## Còn lại
- FT head + domain-adapt (A1 0.36 / A2 0.47) — section 9 bảng thầy.
- Chờ duyệt: full-FT 1.5B/3B trên CUDA, hoặc chốt Âm chuyển P4.
- Code + số: đã push GitHub. Chạy lại: xem README mục P2.
