# Ket qua 6.3 (dong 53/55/56/57/58) — chay tren sus (RX 9060 XT 16GB, ROCm)

## Dong 53 — TTP head LAB->REAL, 3 seed (mean_AP, 21 TTP chung, N>=20/TTP)
- TF-IDF char 2-5 + LR (tuned C):        0.2553 ± 0.0035   [head_tfidf.json]
- Qwen2.5-0.5B seq-cls head (bf16):      0.1457 ± 0.0202   [head_slm.json]
- SecureBERT2.0-base head (lr 3e-5):     0.2317 ± 0.0070   [head_securebert.json]
Nhan xet: head SLM/SecureBERT thua TF-IDF khi LAB->REAL (domain shift) — nguoc so voi so tren subset TTP chung cu (0.370 vs 0.348).

## Dong 55 — 7B tham chieu: CHAN
- eve.py run --mode eve / json_enum voi Qwen2.5-7B-Instruct bf16 deu OOM:
  "CUDA out of memory. Tried to allocate 1.05 GiB / 538 MiB (batch 2 va batch 1)".
- Nguyen nhan: slm.py tinh logits toan vocab (152K) cho moi token cua head chuoi dai → ~0.9-1.5GB transient + model 15.3GB > VRAM 16GB.
- Quy dinh: khong patch, khong quant — dung theo yeu cau. Can GPU >= 24GB hoac dong y thay doi de chay tiep.

## Dong 56 — SecureBERT 2.0: TTP + verdict / verdict TF-IDF
- TTP: xem dong 53 (0.2317).
- Verdict SecureBERT: AUC 0.7705, AP 0.8042 (train 11,158 LAB -> test 1,000 REAL)  [verdict_securebert.json]
- Verdict TF-IDF:      AUC 0.7516, AP 0.7953                                    [verdict_tfidf.json]

## Dong 57 — Baseline extractive
[eval_extractive.json] (test 1,000 REAL, dev = q05_LAB_dev_eve)
- extractive (tfidf-lr): ttp1 0.223 (majority 0.240), entailment 0.093, minimal 0.000, verdict AUC 0.767
- eve (0.5B):            ttp1 0.399, entailment 1.000, minimal 1.000, verdict AUC 0.459 (fpr 0.100 voi threshold chon bang dev)

## Dong 58 — Pareto chat luong / latency
[pareto_ttp.png] + [pareto_ttp.csv] — 2 diem (extractive, eve 0.5B), duong baseline majority 0.229.
Diem 7B thieu do dong 55 chan.

## Luu y ky thuat
- head_slm chay theo dung lenh thay (batch 8, grad_ckpt). 7B thu batch 2 roi batch 1 — deu OOM.
- File ket qua nay cung nam trong git repo (optc-explain-pack/results/).
