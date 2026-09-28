# EVE: Entailment-Verified Explanation

Đặt thư mục `eve/` vào `P2/Code/eve/`. Chạy mọi lệnh từ trong thư mục này.

Yêu cầu: `torch`, `transformers` (4.4x hoặc 5.x), `scikit-learn`, `tokenizers`.

## File

| File | Vai trò |
|---|---|
| `tech_preconditions.json` | Cơ sở tri thức φt: 10 technique, 28 clause |
| `kb.py` | Kiểm φt, tính C(E), witness, rút gọn tối thiểu, kiểm entailment và counterfactual, đánh dấu field do hệ thống ghi nhận (attested) |
| `slm.py` | Chấm logprob có cache prefix, chat template, sinh văn bản |
| `eve.py` | `run`: 5 mode (`eve`, `anchored`, `json_enum`, `json`, `free`). `kbstats`: thống kê KB, không cần model |
| `eval_eve.py` | Verdict AUC/FPR, TTP so với majority, evidence so với hint/gold, bootstrap CI |
| `conformal.py` | Tập technique có bảo đảm phủ, bản weighted cho LAB→REAL, phân rã lỗi (Định lý 3) |
| `make_splits.py` | Tạo tập đánh giá cân theo độ dài, bản ngẫu nhiên, dev, calib, nontrivial, kèm manifest |
| `inject.py` | RQ4: tạo dữ liệu bị chèn chuỗi tấn công, so tỉ lệ bị lật |
| `tests/` | `test_kb.py` (không cần model), `smoke.py` (đầu-cuối với model tí hon) |

## Kiểm tra cài đặt

```
python tests/test_kb.py
python tests/smoke.py
```

## Thứ tự chạy

**0. Tạo tập đánh giá (cân độ dài, chống lối tắt bẫy 7):**
```
python make_splits.py --data ../../../P1/Output/data/adgen-v2.jsonl --outdir ../../../P1/Output/data/splits
```
Kiểm `splits_manifest.json`: `auc_length_events` của các tập `*_matched` phải gần 0.5, của `REAL_test_random` thường cao hơn (minh chứng bẫy 7). Tạo ra:
`REAL_test_matched` (RQ1, RQ3, RQ4), `REAL_test_random` (bẫy 7), `REAL_nontrivial` (evidence RQ1), `LAB_dev_matched` (chọn ngưỡng verdict), `LAB_calib` (conformal RQ2).


Dữ liệu vào là file của `convert_adgen_v2.py` (`adgen-v2*.jsonl`).

**1. Thống kê KB (E3, chưa cần model).** Xem `kb_recall_in_scope`, `benign_nonempty_C`, `per_technique` để chỉnh φt:
```
python eve.py kbstats --data ../../../P1/Output/data/adgen-v2.jsonl --out results/kbstats.json
```

**2. Chạy nhỏ để kiểm tra (100 mẫu):**
```
python eve.py run --data <balanced> --model Qwen/Qwen2.5-0.5B-Instruct --mode eve --out results/q05_eve_test.jsonl --limit 100
```
File `.summary.json` phải có `verdict_tie_rate` = 0 và `entailment_ok_rate_among_predicted` = 1.0.

**3. RQ1: 5 mode × model (E4).**
```
for m in eve anchored json_enum json free; do
  python eve.py run --data <balanced_REAL> --model Qwen/Qwen2.5-0.5B-Instruct --mode $m --out results/q05_REAL_$m.jsonl --resume
done
python eval_eve.py --results results/q05_REAL_*.jsonl --data <balanced_REAL> --dev results/q05_LAB_eve.jsonl --out results/eval_q05.json
```
Thêm `--gold gold.jsonl` khi có nhãn tay. Mỗi dòng gold có dạng `{"sample_id", "techniques": [...], "evidence": [{"event", "field"}]}`.

**4. RQ2: conformal (E6).** Chạy `eve` trên LAB (calib) và REAL (test):
```
python conformal.py --calib results/q05_LAB_eve.jsonl --test results/q05_REAL_eve.jsonl --weighted --calib_data <LAB> --test_data <REAL> --out results/conformal_q05.json
```
Chỉ số cần xem: `coverage_inscope` ≥ `target_inscope`, `bound_holds` = true, `knowledge_miss`, `model_miss`.

**5. RQ4: injection (E8).**
```
for p in instruction claim_technique fake_fields keyword_stuffing; do
  python inject.py make --data <balanced_REAL> --out data_inj_$p.jsonl --payload $p
  python eve.py run --data data_inj_$p.jsonl --model <model> --mode eve --out results/q05_eve_inj_$p.jsonl
  python inject.py compare --clean results/q05_REAL_eve.jsonl --injected results/q05_eve_inj_$p.jsonl --out results/inj_$p.json
done
```
Lặp lại với `--attested_only` (bản chỉ dùng field do hệ thống ghi nhận) và với mode `free`/`json` để so sánh.

## Tham số chính

- `--dtype float32` (mặc định; không dùng fp16).
- `--batch_size` (giảm xuống 4 cho 1.5B nếu thiếu RAM).
- `--threads`.
- `--max_witnesses 24`, `--max_spans 48`, `--topk_evidence 2`.
- `--attested_only`.
- `--resume`.
- `--attack_map` (dùng `attack_id_map_v15.json` để kiểm ID technique của mode sinh).
