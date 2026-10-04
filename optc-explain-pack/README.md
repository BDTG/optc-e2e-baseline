# EVE: Entailment-Verified Explanation

Đặt thư mục `eve/` vào `P2/Code/eve/`. Chạy mọi lệnh từ trong thư mục này.

Yêu cầu: `torch`, `transformers` (4.4x hoặc 5.x), `scikit-learn`, `tokenizers`.

## File

| File | Vai trò |
|---|---|
| `tech_preconditions.json` | Cơ sở tri thức φt: 10 technique, 28 clause |
| `kb.py` | Kiểm φt, tính C(E), witness, rút gọn tối thiểu, kiểm entailment và counterfactual, đánh dấu field do hệ thống ghi nhận (attested) |
| `slm.py` | Chấm logprob có cache prefix, chat template, sinh văn bản |
| `eve.py` | `run`: 6 mode (`eve`, `anchored`, `json_enum`, `json`, `free`, `kb_only`; `kb_only` không cần `--model`). `kbstats`: thống kê KB, không cần model |
| `eval_eve.py` | Verdict AUC/FPR, TTP so với majority, evidence so với hint/gold, bootstrap CI |
| `conformal.py` | Tập technique có bảo đảm phủ, bản weighted cho LAB→REAL, phân rã lỗi (Định lý 3) |
| `make_splits.py` | Tạo tập đánh giá cân theo độ dài, bản ngẫu nhiên, dev, calib, nontrivial, kèm manifest |
| `attackdata_to_eve.py` | Chuyển log Sysmon của Splunk attack_data sang schema adgen-v2 (nhãn technique độc lập, STT 43) |
| `inject.py` | RQ4: tạo dữ liệu bị chèn chuỗi tấn công, so tỉ lệ bị lật. Payload thích ứng `attested_rename`, `attested_instruction` nhắm vào `Image` (field attested) |
| `make_attack_map.py` | Tạo `attack_id_map_v15.json` (ID technique ATT&CK v15.1 hợp lệ) cho `--attack_map` |
| `conformal_route.py` | Định tuyến theo tập conformal: giữ SLM nhỏ khi tập có 1 technique, escalate lên model lớn khi còn lại. `--lat_small`/`--lat_big` lấy latency từ file đo trên CPU |
| `paired_bootstrap.py` | Bootstrap CI ghép cặp TTP top-1 giữa hai hệ thống (+ McNemar exact), tùy chọn `--gold` |
| `json_enum_check.py` | Kiểm tra đủ n và phân phối technique dự đoán của file kết quả |
| `inject_persample.py` | Kiểm injection theo từng mẫu: technique đổi thì C(E)/evidence có đổi không, evidence có nằm trên field bị chèn không; tách theo nhãn |
| `attackdata_stats.py`, `attackdata_silver_gold.py` | Mô tả tập attack_data; so nhãn silver (.yml) với gold (atomic test trong cây tiến trình), kappa, tách tree/rare |
| `measure_rss.py` | Chạy một lệnh trong process riêng, lấy mẫu RSS bằng psutil (RAM thật trên CPU) |
| `pareto_svg.py` | Vẽ lại Pareto từ CSV của `pareto_plot.py` thành SVG (nhãn tự tránh chồng, frontier bậc thang, cột tùy chọn `ci_lo`/`ci_hi` để vẽ CI) + xuất PNG qua Edge/Chrome headless |
| `newrun.py` | Tạo thư mục kết quả riêng cho mỗi lần chạy: `results/<dataset>/<tag>/` (+ `logs/`, `run.json` ghi commit của thư mục code `--code` và số file chưa commit) |
| `tests/` | `test_kb.py` (không cần model), `test_parse_gen.py` (parser mode sinh), `smoke.py` (đầu-cuối với model tí hon) |

## Kiểm tra cài đặt

```
python tests/test_kb.py
python tests/smoke.py
```

## Cấu trúc kết quả

Mỗi STT/thí nghiệm ghi vào thư mục riêng, **không ghi thẳng vào `results/`**:

```
results/
  adgen/        00_prep, 00_core, stt38_..., stt39_..., ..., stt58_pareto
  attackdata/   stt43
  optc/         stt44
  _smoke/       smoke test, _superseded/  kết quả cũ bị thay thế, legacy_explain/  explain_*.py gốc
```

Mỗi thư mục run gồm file kết quả, `logs/` và `run.json` (dataset, tag, git commit, thời điểm tạo). Tag đặt theo dạng `stt<số>[a-z]_<mô_tả>`.

```
R=$(python newrun.py adgen stt60_lora_head --note "mo ta ngan")
python eve.py run ... --out $R/q05_REAL_eve.jsonl > $R/logs/eve.log 2>&1
python newrun.py --list
```

Các lệnh bên dưới viết `results/<file>` cho gọn; khi chạy thật, thay `results/` bằng `$R/`.

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

**3b. Ablation không dùng model (bắt buộc cho RQ1).** Cùng C(E), chọn technique bằng luật cố định thay cho SLM. So với `eve` để đo phần đóng góp của SLM:
```
for r in specific first alpha; do
  python eve.py run --data <balanced_REAL> --mode kb_only --kb_rule $r --out results/kb_only_$r.jsonl
done
python eval_eve.py --results results/q05_REAL_eve.jsonl results/kb_only_*.jsonl --data <balanced_REAL>
```
`first` (mặc định): witness ở event sớm nhất. `alpha`: mã technique nhỏ nhất. `specific`: ưu tiên sub-technique, rồi witness nhiều field hơn, rồi event sớm hơn. Xác suất technique chia đều trên C(E). Verdict của `kb_only` là luật thuần: có witness thì malicious.

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
- `--attack_map` (dùng `attack_id_map_v15.json` để kiểm ID technique của mode sinh; tạo bằng `python make_attack_map.py`).
- `--max_new_tokens` (mặc định 384; 192 làm cụt JSON của mode `free`/`json`). Parser mode sinh đọc trường `technique` trong JSON (không lấy ID nằm trong đường dẫn evidence), chịu được code fence quanh JSON, escape `\` sai và JSON bị cắt.

## Dataset độc lập: Splunk attack_data (STT 43)

Nhãn technique lấy từ file `.yml` của từng dataset (theo cách tạo dữ liệu), độc lập với nhãn LLM. Chỉ đo TTP, không có evidence gold.

**1. Tải repo, chưa tải log (log dùng Git LFS):**
```
git lfs install --skip-smudge
git clone --filter=blob:none https://github.com/splunk/attack_data
```
**2. Liệt kê file Sysmon cần tải, rồi tải đúng các file đó:**
```
python attackdata_to_eve.py list --root <attack_data> > lfs_files.txt
```
PowerShell: `cd <attack_data>; git lfs pull --include=((Get-Content ..\lfs_files.txt) -join ',')`

**3. Chuyển đổi, rồi chạy như AD-GEN:**
```
python attackdata_to_eve.py convert --root <attack_data> --out attackdata.jsonl
python eve.py kbstats --data attackdata.jsonl --out results/kbstats_attackdata.json
python eve.py run --data attackdata.jsonl --model <M> --mode eve --out results/q05_attackdata_eve.jsonl
python eve.py run --data attackdata.jsonl --mode kb_only --out results/kb_only_attackdata.jsonl
python eval_eve.py --results results/q05_attackdata_eve.jsonl results/kb_only_attackdata.jsonl --data attackdata.jsonl
```
Kiểm tra `attackdata.jsonl.manifest.json`: `file_stats` (không còn `file_lfs_pointer`), `alerts_per_technique`, `alerts_by_label_source`.

Cách chọn alert: `tree` là cây tiến trình con của lệnh chạy test (Invoke-AtomicTest), dùng cho dataset atomic_red_team; `rare` là tiến trình có cặp Image + CommandLine ít xuất hiện giữa các file, dùng cho dataset khác. Nhãn ở mức file nên có thể có alert không mang dấu vết của technique; đây là hạn chế cần ghi trong paper. Trường `RuleName` của Sysmon (chứa gợi ý technique) bị loại khỏi input của model.
