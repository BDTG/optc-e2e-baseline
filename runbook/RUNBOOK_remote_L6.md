# RUNBOOK — L6 Baseline trên máy khỏe (PIDSMAKER + OpTC 051)

> Mục tiêu: chạy Flash / MAGIC / Velox / ORTHRUS trên DB full 7 ngày (19.8M events)
> với split chuẩn train=19-21 / val=22 / test=23-25 — việc máy 16GB không làm được (OOM).
> Chuẩn bị bởi Hermes (máy BDTG) 2026-08-15. Số pilot subset tham khảo: `LOG.md` P1-41→P1-45.

---

## 0. Yêu cầu phần cứng

| Item | Yêu cầu |
|------|---------|
| RAM | **≥32 GB** (training 19.8M events: word2vec embeds 6-10GB + batching load) |
| Docker | Docker Desktop / Docker Engine + Docker Compose |
| Disk | ≥20 GB trống cho DB + artifacts |
| GPU | Không bắt buộc (`--cpu` OK; GPU chỉ cần nếu chạy SLM tầng hai) |

---

## 1. Cài đặt

```bash
# 1.1 Clone PIDSMAKER (commit chuẩn)
git clone --depth 1 https://github.com/ubc-provenance/PIDSMaker.git
cd PIDSMaker
# checkout đúng commit nếu cần: 2289cd9b0adf7289a093f63ca7ff11a3b97e46c3

# 1.2 Áp patches (đã đóng gói trong patches/)
#   - all_patches.diff  : 8 file code (config, preprocessing, pipeline, utils)
#   - pidsmaker_patch.diff : skip JSON lỗi trong create_database_optc.py
git apply ../patches/all_patches.diff
git apply ../patches/pidsmaker_patch.diff

# 1.3 Thay config.py (đã patch sẵn: database=optc_h051_full, dates chuẩn)
cp ../config/config.py pidsmaker/config/config.py

# 1.4 Build Docker image
docker compose -f compose-postgres.yml up -d
docker compose -f compose-pidsmaker.yml build
docker compose -f compose-pidsmaker.yml up -d
```

**Lưu ý mount**: container `pidsmaker-pids` cần mount:
- DB dumps → `/tmp/dumps/`
- artifacts → `/data/artifacts/`

---

## 2. Nạp database

```bash
# 2.1 Tạo database + schema
docker exec pg-pids psql -U postgres -c "CREATE DATABASE optc_h051_full;"
# schema: chạy init-create-databases.sh HOẶC dùng schema trong config.py (xem dưới)

# 2.2 Restore dump (custom format — nhanh hơn nhiều so với INSERT)
docker cp optc_h051_full.dump pg-pids:/tmp/
docker exec pg-pids pg_restore -U postgres -d optc_h051_full --no-owner /tmp/optc_h051_full.dump

# Verify
docker exec pg-pids psql -U postgres -d optc_h051_full -t -c \
  "SELECT count(*) FROM event_table;"   # kỳ vọng: 19,815,600
```

> Schema lưu trong dump luôn (custom format). Nếu restore lỗi owner, thêm `--no-owner --no-privileges`.

---

## 3. Chạy L6 baseline

### 3.1 Flash (word2vec flash featurization)

```bash
docker exec pidsmaker-pids bash -c "cd /home/pids && mkdir -p /data/logs /data/artifacts && \
PYTHONPATH=/home/pids python -m pidsmaker.main flash optc_h051 --cpu \
  --database_host pg-pids --database_user postgres --database_password postgres \
  --artifact_dir /data/artifacts \
  --evaluation.ground_truth_version orthrus \
  --batching.intra_graph_batching.used_methods none \
  --training.encoder.used_methods none \
  --training.decoder.use_few_shot False \
  --construction.multi_dataset none \
  --featurization.used_method flash \
  --featurization.training_split all \
  --featurization.multi_dataset_training False \
  --batching.multi_dataset_training False \
  2>&1 | tee /data/logs/flash_full_v2.log; echo FLASH_EXIT=\$?"
```

**Thời gian ước tính**: construction ~1-2h, feat_inference ~1-2h, training 12 epochs ~1-2h (CPU). Tổng **~4-6h**.

### 3.2 MAGIC (only_type featurization — nhanh nhất)

```bash
docker exec pidsmaker-pids bash -c "cd /home/pids && \
PYTHONPATH=/home/pids python -m pidsmaker.main magic optc_h051 --cpu \
  --database_host pg-pids --database_user postgres --database_password postgres \
  --artifact_dir /data/artifacts \
  --evaluation.ground_truth_version orthrus \
  --batching.intra_graph_batching.used_methods none \
  --training.encoder.used_methods none \
  --training.decoder.use_few_shot False \
  --construction.multi_dataset none \
  --featurization.used_method only_type \
  --featurization.training_split all \
  --featurization.multi_dataset_training False \
  --batching.multi_dataset_training False \
  2>&1 | tee /data/logs/magic_full_v2.log; echo MAGIC_EXIT=\$?"
```

**Ước tính**: ~2-3h (construction cache chung, only_type không train embedding).

### 3.3 Velox (word2vec 50 epochs — NẶNG NHẤT, lý do cần máy khỏe)

```bash
docker exec pidsmaker-pids bash -c "cd /home/pids && \
PYTHONPATH=/home/pids python -m pidsmaker.main velox optc_h051 --cpu \
  --database_host pg-pids --database_user postgres --database_password postgres \
  --artifact_dir /data/artifacts \
  --evaluation.ground_truth_version orthrus \
  --batching.intra_graph_batching.used_methods none \
  --training.encoder.used_methods none \
  --training.decoder.use_few_shot False \
  --construction.multi_dataset none \
  --featurization.used_method word2vec \
  --featurization.training_split all \
  --featurization.multi_dataset_training False \
  --batching.multi_dataset_training False \
  2>&1 | tee /data/logs/velox_full_v1.log; echo VELOX_EXIT=\$?"
```

**Ước tính**: ~4-6h. **Theo dõi RAM**: nếu `free -m` < 2GB khi training bắt đầu → giảm
`time_window_size` trong flash.yml/velox.yml (15.0 → 30.0) để giảm số graphs.

### 3.4 ORTHRUS (cùng word2vec — chạy sau Velox, construction cache)

```bash
docker exec pidsmaker-pids bash -c "cd /home/pids && \
PYTHONPATH=/home/pids python -m pidsmaker.main orthrus optc_h051 --cpu \
  ... (cùng flags như Velox, đổi model name + featurization.used_method word2vec) ..."
```

---

## 4. Xác minh kết quả

### 4.1 Số liệu chuẩn từ log

```bash
grep -E "precision:|recall:|auc:|tp:|fp:|tn:|fn:" /data/logs/*_v2.log | tail -12
grep -E "Found .*edge labels|Total distinct" /data/logs/*_v2.log
```

**Kỳ vọng (so với pilot subset 3 ngày máy 16GB — P1-41→P1-43):**
- GT edge labels: 135,920/135,928 (pilot subset) — full phải ≥ con số này
- GT nodes time-window: 114/114
- Flash: Mean Loss giảm thật, AUC > 0.678
- MAGIC: tp > 69, recall > 0.602 (nhiều train data hơn)

### 4.2 precision@1hop/2hop (script kèm theo)

```bash
# Sửa SCORES path trong script tới scores_model_epoch_*.pkl của run
docker cp scripts/precision_1hop_2hop.py pidsmaker-pids:/home/pids/
docker exec pidsmaker-pids bash -c "cd /home/pids && PYTHONPATH=/home/pids python precision_1hop_2hop.py"
# Cần: SCORES path đúng, DB optc_051_sub HOẶC sửa database=optc_h051_full
```

### 4.3 Calib threshold (script kèm theo)

```bash
docker cp scripts/calib_threshold.py pidsmaker-pids:/home/pids/
docker exec pidsmaker-pids bash -c "cd /home/pids && PYTHONPATH=/home/pids python calib_threshold.py"
# Output: bảng FPR target -> TP/FP/recall/precision@1hop/@2hop. Điểm vận hành khuyến nghị: FPR 0.1%.
```

---

## 5. Trả về máy BDTG

| File | Nội dung |
|------|----------|
| `flash_full_v2.log` + `results.pth` | Flash full metrics |
| `magic_full_v2.log` + `results.pth` | MAGIC full metrics |
| `velox_full_v1.log` + `results.pth` | Velox full metrics |
| `orthrus_full_v1.log` + `results.pth` | ORTHRUS full metrics |
| `precision_1hop_2hop_*.txt` | Kết quả phân tích |

Hermes (máy BDTG) sẽ cập nhật `baseline_comparison.md` §2/§5 + LOG.md với số liệu chính thức.

---

## 6. Pitfalls đã biết (từ máy 16GB — P1-14→P1-45)

1. **`config/default` phải là `default`** (không có tiền tố `config/`) — `check_args` so sánh chuỗi.
2. **Dotted CLI args** (`--featurization.used_method`) bị argparse nuốt trong vài phiên bản → nếu lỗi config None, set trong YAML thay vì CLI.
3. **`get_yml_cfg(args)` nhận Namespace**, không nhận list — dùng qua `pidsmaker.main`, đừng gọi trực tiếp với list.
4. **Schema khớp**: `event_table` 9 cột (có `edge_label`), node tables có `path/cmd/index_id` — dump đã chứa schema đúng, đừng recreate bằng init-create-databases.sh cũ (thiếu cột).
5. **GT UUID→index**: dùng `uuid_index_map.csv` (113/113 từ DB full). Index trong dump KHỚP với map (giữ nguyên index_id gốc).
6. **OOM pattern**: log 0 bytes + exit 0 = python bị OOM-kill. Xác nhận bằng `/sys/fs/cgroup/memory.events` (oom_kill > 0). Không phải lỗi code.
7. **tee không flush** khi process bị kill — chạy với `python -u` để log đầy đủ.
