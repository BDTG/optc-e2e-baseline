# 🔧 SỔ TAY TỰ VẬN HÀNH — Chạy L6 Baseline trên máy thầy (không cần agent)

> Dành cho trường hợp bạn tự kéo file + tự chạy trên máy thầy (qua UltraViewer hoặc trực tiếp).
> Nếu bạn có SSH thì đưa cho Hermes, sẽ làm tự động; bản này cho đường manual.

---

## ❓ XÁC NHẬN TRƯỚC (trả lời 3 câu này rồi làm)

| # | Câu hỏi | Ảnh hưởng |
|---|---|---|
| 1 | Máy thầy là **Windows hay Linux**? | Lệnh khác nhau |
| 2 | Máy có **Docker** (Desktop/Engine) không? | DB + PIDSMAKER chạy trong Docker |
| 3 | RAM bao nhiêu **≥32GB**? | Velox/ORTHRUS full cần ≥32GB; nếu <32GB chỉ chạy Flash/MAGIC |

> **Máy thầy (đã xác nhận 2026-08-20):** Windows 10 · **không có Docker** · RAM 32GB
> → RAM đủ chạy cả 4 systems (32GB ≥ ngưỡng Velox). Cần **cài Docker Desktop trước** (Bước 0).

---

## BƯỚC 0 — CÀI DOCKER DESKTOP (Windows 10, chưa có Docker)

### 0a. Kiểm tra điều kiện trước (quan trọng — Windows 10 Docker Desktop cần WSL2)
```
# Mở PowerShell (Admin):
systeminfo | Select-String "Hyper-V"        # phải hiện "Hyper-V Requirements: A hypervisor has been detected"
wsl --status                                # xem WSL đã cài chưa
```
- **Bắt buộc**: CPU hỗ trợ **virtualization** (Intel VT-x / AMD-V) và **đã bật trong BIOS**
  (cài máy thầy thường bật sẵn; kiểm tra Task Manager → Performance → CPU → Virtualization: Enabled)
- Windows 10 Pro/Enterprise: cần bật Hyper-V (optional feature) hoặc WSL2 đều được
- Windows 10 Home: **chỉ dùng WSL2** (Hyper-V không có sẵn trên Home) — Docker Desktop tự xử lý

### 0b. Cài WSL2 + Docker Desktop (PowerShell Admin):
```powershell
# 1) Cài WSL2 (kernel mới nhất)
wsl --install
# → khởi động lại máy
# 2) Cài Docker Desktop
#    Tải installer: https://desktop.docker.com/win/main/amd64/DockerDesktopInstaller.exe
#    Chạy, chọn backend "WSL 2", bật "Use the WSL 2 based engine"
# 3) Xác nhận hoạt động (đăng nhập user thường):
docker --version          # vd Docker version 29.x
docker run hello-world    # phải chạy được
```
> Nếu `wsl --install` lỗi hoặc cần full ảnh: `wsl --install -d Ubuntu` rồi cài Docker engine
> trong Ubuntu (cách Linux). Nhưng Docker Desktop Win là nhanh nhất cho Windows 10.

### 0c. KHÔNG muốn Docker? → Đường WSL2 + native Linux (khuyên dùng, không cần Docker)

**PIDSMAKER là pure Python + PostgreSQL — KHÔNG bắt buộc Java/Maven.**
→ Không cần Docker Desktop; chỉ cần WSL2 Ubuntu + cài trực tiếp bằng apt/conda.

```powershell
# PowerShell (Admin):
wsl --install -d Ubuntu     # cài WSL2 + Ubuntu; khởi động lại máy
wsl -d Ubuntu               # mở shell Ubuntu
```

```bash
# Trong Ubuntu:
sudo apt update && sudo apt install -y postgresql postgresql-contrib git wget
# khởi động postgres + tạo user/db
sudo service postgresql start
sudo -u postgres psql -c "CREATE USER postgres SUPERUSER LOGIN PASSWORD 'postgres';" -c "CREATE DATABASE postgres;"
# Python 3.9 (conda hoặc apt python3.9)
#   - nhanh nhất: cài conda (Miniconda) rồi: conda create -n pids python=3.9 -y
conda create -n pids python=3.9 -y && conda activate pids
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn==1.2.0 networkx==2.8.7 xxhash==3.2.0 graphviz==0.20.1 psutil scipy==1.10.1 matplotlib==3.8.4 wandb==0.24.1 chardet==5.2.0 nltk==3.8.1 igraph==0.11.5 cairocffi==1.7.0 wget==3.2 psycopg2-binary tqdm
```
> Đúng là bản deps từ `Dockerfile` upstream (có cả torch CPU). Path `/home/pids` giữ nguyên
> như container → các patch chạy đúng ngay, không cần đổi path.
> Sau bước này → BƯỚC 1 (kéo code) → BƯỚC 3 (restore DB — host localhost) → BƯỚC 5 (chạy).
> **Chạy trực tiếp trong Ubuntu, bỏ wrapper `docker exec`** — lệnh BƯỚC 5 dạng:
> `cd /home/pids && PYTHONPATH=/home/pids python -m pidsmaker.main flash optc_h051 --cpu --database_host localhost ...`
> (cùng flags, chỉ đổi `--database_host pg-pids` → `localhost`, bỏ phần `docker exec ... bash -c "..."`)

---

## BƯỚC 1 — Kéo CODE (từ GitHub, ~400KB)

```bash
# Windows (PowerShell/cmd) hoặc Linux đều được
git clone https://github.com/BDTG/optc-e2e-baseline.git
cd optc-e2e-baseline
```

*> Nếu máy thầy chưa có git: Windows → cài Git for Windows; Linux → `sudo apt install git`*

---

## BƯỚC 2 — Kéo DATA (file lớn, KHÔNG có trên GitHub)

Cần 2 thứ: **DB dumps** (để test nhanh) hoặc **raw** (để build lại), tùy mục đích.

### Lựa chọn A — Nhanh nhất: dùng DB dumps (đã build sẵn bởi máy BDTG)
| File | Dung lượng | Dùng để |
|---|---|---|
| `optc_h051_full.dump` | 1.13 GB | Baseline full 051 (19.8M events) ⭐ chính |
| `optc_051_sub.dump` | 504 MB | Pilot verify nhanh (3 ngày) |
| `optc_h201.dump` / `optc_h501.dump` | *(chưa build)* | 2 host còn lại |

*Nhận từ BDTG qua USB / Drive / LAN. Sau đó bỏ qua Bước 3, nhảy tới Bước 4.*

### Lựa chọn B — Build lại từ raw (cần Docker)
Mac có raw? Kiểm tra:
| Host | Raw cần |
|---|---|
| H051 | `phase1/data/raw/host051_*` (56GB) |
| H501 | `phase1/data/raw/host501_*` (31GB) |
| H201 | `phase1/data/raw/host201_*` (13GB, mới tải đủ) |

Cách A là chuẩn nhất cho bạn — dumps đã có từ trước (`C:\Users\BDTG\OpTC-data\phase1\handoff_remote\dumps\`).

---

## BƯỚC 3 — Restore DB (nếu dùng dump)

```bash
# Nếu máy thầy có Docker: khởi động Postgres container trước
docker run -d --name pg-l6 -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:17

# Restore (Windows)
docker cp optc_h051_full.dump pg-l6:/tmp/
docker exec pg-l6 createdb -U postgres optc_h051_full
docker exec pg-l6 pg_restore -U postgres -d optc_h051_full --no-owner /tmp/optc_h051_full.dump

# Verify
docker exec pg-l6 psql -U postgres -d optc_h051_full -t -c "SELECT count(*) FROM event_table;"
# Kỳ vọng: 19,815,600
```

---

## BƯỚC 4 — Dựng PIDSMAKER (Docker image)

```bash
# Clone upstream PIDSMAKER
git clone https://github.com/ubc-provenance/PIDSMaker.git
cd PIDSMaker

# Áp 2 patch (bắt buộc — xem README phần 'Why the patches are required')
git apply ../optc-e2e-baseline/patches/all_patches.diff
git apply ../optc-e2e-baseline/patches/pidsmaker_patch.diff

# Thay config
cp ../optc-e2e-baseline/config/config.py pidsmaker/config/config.py

# Build + up (theo hướng dẫn của PIDSMAKER)
docker compose -f compose-postgres.yml up -d
docker compose -f compose-pidsmaker.yml build
docker compose -f compose-pidsmaker.yml up -d
```

> Lưu ý: ví dụ trên dùng `docker run pg-l6` đơn giản; nhưng PIDSMAKER kéo theo compose có
> sẵn. Đọc `runbook/RUNBOOK_remote_L6.md` để khớp tên container/network với compose chính thức.

---

## BƯỚC 5 — Chạy 4 Baseline (lệnh đầy đủ trong RUNBOOK)

**Flash** (~4-6h) — **MAGIC** (~2-3h) — **Velox** (~4-6h, ≥32GB) — **ORTHRUS** (~4-6h)

Mỗi lệnh dạng:
```bash
docker exec pidsmaker-pids bash -c "cd /home/pids && \
PYTHONPATH=/home/pids python -m pidsmaker.main flash optc_h051 --cpu \
  --database_host pg-pids --database_user postgres --database_password postgres \
  --artifact_dir /data/artifacts --evaluation.ground_truth_version orthrus \
  --batching.intra_graph_batching.used_methods none \
  --training.encoder.used_methods none --training.decoder.use_few_shot False \
  --construction.multi_dataset none --featurization.used_method flash \
  --featurization.training_split all --featurization.multi_dataset_training False \
  --batching.multi_dataset_training False 2>&1 | tee /data/logs/flash_full_v2.log"
```
(Đầy đủ 4 lệnh: `runbook/RUNBOOK_remote_L6.md` mục 3)

---

## BƯỚC 6 — Verify & thu thập kết quả

```bash
# 1) GT trong graph (phải đầy đủ)
grep -E "Found .*edge labels|Total distinct" /data/logs/*_v2.log
# Kỳ vọng: 135,920/135,928 edge labels · 114/114 nodes

# 2) Metrics epoch cuối
grep -E "precision:|recall:|auc:|tp:|fp:|tn:|fn:" /data/logs/*_v2.log | tail -12

# 3) Copy artifacts về máy BDTG
```

**Gửi về BDTG:**
- `flash_full_v2.log` + `results.pth`
- `magic_full_v2.log` + `results.pth`
- `velox_full_v1.log` + `results.pth` (nếu chạy được)
- `orthrus_full_v1.log` + `results.pth`
- (Hermes sẽ cập nhật baseline_comparison.md + LOG.md)

---

## 🚨 Pitfalls ghi nhớ (đã đúc kết 30+ lần chạy)

1. **`default`** chứ không phải `config/default` (không có tiền tố path)
2. **Dotted CLI args** đôi khi bị argparse nuốt → nếu lỗi, set trong YAML thay vì CLI
3. **OOM pattern**: log 0 bytes + exit 0 = bị kill bộ nhớ, KHÔNG phải lỗi code.
   Xác nhận: `cat /sys/fs/cgroup/memory.events` (oom_kill > 0)
4. **Dùng `python -u`** khi chạy (tee không flush được khi process bị kill)
5. **Không recreate schema bằng init-create-databases.sh cũ** — dump chứa schema đúng (event_table 9 cột)
6. Khi transfer inspect `sha256sum` cả 2 phía (manifest `phase1/manifest/`)

---
*Nếu kẹt bất kỳ bước nào: chụp lỗi + up lên chat, Hermes chẩn đoán.*
