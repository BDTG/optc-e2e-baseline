#!/usr/bin/env bash
# ============================================================================
# setup_all.sh — OpTC L6 Baseline tự động (chạy trên WSL2 Ubuntu máy thầy)
# Chạy:  bash setup_all.sh [--dump <path-to-optc_h051_full.dump>]
#
# Làm TẤT CẢ:
#   1. Cài gói hệ thống   (git wget postgresql openssh-server ...)
#   2. Cài Miniconda      (Python 3.9 env "pids")
#   3. Cài deps           (torch CPU + sklearn/networkx/igraph/cairocffi ...)
#   4. Clone PIDSMAKER    + áp 2 patches + copy config (database=optc_h051_full)
#   5. Restore DB         từ dump (nếu có --dump) và verify count
#   6. In lệnh chạy 4 systems (Flash/MAGIC/Velox/ORTHRUS)
#
# An toàn chạy lại: idempotent — bỏ qua bước đã xong.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$HOME/optec-l6"
PIDS_DIR="$WORK/PIDSMaker"
PIDS_COMMIT="2289cd9b0adf7289a093f63ca7ff11a3b97e46c3"   # commit patches được tạo từ (HEAD upstream)
DUMP_PATH=""

# ---- parse args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dump) DUMP_PATH="$2"; shift 2 ;;
    *) echo "❌ Lệnh không hiểu: $1"; echo "Dùng: bash setup_all.sh [--dump <file.dump>]"; exit 1 ;;
  esac
done

echo "🛠  OpTC L6 Auto-Setup"
echo "   Work dir : $WORK"
echo "   PIDSMAKER: $PIDS_DIR"
[ -n "$DUMP_PATH" ] && echo "   Dump     : $DUMP_PATH"
echo

sudo -v  # cache sudo credential sớm (hỏi mật khẩu 1 lần)

# ---------------------------------------------------------------- 1. packages
echo "==> [1/6] Cài gói hệ thống..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  git wget ca-certificates curl \
  postgresql postgresql-contrib \
  openssh-server sudo build-essential \
  graphviz maven \
  >/dev/null 2>&1 || {
    echo "   (một số gói đã có hoặc apt thiếu — tiếp tục)"
    sudo apt-get install -y --no-install-recommends git wget postgresql openssh-server >/dev/null 2>&1
  }
echo "   OK"

# ------------------------------------------------------------- 2. Miniconda
if [ ! -d "$HOME/miniconda3" ]; then
  echo "==> [2/6] Cài Miniconda (1 lần)..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  rm -f /tmp/miniconda.sh
else
  echo "==> [2/6] Miniconda đã có."
fi

# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda config --set auto_activate_base false

# ---- accept conda ToS (conda 24.x+ yêu cầu trước khi dùng kênh defaults) ----
echo "==> [2/6] Chấp nhận conda Terms of Service..."
for ch in \
  "https://repo.anaconda.com/pkgs/main" \
  "https://repo.anaconda.com/pkgs/r" \
  "https://repo.anaconda.com/pkgs/msys2"; do
  conda tos accept --override-channels --channel "$ch" >/dev/null 2>&1 || true
done
echo "   OK"

if ! conda env list | grep -q '^pids\s'; then
  echo "==> [2/6] Tạo env python=3.9 'pids'..."
  conda create -n pids python=3.9 -y >/dev/null 2>&1
else
  echo "==> [2/6] Env 'pids' đã có."
fi

# ----------------------------------------------------------------- 3. deps
echo "==> [3/6] Cài dependencies (khớp pip list container — torch CPU thay vì cu117)..."
conda run -n pids pip install --quiet --upgrade pip
# --- torch 1.13.1 CPU (máy thầy không GPU; container dùng 1.13.1+cu117) ---
conda run -n pids pip install --quiet torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1+cpu \
  --extra-index-url https://download.pytorch.org/whl/cpu
# --- các dep khác theo Dockerfile/container ---
conda run -n pids pip install --quiet \
  scikit-learn==1.2.0 networkx==2.8.7 xxhash==3.2.0 graphviz==0.20.1 \
  psutil scipy==1.10.1 matplotlib==3.8.4 wandb==0.24.1 chardet==5.2.0 \
  nltk==3.8.1 igraph==0.11.5 cairocffi==1.7.0 wget==3.2 \
  gensim==4.3.1 pytz==2024.1 pandas==2.2.2 yacs==0.1.8 \
  gdown==5.2.0 umap-learn==0.5.6 flask==3.0.3 \
  psycopg2-binary tqdm setuptools==61.0.0 pytest==8.3.5
# --- torch_geometric + PyG libs (CPU build cho torch 1.13) ---
conda run -n pids pip install --quiet torch_geometric==2.5.3
conda run -n pids pip install --quiet pyg_lib==0.2.0 torch_scatter==2.1.1 \
  torch_sparse==0.6.17 torch_cluster==1.6.1 torch_spline_conv==1.2.2 \
  -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
echo "   OK"

# ----------------------------------------------------------------- 4. repo
mkdir -p "$WORK"
# Nếu PIDSMaker đã clone nhưng chưa hoàn chỉnh hoặc sai state → xoá để clone lại sạch
if [ -d "$PIDS_DIR/.git" ] && ! git -C "$PIDS_DIR" rev-parse --verify "$PIDS_COMMIT^{commit}" >/dev/null 2>&1; then
  echo "==> [4/6] PIDSMaker cũ chưa đủ history (ping $PIDS_COMMIT), xoá để clone lại sạch..."
  rm -rf "$PIDS_DIR"
fi

if [ ! -d "$PIDS_DIR/.git" ]; then
  echo "==> [4/6] Clone PIDSMAKER (pin $PIDS_COMMIT) + áp patches..."
  git clone --quiet https://github.com/ubc-provenance/PIDSMaker.git "$PIDS_DIR"
  git -C "$PIDS_DIR" checkout -q "$PIDS_COMMIT"
  echo "   (HEAD=$(git -C "$PIDS_DIR" rev-parse --short HEAD))"
  # Lưu ý: all_patches.diff ĐÃ BAO GỒM CẢ skip-JSON deviation
  # (file create_database_optc.py đã có 2 chỗ JSONDecodeError guard).
  # KHÔNG áp riêng pidsmaker_patch.diff — bản cũ bị hỏng (corrupt) và thừa.
  if ! git -C "$PIDS_DIR" apply --check "$SCRIPT_DIR/patches/all_patches.diff"; then
    echo "   ⚠️ git apply --check fail, dùng patch -p1 (bỏ qua lỗi whitespace)..."
    (cd "$PIDS_DIR" && patch -p1 -f --ignore-whitespace < "$SCRIPT_DIR/patches/all_patches.diff") || {
      echo "   ❌ Cả hai cách áp patch đều thất bại. Báo output này cho BDTG."
      exit 1
    }
  else
    git -C "$PIDS_DIR" apply "$SCRIPT_DIR/patches/all_patches.diff"
  fi
  cp "$SCRIPT_DIR/config/config.py" "$PIDS_DIR/pidsmaker/config/config.py"
else
  echo "==> [4/6] PIDSMAKER đã có tại $PIDS_DIR"
  git -C "$PIDS_DIR" checkout -q "$PIDS_COMMIT" 2>/dev/null || true
  echo "   (nếu thay đổi patches/config: xoá $PIDS_DIR rồi chạy lại)"
fi

# ------------------------------------------------------------ 5. Postgres DB
echo "==> [5/6] Khởi động Postgres + tạo user 'postgres'..."
sudo service postgresql start >/dev/null 2>&1 || sudo /etc/init.d/postgresql start >/dev/null 2>&1 || true
sleep 2

sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='postgres'" | grep -q 1 || \
  sudo -u postgres psql -tc "CREATE USER postgres SUPERUSER LOGIN PASSWORD 'postgres';"
# Quan trọng: user postgres CÓ THỂ đã tồn tại (tạo bởi apt package) với password khác
# → luôn reset password thành 'postgres' để khớp config/CLI.
sudo -u postgres psql -tc "ALTER USER postgres WITH PASSWORD 'postgres';"
sudo -u postgres psql -tc "ALTER USER postgres SUPERUSER;"
echo "   (postgres user: password khớp 'postgres', superuser OK)"

if [ -n "$DUMP_PATH" ] && [ -f "$DUMP_PATH" ]; then
  echo "==> [5/6] Restore DB từ $DUMP_PATH ..."
  sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='optc_h051_full'" | grep -q 1 || \
    sudo -u postgres createdb -O postgres optc_h051_full
  sudo -u postgres pg_restore --no-owner --no-privileges -d optc_h051_full "$DUMP_PATH" || \
    echo "   ⚠️ pg_restore có warning (thường OK — kiểm tra count bên dưới)"
  CNT=$(sudo -u postgres psql -tA -d optc_h051_full -c "SELECT count(*) FROM event_table;" 2>/dev/null | tr -d ' ')
  echo "   ✅ event_table count: ${CNT:-?}  (kỳ vọng 19,815,600)"
else
  echo "==> [5/6] (không có --dump — bỏ qua restore; DB sẽ tự build từ raw hoặc restore thủ công)"
fi

# --------------------------------------------------------------- 6. hướng dẫn
echo
echo "════════════════════════════════════════════════════════════════"
echo "✅ SETUP XONG. Chạy từng baseline:"
echo "════════════════════════════════════════════════════════════════"
echo
echo "cd $PIDS_DIR && PYTHONPATH=$PIDS_DIR conda run -n pids python -m pidsmaker.main \\"
echo "    flash optc_h051 --cpu --database_host localhost --database_user postgres \\"
echo "    --database_password postgres --artifact_dir $WORK/artifacts \\"
echo "    --evaluation.ground_truth_version orthrus \\"
echo "    --batching.intra_graph_batching.used_methods none \\"
echo "    --training.encoder.used_methods none --training.decoder.use_few_shot False \\"
echo "    --construction.multi_dataset none --featurization.used_method flash \\"
echo "    --featurization.training_split all --featurization.multi_dataset_training False \\"
echo "    --batching.multi_dataset_training False 2>&1 | tee $WORK/logs_flash.log"
echo
echo "Thay 'flash' -> 'magic' (only_type) / 'velox' (word2vec) / 'orthrus'."
echo "Chi tiết: $SCRIPT_DIR/runbook/RUNBOOK_remote_L6.md"
