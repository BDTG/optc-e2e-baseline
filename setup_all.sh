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

if ! conda env list | grep -q '^pids\s'; then
  echo "==> [2/6] Tạo env python=3.9 'pids'..."
  conda create -n pids python=3.9 -y >/dev/null 2>&1
else
  echo "==> [2/6] Env 'pids' đã có."
fi

# ----------------------------------------------------------------- 3. deps
echo "==> [3/6] Cài dependencies (torch CPU + sklearn/networkx/igraph...)"
conda run -n pids pip install --quiet --upgrade pip
conda run -n pids pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
conda run -n pids pip install --quiet \
  scikit-learn==1.2.0 networkx==2.8.7 xxhash==3.2.0 graphviz==0.20.1 \
  psutil scipy==1.10.1 matplotlib==3.8.4 wandb==0.24.1 chardet==5.2.0 \
  nltk==3.8.1 igraph==0.11.5 cairocffi==1.7.0 wget==3.2 \
  psycopg2-binary tqdm
echo "   OK"

# ----------------------------------------------------------------- 4. repo
mkdir -p "$WORK"
if [ ! -d "$PIDS_DIR/.git" ]; then
  echo "==> [4/6] Clone PIDSMAKER + áp patches..."
  git clone --depth 1 https://github.com/ubc-provenance/PIDSMaker.git "$PIDS_DIR"
  git -C "$PIDS_DIR" apply "$SCRIPT_DIR/patches/all_patches.diff"
  git -C "$PIDS_DIR" apply "$SCRIPT_DIR/patches/pidsmaker_patch.diff"
  cp "$SCRIPT_DIR/config/config.py" "$PIDS_DIR/pidsmaker/config/config.py"
else
  echo "==> [4/6] PIDSMAKER đã có tại $PIDS_DIR"
  echo "   (nếu thay đổi patches/config: xoá $PIDS_DIR rồi chạy lại)"
fi

# ------------------------------------------------------------ 5. Postgres DB
echo "==> [5/6] Khởi động Postgres + tạo user 'postgres'..."
sudo service postgresql start >/dev/null 2>&1 || sudo /etc/init.d/postgresql start >/dev/null 2>&1 || true
sleep 2

sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='postgres'" | grep -q 1 || \
  sudo -u postgres psql -tc "CREATE USER postgres SUPERUSER LOGIN PASSWORD 'postgres';"

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
