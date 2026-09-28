#!/bin/bash
# 6 run — AD-GEN 704 mau, constrained logprob
set -u
cd "$(dirname "$0")"
mkdir -p output logs
for s in explain_05b.py explain_3shot_05b.py explain_11b.py explain_15b.py explain_teacher.py explain_05b_int8.py; do
  echo "[START] $s $(date +%H:%M:%S)"
  python3 "$s"
  echo "[END]   $s $(date +%H:%M:%S)"
done
echo "[ALL DONE] $(date +%H:%M:%S)"
echo "Gui lai toan bo file trong output/ de tong hop ket qua."
