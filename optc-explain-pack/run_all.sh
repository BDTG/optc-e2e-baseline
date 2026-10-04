#!/bin/bash
# 6 run — AD-GEN 704 mau, constrained logprob
set -u
cd "$(dirname "$0")"
mkdir -p output logs
run() { echo "[START] $1 $(date +%H:%M:%S)"; shift; python3 "$@"; echo "[END]   $(date +%H:%M:%S)"; }
run 05b       explain_slm.py --model Qwen/Qwen2.5-0.5B-Instruct --out output/raw-explain-05b-rerun.json
run 3shot_05b explain_slm.py --model Qwen/Qwen2.5-0.5B-Instruct --shots 3 --out output/raw-explain-3shot-05b-rerun.json
run 11b       explain_slm.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out output/raw-explain-11b-rerun.json
run 15b       explain_slm.py --model Qwen/Qwen2.5-1.5B-Instruct --out output/raw-explain-15b-rerun.json
run teacher   explain_slm.py --model Qwen/Qwen2.5-7B-Instruct --dtype bf16 --resume --out output/raw-explain-teacher-rerun.json
run int8      explain_05b_int8.py
echo "[ALL DONE] $(date +%H:%M:%S)"
echo "Gui lai toan bo file trong output/ de tong hop ket qua."
