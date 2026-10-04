@echo off
chcp 65001 >nul
title AD-GEN explain rerun (5 cau hinh)
cd /d "%~dp0"
if not exist output mkdir output
if not exist logs mkdir logs

echo ============================================================
echo  6 RUN — AD-GEN 704 mau, constrained logprob (khong doi prompt)
echo  Thu tu: 0.5B 0shot - 0.5B 3shot - TinyLlama 1.1B - 1.5B
echo           - 7B teacher (bf16) - 0.5B INT8 (CPU)
echo  Tong thoi gian du kien: 4-7 gio (7B teacher lau nhat)
echo ============================================================
echo.

call :run 05b       explain_slm.py --model Qwen/Qwen2.5-0.5B-Instruct --out output/raw-explain-05b-rerun.json
call :run 3shot_05b explain_slm.py --model Qwen/Qwen2.5-0.5B-Instruct --shots 3 --out output/raw-explain-3shot-05b-rerun.json
call :run 11b       explain_slm.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out output/raw-explain-11b-rerun.json
call :run 15b       explain_slm.py --model Qwen/Qwen2.5-1.5B-Instruct --out output/raw-explain-15b-rerun.json
call :run teacher   explain_slm.py --model Qwen/Qwen2.5-7B-Instruct --dtype bf16 --resume --out output/raw-explain-teacher-rerun.json
call :run int8      explain_05b_int8.py

echo [ALL DONE] %TIME%
echo Gui lai toan bo file trong thu muc output\ de tong hop ket qua.
pause
goto :eof

:run
echo [START] %1  %TIME%
shift
python %1 %2 %3 %4 %5 %6 %7 %8 %9
echo [END]   %TIME%
echo.
goto :eof
