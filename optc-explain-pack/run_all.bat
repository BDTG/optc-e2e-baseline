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

for %%s in (explain_05b.py explain_3shot_05b.py explain_11b.py explain_15b.py explain_teacher.py explain_05b_int8.py) do (
  echo [START] %%s  %TIME%
  python %%s
  echo [END]   %%s  %TIME%
  echo.
)

echo [ALL DONE] %TIME%
echo Gui lai toan bo file trong thu muc output\ de tong hop ket qua.
pause
