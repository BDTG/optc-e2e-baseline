@echo off
title RQ1b hybrid + LoRA - RUNNING
color 0B
cd /d C:\Users\BDTG\optc-bench
echo [1/2] rule-hybrid...
C:\Users\BDTG\venv-cpu\Scripts\python.exe rq1b_hybrid.py
echo [2/2] LoRA seed 42 (CPU, ~10 phut)...
C:\Users\BDTG\venv-cpu\Scripts\python.exe rq1b_lora.py 42
echo.
echo ========== RQ1B DONE ==========
pause
