@echo off
title RQ3 train 43/44 + ensemble - RUNNING
color 0B
cd /d C:\Users\BDTG\optc-bench
echo [1/3] train seed 43...
C:\Users\BDTG\venv-cpu\Scripts\python.exe rq1b_lora4.py 43
echo [2/3] train seed 44...
C:\Users\BDTG\venv-cpu\Scripts\python.exe rq1b_lora4.py 44
echo [3/3] majority-vote ensemble...
C:\Users\BDTG\venv-cpu\Scripts\python.exe rq3_ensemble.py
echo.
echo ========== RQ3 DONE ==========
pause
