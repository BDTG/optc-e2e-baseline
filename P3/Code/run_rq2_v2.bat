@echo off
title RQ2 A2 multi-seed v2 - RUNNING
color 0B
cd /d C:\Users\BDTG\optc-bench
set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
set EPOCHS=1
set BS=16
set MAXLEN=256
for %%s in (42 43 44 45 46) do (
  echo === SEED %%s ===
  set SEED=%%s
  C:\Users\BDTG\venv-ml\Scripts\python.exe a2_multiseed.py
  ren "F:\backup\OpTC-thesis\P1\Output\results_phase2\p3b-domainadapt-result.json" "p3b-da-seed%%s.json" >nul 2>&1
)
echo.
echo ========== RQ2 v2 DONE ==========
pause
