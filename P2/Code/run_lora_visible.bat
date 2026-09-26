@echo off
cd /d D:\OpTC-thesis
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
title LoRA 0.5B Training (GPU)
echo === LoRA 0.5B training - GPU visible window ===
echo Start: %date% %time%
python P1\Code\tier2_lora_train.py
echo.
echo === DONE - Enter to close ===
pause
