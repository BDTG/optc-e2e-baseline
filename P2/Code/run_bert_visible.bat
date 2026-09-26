@echo off
cd /d D:\OpTC-thesis
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
title BERT 150M Training (GPU)
echo === BERT 150M training - GPU visible window ===
echo Start: %date% %time%
python P1\Code\tier2_bert_train.py
echo.
echo === DONE - Enter to close ===
pause