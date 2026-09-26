@echo off
title RQ4 GGUF Q8_0 - RUNNING
color 0B
cd /d C:\Users\BDTG\optc-bench
echo [1/4] download Qwen3.5-4B Q8_0 (~4.4GB)...
C:\Users\BDTG\venv-cpu\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p=hf_hub_download('unsloth/Qwen3.5-4B-GGUF','Qwen3.5-4B-Q8_0.gguf',local_dir=r'F:\tools\manga-vi\models'); print('DL_OK', p)"
if errorlevel 1 (echo DOWNLOAD FAIL & pause & exit /b 1)
echo [2/4] stop old server...
taskkill /IM llama-server.exe /F >nul 2>&1
timeout /t 3 >nul
echo [3/4] start llama-server Q8_0 :8766...
start "GGUF-Q80-SERVER" /MIN "C:\Users\BDTG\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe" -m "F:\tools\manga-vi\models\Qwen3.5-4B-Q8_0.gguf" --host 127.0.0.1 --port 8766 -ngl 99 -c 4096 -np 4
timeout /t 30 >nul
echo [4/4] quality bench 704...
C:\Users\BDTG\venv-cpu\Scripts\python.exe gguf_quality_gpu3.py
echo.
echo ========== RQ4 DONE ==========
pause
