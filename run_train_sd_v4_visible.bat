@echo off
wsl --exec bash -c "cd /home/vung2 && /home/vung2/miniconda3/envs/pids/bin/python -u P1/Code/train_byt5_sd_v4.py 2>&1 | tee /tmp/byt5-v4.log"
pause
