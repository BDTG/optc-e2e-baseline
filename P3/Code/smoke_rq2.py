"""Smoke test: chay a2_multiseed SEED=44 va CHI DUNG lai sau TFIDF block (chi cho 2 TTP).
Bat nhanh trong 60s — khong giai thich qua nhieu."""
import sys, subprocess, os, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
env = dict(os.environ); env["SEED"] = "44"
p = r"C:\Users\BDTG\optc-bench\a2_multiseed.py"
# chay voi timeout ngan 90s — chi coi TFIDF in
try:
    r = subprocess.run([r"C:\Users\BDTG\venv-cpu\Scripts\python.exe", p],
                       capture_output=True, text=True, timeout=90, env=env)
    print(r.stdout[-1200:]); print("ERR:", r.stderr[-400:])
except subprocess.TimeoutExpired as e:
    print("TIMEOUT OK — process log so far:")
    print((e.stdout or b"").decode("utf-8", "replace")[-800:] if isinstance(e.stdout, bytes) else (e.stdout or "")[-800:])
