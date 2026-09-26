"""RQ4 orchestrator — start llama-server Q8_0 (Popen ke thua cua so), poll port, bench, kill.
Khong con race 10061: chi chay bench khi server thuc su listen."""
import subprocess, time, sys, os, urllib.request, json
sys.stdout.reconfigure(line_buffering=True)
SRV = r"C:\Users\BDTG\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
MOD = r"F:\tools\manga-vi\models\Qwen3.5-4B-Q8_0.gguf"
PORT = 8766

# 1) kill cu
subprocess.run("taskkill /IM llama-server.exe /F", shell=True, capture_output=True)
time.sleep(2)
# 2) start — log file de debug
logf = open(r"C:\Users\BDTG\optc-bench\rq4-srv.log", "w")
p = subprocess.Popen([SRV, "-m", MOD, "--host", "127.0.0.1", "--port", str(PORT),
                      "-ngl", "99", "-c", "4096", "-np", "4"],
                     stdout=logf, stderr=subprocess.STDOUT)
print("server pid", p.pid)
# 3) poll toi da 90s
ok = False
for i in range(45):
    time.sleep(2)
    if p.poll() is not None:
        print("SERVER DIED rc=", p.returncode)
        print(open(r"C:\Users\BDTG\optc-bench\rq4-srv.log", encoding="utf-8", errors="replace").read()[-800:])
        sys.exit(1)
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()
        ok = True
        break
    except Exception:
        pass
if not ok:
    print("server khong listen sau 90s"); sys.exit(1)
print(f"server READY ({(i+1)*2}s) — chay bench...")
# 4) bench trong cung session
rc = subprocess.run([sys.executable, r"C:\Users\BDTG\optc-bench\gguf_quality_gpu3.py"]).returncode
print("bench rc", rc)
# 5) kill server
p.terminate()
print("RQ4_ORCH_DONE")
