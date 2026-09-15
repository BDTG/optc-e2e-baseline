"""Viec 3 — bench CPU on-device v2 (nhieu cau hinh: threads / int8 / burn + spec capture).
Usage:
  python bench_cpu.py --model Qwen/Qwen2.5-0.5B-Instruct --n 40 --out cpu-05b.json
  python bench_cpu.py --model ... --threads 4 --n 20 --out cpu-05b-t4.json
  python bench_cpu.py --model ... --int8 --n 20 --out cpu-05b-int8.json
  python bench_cpu.py --model ... --burn 8 --n 15 --out cpu-05b-burn8.json
"""
import argparse, json, os, platform, statistics, subprocess, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--bench", default=os.path.join(HERE, "adgen-ttp-bench.jsonl"))
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--out", default="cpu-bench.json")
ap.add_argument("--burn", type=int, default=0, help="so tien trinh dot CPU (mo phong may ban)")
ap.add_argument("--threads", type=int, default=0, help="0 = mac dinh torch")
ap.add_argument("--int8", action="store_true", help="dynamic quantization (do tran tren)")
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)

sys.path.insert(0, HERE)
from product_explain import explain  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

try:
    import psutil
    PROC = psutil.Process()
    def rss(): return PROC.memory_info().rss / 1e9
    PHYS = psutil.cpu_count(logical=False)
    LOGI = psutil.cpu_count(logical=True)
    RAM_GB = round(psutil.virtual_memory().total / 1e9, 1)
except ImportError:
    PROC = None
    PHYS = LOGI = 0
    RAM_GB = -1.0
    def rss(): return -1.0

burners = []
if a.burn:
    for _ in range(a.burn):
        burners.append(subprocess.Popen(
            [sys.executable, "-c", "import numpy\nwhile True:\n a=numpy.random.rand(512,512);a@a"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

rows = []
with open(a.bench, encoding="utf-8") as f:
    for l in f:
        try:
            rows.append(json.loads(l))
        except Exception:
            pass
rows = rows[:a.n]


def text_of(o):
    ch = o.get("parent_chain", []) or []
    return " | ".join([(c.get("msg") or "") for c in ch[-5:]])


rss0 = rss()
tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    a.model, torch_dtype=torch.float32, device_map=None).to("cpu")
model.eval()
cfg = model.config
if a.int8:
    import torch.quantization as tq
    model = tq.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.eval()
rss1 = rss()
kv_heads = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 1))
hd = getattr(cfg, "head_dim", cfg.hidden_size // getattr(cfg, "num_attention_heads", 1))
kv_mb_512 = 2 * cfg.num_hidden_layers * kv_heads * hd * 512 * (1 if a.int8 else 4) / 1e6
try:
    explain(text_of(rows[0]), model, tok)
except Exception as e:
    print("warmup fail:", e)
rss_warm = rss()

lats, outs = [], []
import threading
STOP = False
peak_rss = [rss()]


def _sampler():
    while not STOP:
        v = rss()
        if v > peak_rss[0]:
            peak_rss[0] = v
        time.sleep(0.1)


th = threading.Thread(target=_sampler, daemon=True)
th.start()
for i, o in enumerate(rows):
    t0 = time.time()
    try:
        r = explain(text_of(o), model, tok)
    except Exception as e:
        print(f"sample {i} fail:", str(e)[:100])
        continue
    lats.append(time.time() - t0)
    outs.append({"nid": o.get("nid"), "latency_s": round(lats[-1], 3), **{k: r[k] for k in
                ("verdict", "mitre_technique", "evidence", "recommended_action")}})
    if (i + 1) % 5 == 0:
        print(f"{i+1}/{len(rows)} p50={statistics.median(lats):.2f}s", flush=True)
rss2 = rss()
STOP = True
time.sleep(0.2)
for p in burners:
    p.kill()
lats.sort()
q = lambda p: lats[min(int(p * len(lats)), len(lats) - 1)]
res = {"model": a.model, "device": "cpu", "threads": torch.get_num_threads(),
       "cpu": platform.processor()[:80] or platform.uname().machine,
       "cpu_physical": PHYS, "cpu_logical": LOGI, "ram_total_gb": RAM_GB,
       "os": platform.platform()[:70], "torch": torch.__version__,
       "int8": bool(a.int8), "n": len(lats), "burn": a.burn,
       "p50_s": round(q(0.50), 3), "p95_s": round(q(0.95), 3), "p99_s": round(q(0.99), 3),
       "ram_rss_gb": round(rss2, 2), "ram_model_only_gb": round(rss1 - rss0, 2) if rss0 > 0 else -1,
       "ram_steady_gb": round(rss_warm, 2) if rss_warm > 0 else -1,
       "ram_peak_gb": round(peak_rss[0], 2),
       "kv_mb_ctx512": round(kv_mb_512, 1),
       "throughput_per_day": int(86400 / q(0.50)) if q(0.50) > 0 else -1,
       "samples": outs}
json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
print("SAVED", a.out, "| p50", res["p50_s"], "p95", res["p95_s"], "RAM", res["ram_rss_gb"],
      "| senti/ngay", res["throughput_per_day"], flush=True)
