"""Patch gguf_quality_gpu3.py: output -> rq4-q80-quality.json (khong ghi de Q4)."""
import ast, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = r"C:\Users\BDTG\optc-bench\gguf_quality_gpu3.py"
t = open(p, encoding="utf-8").read()
if "gguf-quality.json" in t:
    t = t.replace("gguf-quality.json", "rq4-q80-quality.json")
    open(p, "w", encoding="utf-8").write(t)
    ast.parse(t)
    print("patched output name")
else:
    print("pattern not found; keys:", [l.strip()[:80] for l in t.splitlines() if "json" in l.lower()][:5])
