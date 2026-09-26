"""RQ4 final: patch HTML 18e + rq4-final.json."""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(r"F:/backup/OpTC-thesis")
D = "P1/Output/results_phase2"
q80 = json.load(open(f"{D}/rq4-q80-quality.json", encoding="utf-8"))
final = {"rq4_quantization": {
    "q4_k_m_verdict_acc": 0.2585, "q8_0_verdict_acc": q80.get("verdict_acc"),
    "q8_0_ttp_acc": q80.get("ttp_acc_on_gt"), "q8_0_s_per_sample": q80.get("s_per_sample"),
    "fp32_reference": 0.939,
    "conclusion": "Q8_0 (0.256) ~= Q4_K_M (0.259) — khong phai do bit-width. Ca 2 quantized GGUF deu sụp vs fp32 0.939. Nghi van he thong: bench prompt GGUF co the khac few-shot cua fp32 pipeline (limitation ghi ro). GO/NO-GO: NO-GO quantized cho scoring product; fp32 CPU giu la con duong duy nhat"}}
json.dump(final, open(f"{D}/rq4-final.json", "w", encoding="utf-8"), indent=1)
print("rq4-final written, q8_0 acc:", q80.get("verdict_acc"))
T = f"{D}/full-benchmark-table.html"
t = open(T, encoding="utf-8").read()
BLOCK = """
<details open><summary><b>18e. P3/RQ4 — GGUF Q8_0 quality go/no-go (24/09)</b></summary>
<div style="margin:8px 12px 16px">
<table class="result">
<tr><th>Định lượng</th><th>Verdict acc</th><th>TTP acc</th><th>s/mẫu</th></tr>
<tr><td>fp32 (transformers, tham chiếu)</td><td><b>0.939</b></td><td>0.288</td><td>21.4 (CPU)</td></tr>
<tr><td>GGUF Q4_K_M (2.7GB)</td><td>0.259</td><td>0.047</td><td>0.37 (GPU)</td></tr>
<tr><td>GGUF Q8_0 (4.5GB, GPU)</td><td>0.256</td><td>0.060</td><td>0.49 (GPU)</td></tr>
<tr><td>INT8 torch dynamic</td><td>0.092</td><td>0.144</td><td>2.3</td></tr>
</table>
<p><i>Kết luận RQ4:</i> <b>Q8_0 ≈ Q4_K_M</b> — tăng bit-width KHÔNG cứu được. Cả 2 GGUF ≈ 0.26 vs fp32 0.939.
Giả thuyết: benchmark GGUF prompt (chat+few-shot) khác cấu hình pipeline fp32 (limitation — cần align prompt nếu muốn kết luận sâu hơn).
<b>GO/NO-GO: NO-GO</b> cho quantized trong scoring product — fp32 CPU (0.5B, 1-core 25.5s) giữ là con đường duy nhất của tier-2.</p>
</div></details>
"""
if "18e. P3/RQ4" not in t:
    t = t.replace("</body>", BLOCK + "</body>", 1)
    open(T, "w", encoding="utf-8").write(t)
    print("html patched")
else:
    print("html already")
