"""Them 18f: RQ1b v5 benign-noise (fail co gia tri method) vao HTML."""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(r"F:/backup/OpTC-thesis")
T = "P1/Output/results_phase2/full-benchmark-table.html"
t = open(T, encoding="utf-8").read()
BLOCK = """
<details open><summary><b>18f. P3/RQ1b v5 — Retrain với benign nhiễu THẬT (24/09, negative result)</b></summary>
<div style="margin:8px 12px 16px">
<p><b>Setup:</b> benign mới = 869 chain≥2 thật (svchost/wmiprvse/services) từ 2250 cascade → train 84 mal / 103 ben, seed 43, 2 ep.</p>
<table class="result">
<tr><th>Config</th><th>TP</th><th>FP</th><th>F1</th></tr>
<tr><td>v4 (benign trắng 19+57)</td><td>11</td><td>6</td><td><b>0.759</b></td></tr>
<tr><td>v5 (benign nhiễu thật)</td><td>3</td><td>3</td><td>0.333</td></tr>
</table>
<p><i>Đọc:</i> feature "chain≥2" mất ý nghĩa phân biệt khi benign cũng chain≥2 → model lệch BENIGN (recall 1.0→0.25).
<b>Empirical proof của delimitation RQ1b:</b> SLM 0.5B không đủ capacity tách benign noise thật khỏi mal chain thật —
không phải thiếu data mà là khoảng cách tín hiệu quá mỏng cho tier-2. v4 giữ làm config tốt nhất (headline).</p>
</div></details>
"""
if "18f. P3/RQ1b v5" not in t:
    t = t.replace("</body>", BLOCK + "</body>", 1)
    open(T, "w", encoding="utf-8").write(t)
    print("html 18f patched")
else:
    print("already")
