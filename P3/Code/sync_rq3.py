"""RQ3 sync: html section 18c + final json + git bundle push."""
import json, os
os.chdir(r"F:/backup/OpTC-thesis")
T = r"F:\backup\OpTC-thesis\P1\Output\results_phase2\full-benchmark-table.html"
t = open(T, encoding="utf-8").read()
BLOCK = """
<details open><summary><b>18c. P3/RQ3 — Ensemble 3×0.5B majority-vote (23/09)</b></summary>
<div style="margin:8px 12px 16px">
<p><b>Setup:</b> 3 adapters LoRA (seeds 42/43/44, cùng data v4 balanced 84/76) — mỗi model tự chấm 21 dị nghi, vote ≥2/3 → MALICIOUS.</p>
<table class="result">
<tr><th>Hệ (trên 21 flagged)</th><th>TP</th><th>TN</th><th>FP</th><th>acc</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
<tr><td>Seed 42 đơn lẻ</td><td>11</td><td>3</td><td>6</td><td>0.667</td><td>0.647</td><td>0.917</td><td>0.759</td></tr>
<tr><td>Seed 43 đơn lẻ</td><td><b>12</b></td><td><b>4</b></td><td><b>5</b></td><td><b>0.762</b></td><td>0.706</td><td>1.0</td><td><b>0.828</b></td></tr>
<tr><td>Seed 44 đơn lẻ</td><td>12</td><td>0</td><td>9</td><td>0.571</td><td>0.571</td><td>1.0</td><td>0.727</td></tr>
<tr><td><b>Ensemble majority-vote</b></td><td>12</td><td>1</td><td>8</td><td>0.619</td><td>0.600</td><td>1.0</td><td>0.750</td></tr>
</table>
<p><i>Kết luận RQ3:</i> seed 43 đơn lẻ (F1 0.828) &gt; ensemble (0.750) — voter yếu (seed 44 constant-MALICIOUS)
kéo ensemble về phía MALICIOUS. <b>Bài học:</b> ensemble không cứu được khi benign thiếu nhiễu — đường trên vẫn là
bổ sung benign chất lượng (delimitation RQ1b), không phải cộng thêm mô hình.</p>
</div></details>
"""
if "18c. P3/RQ3" not in t:
    t = t.replace("</body>", BLOCK + "</body>", 1)
    open(T, "w", encoding="utf-8").write(t)
    print("html patched")
else:
    print("already")
final = {"rq3_ensemble": {
    "single": {"s42": {"TP": 11, "TN": 3, "FP": 6, "F1": 0.759},
               "s43": {"TP": 12, "TN": 4, "FP": 5, "F1": 0.828},
               "s44": {"TP": 12, "TN": 0, "FP": 9, "F1": 0.727}},
    "majority_vote_3x0.5B": {"TP": 12, "TN": 1, "FP": 8, "acc": 0.619, "precision": 0.6,
                             "recall": 1.0, "F1": 0.75},
    "conclusion": "Seed 43 đơn (F1 0.828) > ensemble (0.750) — voter yếu kéo ensemble; đa dạng model không cứu virus benign cạn"
}}
json.dump(final, open("P1/Output/results_phase2/rq3-final.json", "w", encoding="utf-8"), indent=1)
print("rq3-final written")
