import json, time, os
os.chdir(r"F:/backup/OpTC-thesis")
# 1) Append PLAN-P3 vào repo + update summary
plan_src = r"C:\Users\BDTG\Desktop\PLAN-P3.md"
plan_dst = r"F:\backup\OpTC-thesis\archive\md\PLAN-P3.md"
os.makedirs(os.path.dirname(plan_dst), exist_ok=True)
open(plan_dst, "w", encoding="utf-8").write(open(plan_src, encoding="utf-8").read())
print("plan copied")

# 2) Patch full-benchmark-table.html s16d — thêm RQ1b dòng (tìm </body>)
T = r"F:\backup\OpTC-thesis\P1\Output\results_phase2\full-benchmark-table.html"
t = open(T, encoding="utf-8").read()
BLOCK = """
<details open><summary><b>18b. P3/RQ1b — LoRA fine-tune trên cascade flagged (23/09)</b></summary>
<div style="margin:8px 12px 16px">
<p><b>Setup:</b> rule "chain ≥ 2" pre-filter → 21 dị nghi (12 mal + 9 FP) → LoRA r8 trên Qwen2.5-0.5B,
train balanced 84 mal / 76 benign (160 mẫu, 2 epochs, 15 phút CPU GPD), eval generation đúng prompt train.</p>
<table class="result">
<tr><th>Hệ</th><th>TP</th><th>TN</th><th>FP</th><th>FN</th><th>F1</th></tr>
<tr><td>Rule-only (chain ≥ 2)</td><td>12</td><td>183</td><td>9</td><td>0</td><td>0.727</td></tr>
<tr><td><b>LoRA v4 (balanced 84/76)</b></td><td>11</td><td>3</td><td>6</td><td>1</td><td><b>0.759</b></td></tr>
<tr><td>SLM zero-shot</td><td>0</td><td>—</td><td>—</td><td>12</td><td>0</td></tr>
</table>
<p><i>Phát hiện:</i> benign dữ liệu cạn (19 template + 57 chain trắng đồng nhất) → ceiling FP lọc 3/9.
Cần benign có nhiễu thật để đạt Dương ≥0.60 (khung thầy). SLM zero-shot TP 0/12 xác nhận RQ1a.</p>
</div></details>
"""
if "18b. P3/RQ1b" not in t:
    t = t.replace("</body>", BLOCK + "</body>", 1)
    open(T, "w", encoding="utf-8").write(t)
    print("html patched")
else:
    print("html already patched")

# 3) summary JSON cho rq1b final
final = {
    "rq1b_final": {
        "rule_only": {"TP": 12, "TN": 183, "FP": 9, "FN": 0, "precision": 0.5714, "recall": 1.0, "F1": 0.7273},
        "lora_v4_seed42": {"n_train": 160, "mal": 84, "ben": 76, "eval_flagged": 21,
                           "TP": 11, "TN": 3, "FP": 6, "FN": 1, "acc": 0.6667,
                           "precision": 0.6471, "recall": 0.9167, "F1": 0.759},
        "conclusion": "LoRA lọc 3/9 FP giữ 11/12 TP — F1 0.727→0.759; benign cần nhiễu thật (delimitation)"
    }}
json.dump(final := final, open("P1/Output/results_phase2/rq1b-final.json", "w", encoding="utf-8"), indent=1)
print("final json written")
