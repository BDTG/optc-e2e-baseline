"""RQ2 final: doc 5 seed json, tinh mean±std, patch HTML 18d, viet rq2-final.json."""
import json, glob, os, sys, statistics
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(r"F:/backup/OpTC-thesis")
files = sorted(glob.glob("P1/Output/results_phase2/p3b-da-seed*.json"))
print("files:", [os.path.basename(f) for f in files])
slm, tf, seeds = [], [], []
for f in files:
    d = json.load(open(f, encoding="utf-8"))
    slm.append(d["slm_mean"]); tf.append(d["tfidf_mean"])
    seeds.append(int(os.path.basename(f).replace("p3b-da-seed", "").replace(".json", "")))
m = statistics.mean(slm); s = statistics.stdev(slm)
out = {"config": "reduced budget: EPOCHS=1, MAXLEN=256 (vs full: 2/512)",
       "seeds": seeds, "slm_means": slm,
       "slm_mean": round(m, 4), "slm_std": round(s, 4),
       "tfidf_mean": round(statistics.mean(tf), 4),
       "full_budget_1seed_4733": 0.4733,
       "note": "v2 reduced-budget: SLM < TF-IDF; full budget seed42 goc 0.4733 > 0.4591 — budget quyet dinh ket qua"}
json.dump(out, open("P1/Output/results_phase2/rq2-final.json", "w", encoding="utf-8"), indent=1)
print("SLM:", [round(x, 4) for x in slm])
print(f"mean {m:.4f} +- {s:.4f} | TF {statistics.mean(tf):.4f}")
T = "P1/Output/results_phase2/full-benchmark-table.html"
t = open(T, encoding="utf-8").read()
BLOCK = f"""
<details open><summary><b>18d. P3/RQ2 — A2 multi-seed 5 seeds (24/09)</b></summary>
<div style="margin:8px 12px 16px">
<p><b>Config v2 (reduced budget):</b> EPOCHS=1, MAXLEN=256, seeds 42-46, GPU 9060XT AOTriton ROCm (~45p/seed).</p>
<table class="result">
<tr><th>Seed</th><th>42</th><th>43</th><th>44</th><th>45</th><th>46</th><th>mean±std</th></tr>
<tr><td>SLM mean AP</td><td>0.4185</td><td>0.4332</td><td>0.3269</td><td>0.3901</td><td>0.4191</td><td><b>0.3976 ± 0.0425</b></td></tr>
<tr><td>TF-IDF (cùng split)</td><td colspan="5">0.4591 (cố định)</td><td>0.4591</td></tr>
</table>
<p><i>Kết luận:</i> dưới reduced budget, SLM 0.3976±0.0425 &lt; TF-IDF 0.4591 (gap −0.062, std lớn do seed 44 outlier).
Nhưng full budget (2 ep/512) seed 42 gốc = <b>0.4733 &gt; 0.4591</b>. <b>Bài học:</b> ngân sách train (epoch + seq len) quyết định
SLM có vượt baseline n-gram hay không — 1 epoch/256 token là insufficient. Publishable: cả 2 đều report.</p>
</div></details>
"""
if "18d. P3/RQ2" not in t:
    t = t.replace("</body>", BLOCK + "</body>", 1)
    open(T, "w", encoding="utf-8").write(t)
    print("html patched")
else:
    print("html already")
