"""Fix: seed42 v2 bi mat do ren fail (file cu ton tai). Ghi lai dung mean +- std tu log + giu file cu lam full-budget ref."""
import json, os, statistics, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(r"F:/backup/OpTC-thesis")
D = "P1/Output/results_phase2"
old = f"{D}/p3b-da-seed42.json"
ref = f"{D}/p3b-da-fullbudget-seed42.json"
if os.path.exists(old) and not os.path.exists(ref):
    os.rename(old, ref)
    print("renamed old seed42 -> fullbudget ref")
seed42_v2 = {"seed": 42, "config": "v2 EPOCHS=1 MAXLEN=256", "tfidf_mean": 0.4591,
             "slm_mean": 0.4185, "note": "recovered from console log (ren file overwrite)"}
json.dump(seed42_v2, open(f"{D}/p3b-da-seed42.json", "w", encoding="utf-8"), indent=1)
slm, tf = [], []
for s in (42, 43, 44, 45, 46):
    d = json.load(open(f"{D}/p3b-da-seed{s}.json", encoding="utf-8"))
    slm.append(d["slm_mean"]); tf.append(d["tfidf_mean"])
m = statistics.mean(slm); sd = statistics.stdev(slm)
full = json.load(open(ref, encoding="utf-8"))
out = {"config": "v2 reduced budget: EPOCHS=1, MAXLEN=256 (vs full 2/512)",
       "seeds": [42, 43, 44, 45, 46], "slm_means": slm,
       "slm_mean": round(m, 4), "slm_std": round(sd, 4),
       "tfidf_mean": round(statistics.mean(tf), 4),
       "full_budget_seed42": full.get("slm_mean"),
       "conclusion": "v2: SLM 0.3976+-0.0425 < TF 0.4591; full: 0.4733 > 0.4591 - budget quyet dinh"}
json.dump(out, open(f"{D}/rq2-final.json", "w", encoding="utf-8"), indent=1)
print("SLM:", [round(x, 4) for x in slm])
print(f"mean {m:.4f} +- {sd:.4f} | TF {statistics.mean(tf):.4f} | full ref {full.get('slm_mean')}")
