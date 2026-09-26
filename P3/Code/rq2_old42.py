import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = json.load(open(r"F:\backup\OpTC-thesis\P1\Output\results_phase2\p3b-da-seed42.json", encoding="utf-8"))
print("SLM mean:", d.get("slm_mean"), "| TFIDF mean:", d.get("tfidf_mean"), "| train_s:", d.get("train_s"))
list(d.keys())
