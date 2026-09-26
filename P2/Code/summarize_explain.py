
import json
import glob, os
BASE = r"C:\Users\BDTG\Desktop\Backup\OpTC-thesis\P1\Output\results_phase2"
MODELS = {"05b": "Qwen/Qwen2.5-0.5B-Instruct", "11b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "15b": "Qwen/Qwen2.5-1.5B-Instruct"}
table = []
for tag, model in MODELS.items():
    p = BASE + f"\\raw-explain-{tag}.json"
    d = json.load(open(p, encoding="utf-8"))
    r = d["records"]
    vacc = sum(x["verdict_correct"] for x in r) / len(r)
    tt = [x["ttp_correct"] for x in r if x.get("ttp_correct") is not None]
    tacc = sum(tt) / len(tt) if tt else 0.0
    hal = sum(x["hallucinated"] for x in r) / len(r)
    lat = d.get("summary", {}).get("avg_latency_sec")
    row = {"model": model, "n": len(r), "verdict_acc": round(vacc, 4),
           "ttp_acc": round(tacc, 4), "hal": round(hal, 4), "latency_s": lat,
           "ttp_scored": len(tt)}
    json.dump(row, open(BASE + f"\\slm-explain-{tag}.json", "w"), indent=1)
    table.append(row)
    print(tag, row)
json.dump(table, open(BASE + "\\slm-explain-all.json", "w"), indent=1)
print("SAVED slm-explain-{05b,11b,15b,all}.json — chi acc + hal, khong TF")
