
import json, statistics
BASE = r"C:\Users\BDTG\Desktop\Backup\OpTC-thesis\P1\Output\results_phase2"
MODELS = {"05b": "Qwen/Qwen2.5-0.5B-Instruct", "11b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "15b": "Qwen/Qwen2.5-1.5B-Instruct"}
table = []
for tag, model in MODELS.items():
    d = json.load(open(BASE + f"\\raw-explain-bal-{tag}.json", encoding="utf-8"))
    r = d["records"]
    vacc = sum(x["verdict_correct"] for x in r) / len(r)
    tp = sum(1 for x in r if x["is_mal"] and x["verdict"] == 1)
    tn = sum(1 for x in r if not x["is_mal"] and x["verdict"] == 0)
    fp = sum(1 for x in r if not x["is_mal"] and x["verdict"] == 1)
    fn = sum(1 for x in r if x["is_mal"] and x["verdict"] == 0)
    tt = [x["ttp_correct"] for x in r if x.get("ttp_correct") is not None]
    tacc = sum(tt) / len(tt) if tt else 0.0
    hal = sum(x["hallucinated"] for x in r) / len(r)
    sc_ok = [x["verdict_score"] for x in r if x["verdict_correct"] == 1]
    sc_bad = [x["verdict_score"] for x in r if x["verdict_correct"] == 0]
    row = {"model": model, "n": len(r), "verdict_acc": round(vacc, 4),
           "TP": tp, "TN": tn, "FP": fp, "FN": fn,
           "ttp_acc": round(tacc, 4), "ttp_scored": len(tt),
           "hal": round(hal, 4),
           "score_median_correct": round(statistics.median(sc_ok), 4) if sc_ok else None,
           "score_median_wrong": round(statistics.median(sc_bad), 4) if sc_bad else None}
    json.dump(row, open(BASE + f"\\slm-explain-bal-{tag}.json", "w"), indent=1)
    table.append(row)
    print(tag, row)
json.dump(table, open(BASE + "\\slm-explain-bal-all.json", "w"), indent=1)
print("SAVED slm-explain-bal-{05b,11b,15b,all}.json")
