import json, sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
recs = [json.loads(l) for l in open(r"F:\backup\OpTC-thesis\P1\Output\data\rq1a-subset.jsonl", encoding="utf-8")]
# Rule "len >= 2 how many caught"
def rule_all(ch): return len(ch) >= 2
mal_hits = ben_hits = 0
for r in recs:
    ch = r.get("parent_chain", [])
    if isinstance(ch, str):
        try: ch = json.loads(ch.replace("'", '"'))
        except Exception: ch = []
    if rule_all(ch):
        if r["label"] == 1: mal_hits += 1
        else: ben_hits += 1
print(f"RULE 'chain>=2' naively: mal {mal_hits}/12, benign {ben_hits}/192")
# SLM verdict "+ score" ts tach o -2.2746 constant cua 204 recs with T1082 ttp — model gui only one JSON shape?
# bench y system SLM du tra 1 predictions 1 heuristic, dung khong change data. So at all.
# Check if chain>=2 for all 12 mal: 12, so proposed rule-based classifier: 100% recall, FP = ben>=2 (12 ben hits / 192 = 6.25%)
print(f"rule-based result: mal {mal_hits}/12 | benign false alarm {ben_hits}/192")
