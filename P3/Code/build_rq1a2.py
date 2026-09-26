import json, sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Not labeled file — kiem tra score_map + gt_nids de danh label. 12 mal v'gt va 2250 benign
# Trong alerts-enriched-v2: 2250 = 12 mal + 2238 benign (theo previous phan tich)
# file nay khong co label field: label lay tu gt_nids
d = json.load(open(r"F:\backup\OpTC-thesis\P1\Output\data\gt_and_scores.json", encoding="utf-8"))
gt_set = set(d["gt_nids"])
mal = []; ben = []
for l in open(r"F:\backup\OpTC-thesis\P1\Output\data\alerts-enriched-v2.jsonl", encoding="utf-8"):
    try: o = json.loads(l)
    except Exception: continue
    r = {"nid": o["nid"], "label": 1 if o["nid"] in gt_set else 0,
         "parent_chain": o.get("parent_chain", []), "event_seq": o.get("event_seq", []),
         "self_label": o.get("self_label", ""), "score": o.get("score")}
    (mal if r["label"] == 1 else ben).append(r)
print(f"mal {len(mal)}, benign {len(ben)}")
print("mal nids sample:", [m["nid"] for m in mal[:14]])
# save full subset 12 mal + 200 benign
sub = mal[:12] + ben[:192]
with open(r"F:\backup\OpTC-thesis\P1\Output\data\rq1a-subset.jsonl", "w", encoding="utf-8") as f:
    for r in sub:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"subset: {len(sub)} = {12} mal + {len(ben[:192])} benign")
