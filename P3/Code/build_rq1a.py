import json, sys, re, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# P3/RQ1a: SLM chạy trên các chain đã được cascade lọc (alerts-enriched-v2 = 2250 benign / 12 mal)
# Đây là tập doc đã có — s/it ~21s → chỉ hỗ 12 mal + 200 benign subset (đạt "many FP" question)
p = r"F:\backup\OpTC-thesis\P1\Output\data\alerts-enriched-v2.jsonl"
sel = []
for i, l in enumerate(open(p, encoding="utf-8")):
    try:
        o = json.loads(l)
    except Exception:
        continue
    sel.append({"nid": o["nid"], "label": int(o.get("label", 0)), "parent_chain": o.get("parent_chain", []),
                "event_seq": o.get("event_seq"), "self_label": o.get("self_label", "")})
# 12 mal toan bộ + 200 benign đầu
mal = [r for r in sel if r["label"] == 1][:12]
ben = [r for r in sel if r["label"] == 0][:200]
sub = mal + ben
with open(r"F:\backup\OpTC-thesis\P1\Output\data\rq1a-subset.jsonl", "w", encoding="utf-8") as f:
    for r in sub:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"subset saved: {len(sub)} = {len(mal)} mal + {len(ben)} benign (cascade output)")
print("tabel verdict 0/1 in subset:", collections.Counter(r["label"] for r in sub))
