import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = json.load(open(r"F:\backup\OpTC-thesis\P1\Output\data\rq1a-subset.jsonl", encoding="utf-8"))
# xem parent_chain cua 12 mal - bien thanh empty?
for r in d:
    if r["label"] == 1:
        ch = r.get("parent_chain", [])
        print(r["nid"], "| type", type(ch).__name__, "| len", len(ch) if hasattr(ch, "__len__") else "?")
        s = json.dumps(ch)[:300] if ch else "EMPTY"
        print("  chain:", s)
        es = r.get("event_seq", [])
        print("  event_seq len", len(str(es)) if es else 0)
        self_label = r.get("self_label", "")
        print("  self_label:", self_label[:100])
        break
