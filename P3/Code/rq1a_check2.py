import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = None
for l in open(r"F:\backup\OpTC-thesis\P1\Output\data\rq1a-subset.jsonl", encoding="utf-8"):
    try:
        d = json.loads(l)
        if d.get("label") == 1:
            break
    except Exception:
        continue
ch = d.get("parent_chain", [])
print("mal nid", d["nid"], "| type", type(ch).__name__, "| len", len(ch) if hasattr(ch, "_len__") else "?")
s = json.dumps(ch)[:400] if ch else "EMPTY"
print("chain:", s)
es = d.get("event_seq", [])
print("event_seq len", len(str(es)))
sl = d.get("self_label", "")
print("self_label:", sl[:120])
