import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
recs = [json.loads(l) for l in open(r"F:\backup\OpTC-thesis\P1\Output\data\rq1a-subset.jsonl", encoding="utf-8")]
# xem ALL 12 mal records chain de quan sat pattern
for r in recs:
    if r["label"] != 1:
        continue
    ch = r.get("parent_chain", [])
    n_br = sum(1 for c in (ch or []) if (c.get("msg") or "").startswith("netflow"))
    ev = r.get("event_seq", [])
    es = json.loads(str(es).replace("'", '"')) if isinstance(es, str) else es
    n_events = len(es) if isinstance(es, list) else 0
    sl = r.get("self_label", "")
    print(f"{r['nid']}: chain {len(ch)} els ({n_br} netflow) | es {n_br if False else len(es) if isinstance(es, list) else '?'} | self {sl[:50]}")
