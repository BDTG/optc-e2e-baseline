import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
recs = [json.loads(l) for l in open(r"F:\backup\OpTC-thesis\P1\Output\data\rq1a-subset.jsonl", encoding="utf-8")]
for r in recs:
    if r["label"] != 1:
        continue
    ch = r.get("parent_chain", [])
    n_br = sum(1 for c in (ch or []) if (c.get("msg") or "").startswith("netflow"))
    es = r.get("event_seq", [])
    if isinstance(es, str):
        try:
            es = json.loads(es.replace("'", '"'))
        except Exception:
            es = []
    sl = r.get("self_label", "")
    print(f"{r['nid']}: chain {len(ch) if isinstance(ch, list) else '?'} ({n_br} netflow) | es {len(es) if isinstance(es, list) else '?'} | self {sl[:60]}")
