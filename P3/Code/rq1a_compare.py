import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
recs = [json.loads(l) for l in open(r"F:\backup\OpTC-thesis\P1\Output\data\rq1a-subset.jsonl", encoding="utf-8")]
# benign ben — chain led? 192 benign chaining so sanh
mal_chains = [len(r.get("parent_chain", [])) for r in recs if r["label"] == 1]
ben_chains = []; ben_nflow = 0
for r in recs:
    if r["label"] == 0:
        ch = r.get("parent_chain", [])
        if isinstance(ch, list):
            ben_chains.append(len(ch))
        if isinstance(ch, list):
            ben_nflow += sum(1 for c in ch if (c.get("msg") or "").startswith("netflow"))
print("mal chain avg:", sum(mal_chains) / 12, "| benign chain avg:", sum(ben_chains) / max(len(ben_chains),1))
# nhin 3 benign sample text
for r in recs:
    if r["label"] == 0:
        ch = r.get("parent_chain", [])
        el3 = " | ".join([( (c or {}).get("msg") or "") for c in (ch or [])[-3:]])[:400]
        print(f"-- benign nid {r['nid']} chain_text: {el3}")
        if sum(1 for x in recs if x["label"] == 0 and x["nid"] <= r["nid"]) > 4:
            break
