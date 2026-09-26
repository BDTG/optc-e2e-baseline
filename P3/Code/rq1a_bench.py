import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
recs = [json.loads(l) for l in open(r"F:\backup\OpTC-thesis\P1\Output\data\rq1a-subset.jsonl", encoding="utf-8")]
# benign có chain rỗng hTrace — không giúp decision. Ma chain 3/2/3-3 els, benign 0.15 — phân biệt by chain length!
ben_chain_counts = [len(r.get("parent_chain", [])) if isinstance(r.get("parent_chain"), list) else 0 for r in recs if r["label"] == 0]
print("ben chain 0 =", ben_chain_counts.count(0), "/ 192")
print("mal chain >1 =", sum(1 for c in [len(r.get("parent_chain", [])) for r in recs if r["label"]==1] if c > 1), "/ 12")
# rule đơn giản "chain >= 2 → MALICIOUS" cũng đã đủ kiem intrusion theo sample tre — rieng rule-based evidence 152/192 benign has parent_chain -> "parent_chain" RULE applied => ev returns parent_chain?
# Are most benign really chain-less in cascade output?
ben_chain0 = [r for r in recs if r["label"]==0 and len(r.get("parent_chain",[])) == 0]
ben_chain1 = [r for r in recs if r["label"]==0 and len(r.get("parent_chain",[])) > 0]
print(f"ben chain=0: {len(ben_chain0)} | ben chain>=1: {len(ben_chain1)}")
for r in ben_chain1[:3]:
    ch = r.get("parent_chain", [])
    print("  chain>0 benign:", r["nid"], ":", json.dumps([( (c or {}).get("msg")) for c in (ch or [])[:2]])[:300])
