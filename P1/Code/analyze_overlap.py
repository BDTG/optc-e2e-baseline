"""
Dem phan bo overlap 3 truc tinh hieu r:
  T1 = co cmd khong rong (cmdline semantic)
  T2 = parent_chain >= 2 tang (provenance context)
  T3 = event_seq >= 3 events (network/file/registry activity)

Output:
  - % thoa >= 1 truc (co it nhat 1 signal)
  - % thoa >= 2 truc (tap giau, nhieu signal)
  - % thoa ca 3 truc (tap rat giau)
  - Phan bo 7 o (chi co T1, chi co T2, ..., ca 3)
"""
import json
from collections import Counter

PATH = "P1/Output/data/alerts-enriched-v2.jsonl"
recs = [json.loads(l) for l in open(PATH, encoding='utf-8') if l.strip()]
n = len(recs)
print(f"Total alerts: {n}\n")

def has_cmd(r):
    chain = r.get("parent_chain", []) or []
    for c in chain:
        msg = c.get("msg", "") or ""
        if "| cmd:" in msg:
            v = msg.split("| cmd:", 1)[1].strip()
            if v and v.lower() != "none":
                return True
    return False

def has_chain(r):
    return len(r.get("parent_chain", []) or []) >= 2

def has_events(r):
    return len(r.get("event_seq", []) or []) >= 3

# === per-truc count ===
n_T1 = sum(1 for r in recs if has_cmd(r))
n_T2 = sum(1 for r in recs if has_chain(r))
n_T3 = sum(1 for r in recs if has_events(r))

print(f"=== Per-truc ===")
print(f"T1 (cmd non-empty):    {n_T1:4d} ({n_T1/n*100:5.2f}%)")
print(f"T2 (chain depth >=2):  {n_T2:4d} ({n_T2/n*100:5.2f}%)")
print(f"T3 (event_seq >= 3):   {n_T3:4d} ({n_T3/n*100:5.2f}%)")

# === overlap 7-cell (each combination) ===
combo_counter = Counter()
for r in recs:
    flags = (has_cmd(r), has_chain(r), has_events(r))
    combo_counter[flags] += 1

print(f"\n=== Phan bo 7 o (T1, T2, T3) ===")
labels = [
    ("chi T1",       (True, False, False)),
    ("chi T2",       (False, True, False)),
    ("chi T3",       (False, False, True)),
    ("T1+T2",        (True, True,  False)),
    ("T1+T3",        (True, False, True)),
    ("T2+T3",        (False, True, True)),
    ("ca 3 T1+T2+T3",(True, True,  True)),
]
for label, key in labels:
    c = combo_counter.get(key, 0)
    bar = "#" * int(c / n * 50)
    print(f"  {label:18s}: {c:4d} ({c/n*100:5.2f}%)  {bar}")

n_zero = combo_counter.get((False, False, False), 0)
print(f"  {'KHONG TRUC NAO':18s}: {n_zero:4d} ({n_zero/n*100:5.2f}%)")

# === aggregate overlap ===
n_ge1 = sum(c for k, c in combo_counter.items() if any(k))
n_ge2 = sum(c for k, c in combo_counter.items() if sum(k) >= 2)
n_all3 = combo_counter.get((True, True, True), 0)

print(f"\n=== Tong hop ===")
print(f"  >= 1 truc (co signal):   {n_ge1:4d} ({n_ge1/n*100:5.2f}%)")
print(f"  >= 2 truc (tap giau):    {n_ge2:4d} ({n_ge2/n*100:5.2f}%)")
print(f"  ca 3 truc (tap rat giau): {n_all3:4d} ({n_all3/n*100:5.2f}%)")

# === breakdown by gt (malicious) vs benign ===
gt = json.load(open("P1/Output/data/gt_and_scores.json"))["gt_nids"]
gt_set = set(gt)
mal_nids = set()
for r in recs:
    if str(r.get("nid")) in gt_set:
        mal_nids.add(str(r.get("nid")))
print(f"\n=== Phan bo theo GT ===")
print(f"Pos (malicious): {len(mal_nids)}")

for label, key in labels + [("KHONG TRUC NAO", (False, False, False))]:
    c = combo_counter.get(key, 0)
    if c == 0: continue
    # count malicious in this bucket
    mal_in_bucket = sum(1 for r in recs if (has_cmd(r), has_chain(r), has_events(r)) == key
                        and str(r.get("nid")) in gt_set)
    print(f"  {label:18s}: {c:4d} alerts (mal={mal_in_bucket})")

print(f"\n=== % malicious trong moi bucket ===")
for label, key in labels:
    c = combo_counter.get(key, 0)
    if c == 0: continue
    mal = sum(1 for r in recs if (has_cmd(r), has_chain(r), has_events(r)) == key
              and str(r.get("nid")) in gt_set)
    if c > 0:
        print(f"  {label:18s}: {mal}/{c} = {mal/c*100:.2f}% malicious")