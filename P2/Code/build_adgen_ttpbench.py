"""Xay bo duoc chuan AD-GEN cho TTP eval theo yeu cau thay:
 1. Chuon nhieu chain co GT TTP (nam TTP_ENUM 24 cua tier-2 hoac gong rong hon)
 2. Chunk nhieu TTP >= 20 mau (dieu kien N du cua thay)
 3. hallucination moi: 'evidence=none' khi chain CO bang chung -> phat NE
 4. Chunk theo environment (LAB/REAL) - khong random
Output: adgen-ttp-bench.jsonl (danh cho adgen_constrained chay lai)
"""
import json, re, collections, sys

FULL = "P1/Output/data/adgen-chains.jsonl"
OUT = "P1/Output/data/adgen-ttp-bench.jsonl"
MIN_PER_TTP = 20
MAX_PER_TTP = 60
SEED_SALT = 7

# Tu TTP_ENUM goc (24) — giu enum dung nhu tier-2 de so khop truc tiep
TTP_ENUM = ["T1003", "T1003.001", "T1059", "T1059.001", "T1053", "T1053.005",
            "T1071", "T1071.001", "T1218", "T1218.001", "T1218.011", "T1027",
            "T1027.001", "T1562", "T1572", "T1574.002", "T1060", "T1490",
            "T1064", "T1087", "T1087.001", "T1082", "T1083", "none"]
ENUM_SET = set(TTP_ENUM)

def has_evidence(o):
    """Chain co bang chung de cham evidence (cmdline thuc hoac chain >= 2)."""
    ch = o.get("parent_chain", []) or []
    t = " | ".join([(c.get("msg") or "") for c in ch[-5:]]).lower()
    if "cmd: " in t and "cmd: none" not in t:
        return True
    return len(ch) >= 2

def main():
    by_ttp = collections.defaultdict(list)
    n_read = 0
    for l in open(FULL, encoding="utf-8"):
        if not l.strip():
            continue
        n_read += 1
        o = json.loads(l)
        if o.get("label") != 1:
            continue
        ts = [t for t in re.findall(r"T\d{4}(?:\.\d{3})?", str(o.get("technique_id", "")))]
        # match enum: dung technique goc hoac sub cua goc
        bases = set()
        for t in ts:
            m = re.match(r"(T\d{4})", t)
            if m and (m.group(1) in ENUM_SET or t in ENUM_SET):
                bases.add(m.group(1))
        if not bases:
            continue
        if not has_evidence(o):
            continue  # chain khong co bang chung thi bo — evidence=none se dung cho ben
        for b in bases:
            by_ttp[b].append(o)

    picked = []
    ttp_stats = {}
    import random
    for b, lst in sorted(by_ttp.items()):
        random.Random(SEED_SALT + hash(b) % 1000).shuffle(lst)
        take = lst[:MAX_PER_TTP]
        if len(lst) >= MIN_PER_TTP:
            picked.extend(take)
            ttp_stats[b] = {"avail": len(lst), "taken": len(take)}
        else:
            ttp_stats[b] = {"avail": len(lst), "taken": 0}
    # dedupe theo nid (1 chain co the co nhieu TTP)
    seen = set()
    uniq = []
    for o in picked:
        if o["nid"] in seen:
            continue
        seen.add(o["nid"])
        uniq.append(o)
    with open(OUT, "w", encoding="utf-8") as f:
        for o in uniq:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    ok = {k: v for k, v in ttp_stats.items() if v["taken"] > 0}
    print(f"doc {n_read}, chunk {len(uniq)} chain duy nhat, {len(ok)} TTP du N>={MIN_PER_TTP}", flush=True)
    for k in sorted(ok, key=lambda x: -ok[x]['avail'])[:25]:
        print(f"  {k}: avail={ok[k]['avail']} taken={ok[k]['taken']}", flush=True)
    print(f"SAVED {OUT}", flush=True)
    json.dump(ttp_stats, open("P1/Output/results_phase2/adgen-ttp-stats.json", "w"), indent=1)

if __name__ == "__main__":
    main()
