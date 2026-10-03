"""Chon ~150 mau tu REAL_test_matched de gan gold.
Phan tang theo technique (base), uu tien mau co |C(E)|>1 (KB nhieu ung vien -> can gold phan xu),
kem mot it benign. Xuat gold_pack.jsonl (ban ghi day du) + gold_pack.csv (de gan tay neu can).
KHONG kem output EVE de tranh thien lech."""
import argparse
import collections
import csv
import json
import zlib
from pathlib import Path

from kb import KB


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def okey(seed, sid):
    return (zlib.crc32(f"{seed}:{sid}".encode()), sid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--kb", default=str(Path(__file__).with_name("tech_preconditions.json")))
    ap.add_argument("--out", default="results/gold_pack.jsonl")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--benign_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    kb = KB(a.kb)

    rows = list(read_jsonl(a.data))
    for r in rows:
        C = kb.candidates(r["events"])
        r["_C"] = C
        r["_nC"] = len(C)
        gold = r.get("techniques_base") or []
        r["_stratum"] = gold[0] if gold else (C[0] if C else "none")

    mal = [r for r in rows if r["label"]]
    ben = [r for r in rows if not r["label"]]
    n_ben = round(a.n * a.benign_frac)
    n_mal = a.n - n_ben

    # nhom mal theo stratum; trong moi nhom uu tien |C|>1 truoc, thu tu deterministic
    by_s = collections.defaultdict(list)
    for r in mal:
        by_s[r["_stratum"]].append(r)
    for s in by_s:
        by_s[s].sort(key=lambda r: (0 if r["_nC"] > 1 else 1, okey(a.seed, r["sample_id"])))

    # cap theo ti le nhom, round-robin de phu het technique
    strata = sorted(by_s, key=lambda s: (-len(by_s[s]), s))
    picked, quota = [], {s: 0 for s in strata}
    while len(picked) < n_mal and any(quota[s] < len(by_s[s]) for s in strata):
        for s in strata:
            if len(picked) >= n_mal:
                break
            if quota[s] < len(by_s[s]):
                picked.append(by_s[s][quota[s]])
                quota[s] += 1

    # benign: uu tien co C khong rong (hiem) roi den ngau nhien deterministic
    ben.sort(key=lambda r: (0 if r["_nC"] > 0 else 1, okey(a.seed, r["sample_id"])))
    picked_ben = ben[:n_ben]

    out = picked + picked_ben
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            rec = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # CSV de gan tay (moi dong 1 mau): liet ke event [E#] va field, cot trong technique/evidence
    csv_path = str(Path(a.out).with_suffix(".csv"))
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "label", "events_fields", "technique_fill", "evidence_fields_fill"])
        for r in out:
            ev = " || ".join(f"[E{e['idx']}] eid{e.get('eid')} " +
                             "; ".join(f"{k}={str(v)[:60]}" for k, v in e.get("fields", {}).items())
                             for e in r["events"][:12])
            w.writerow([r["sample_id"], r["label"], ev[:2000], "", ""])

    man = {
        "n_total": len(out), "n_mal": len(picked), "n_ben": len(picked_ben),
        "strata": {s: quota[s] for s in strata if quota[s]},
        "nC_dist": dict(collections.Counter(r["_nC"] for r in picked).most_common()),
        "multi_candidate_frac": round(sum(1 for r in picked if r["_nC"] > 1) / max(len(picked), 1), 4),
    }
    print(json.dumps(man, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
