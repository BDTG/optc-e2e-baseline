"""attack_data: so khop nhan silver (technique muc file tu .yml) voi gold (technique cua atomic test
thuc su chay, doc tu chinh cay tien trinh cua alert: atomics\\T####, Invoke-AtomicTest T####).
Bao cao rieng label_source = tree / rare: do phu gold, ty le khop, Cohen kappa.

  python attackdata_silver_gold.py --data attackdata.jsonl --out silver_gold.json
"""
import argparse
import collections
import json
import re
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from eve import read_jsonl

GOLD_RE = re.compile(r"(?:atomics[\\/]+|invoke-atomictest\s+)(T\d{4}(?:\.\d{3})?)", re.I)


def gold_of(rec):
    found = collections.Counter()
    for e in rec["events"]:
        for v in e.get("fields", {}).values():
            for t in GOLD_RE.findall(str(v)):
                found[t.upper()[:5]] += 1
    return found


def summarize(rows):
    with_gold = [r for r in rows if r["gold"]]
    agree = [int(bool(set(r["silver"]) & set(r["gold"]))) for r in with_gold]
    exact = [int(set(r["silver"]) == set(r["gold"])) for r in with_gold]
    # nhan don: gold = technique xuat hien nhieu nhat; silver = technique trung gold neu co, khong thi technique dau
    ys = [r["silver"][0] if not set(r["silver"]) & set(r["gold"]) else sorted(set(r["silver"]) & set(r["gold"]))[0]
          for r in with_gold]
    yg = [max(r["gold"], key=lambda t: (r["gold_count"][t], t)) for r in with_gold]
    kappa = cohen_kappa_score(ys, yg) if len(set(ys) | set(yg)) > 1 and with_gold else None
    n = len(rows)
    return {"n": n, "n_with_gold": len(with_gold), "gold_coverage": round(len(with_gold) / n, 4) if n else None,
            "agree_any": round(sum(agree) / len(agree), 4) if agree else None,
            "agree_exact_set": round(sum(exact) / len(exact), 4) if exact else None,
            "cohen_kappa": round(float(kappa), 4) if kappa is not None else None,
            "n_classes": len(set(ys) | set(yg)),
            "disagreements": [{"sample_id": r["id"], "silver": r["silver"], "gold": sorted(r["gold"])}
                              for r, g in zip(with_gold, agree) if not g][:25]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = []
    for r in read_jsonl(a.data):
        g = gold_of(r)
        rows.append({"id": r["sample_id"], "source": r.get("label_source"),
                     "silver": sorted(set(r.get("techniques_base") or [])), "gold": sorted(g), "gold_count": g})
    out = {"data": a.data, "silver": "techniques_base tu .yml (muc file)",
           "gold": "technique atomic test trong cay tien trinh cua alert (regex atomics\\T####, Invoke-AtomicTest T####), muc base",
           "all": summarize(rows)}
    for s in sorted({r["source"] for r in rows}):
        out[s] = summarize([r for r in rows if r["source"] == s])
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k in ["all"] + sorted({r["source"] for r in rows}):
        print(k, {x: y for x, y in out[k].items() if x != "disagreements"})


if __name__ == "__main__":
    main()
