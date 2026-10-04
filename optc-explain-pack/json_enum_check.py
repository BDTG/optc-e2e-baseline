"""Kiem tra file ket qua json_enum: du n (khong trung/thieu sample) va phan phoi technique du doan so voi gold.

  python json_enum_check.py --results q05_REAL_json_enum.jsonl [...] --data REAL_test_matched.jsonl --out check.json
"""
import argparse
import collections
import json
import math
from pathlib import Path

from eve import read_jsonl


def entropy(c):
    n = sum(c.values())
    return round(-sum(v / n * math.log2(v / n) for v in c.values() if v), 4) if n else None


def check(path, data_ids, gold):
    rows = list(read_jsonl(path))
    ids = [r["sample_id"] for r in rows]
    dup = sum(v - 1 for v in collections.Counter(ids).values() if v > 1)
    pred = collections.Counter(r.get("technique") for r in rows)
    pred_mal = collections.Counter(r.get("technique") for r in rows if r.get("label"))
    pred_ben = collections.Counter(r.get("technique") for r in rows if not r.get("label"))
    g = collections.Counter(t for r in rows if r.get("label") for t in gold.get(r["sample_id"], []))
    top, top_n = pred.most_common(1)[0]
    m = [r for r in rows if r.get("label") and gold.get(r["sample_id"])]
    hit = sum((r.get("technique") or "")[:5] in gold[r["sample_id"]] for r in m)
    return {"file": path, "n": len(rows), "n_expected": len(data_ids), "n_ok": len(rows) == len(data_ids),
            "duplicates": dup, "missing": len(set(data_ids) - set(ids)), "extra": len(set(ids) - set(data_ids)),
            "n_distinct_pred": len(pred), "top_pred": top, "top_pred_share": round(top_n / len(rows), 4),
            "pred_entropy_bits": entropy(pred), "pred_dist_all": dict(pred.most_common()),
            "pred_dist_malicious": dict(pred_mal.most_common()), "pred_dist_benign": dict(pred_ben.most_common()),
            "gold_dist_malicious": dict(g.most_common()), "gold_entropy_bits": entropy(g),
            "ttp_n": len(m), "ttp_acc_top1": round(hit / len(m), 4) if m else None,
            "acc_if_always_top_pred": round(sum(top in gold[r["sample_id"]] for r in m) / len(m), 4) if m else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    data = list(read_jsonl(a.data))
    ids = [d["sample_id"] for d in data]
    gold = {d["sample_id"]: {t[:5] for t in d.get("techniques_base") or []} for d in data if d.get("label")}
    res = [check(p, ids, gold) for p in a.results]
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for r in res:
        print({k: r[k] for k in ("file", "n", "n_ok", "duplicates", "missing", "n_distinct_pred", "top_pred",
                                 "top_pred_share", "pred_entropy_bits", "ttp_acc_top1", "acc_if_always_top_pred")})


if __name__ == "__main__":
    main()
