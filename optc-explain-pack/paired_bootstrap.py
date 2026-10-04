"""Bootstrap CI ghep cap cho TTP top-1 giua hai he thong tren cung tap mau (+ McNemar exact).

  python paired_bootstrap.py --pair eve=q05_REAL_eve.jsonl kb_only=kb_only_first.jsonl --name REAL --out ci.json
  --gold gold.jsonl : dung technique trong gold.jsonl thay cho gold_techniques cua ket qua
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from eve import read_jsonl
from kb import KB


def load(path, gold_map):
    out = {}
    for r in read_jsonl(path):
        g = gold_map.get(r["sample_id"]) if gold_map is not None else (
            {t[:5] for t in r.get("gold_techniques") or []} if r.get("label") else None)
        if not g:
            continue
        out[r["sample_id"]] = ((r.get("technique") or "")[:5] in g, g)
    return out


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def ci(x):
    return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]


def compare(na, fa, nb, fb, gold_map, kb, nboot, seed):
    A, B = load(fa, gold_map), load(fb, gold_map)
    ids = sorted(set(A) & set(B))
    res = {}
    for scope in ("all", "in_scope"):
        sel = [i for i in ids if scope == "all" or any(kb.covers(t) for t in A[i][1])]
        if not sel:
            continue
        a = np.array([A[i][0] for i in sel], dtype=float)
        b = np.array([B[i][0] for i in sel], dtype=float)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(sel), size=(nboot, len(sel)))
        ma, mb = a[idx].mean(1), b[idx].mean(1)
        d = ma - mb
        b01, b10 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        res[scope] = {"n": len(sel), f"acc_{na}": round(float(a.mean()), 4), f"ci_{na}": ci(ma),
                      f"acc_{nb}": round(float(b.mean()), 4), f"ci_{nb}": ci(mb),
                      "diff": round(float(a.mean() - b.mean()), 4), "diff_ci": ci(d),
                      "p_boot_two_sided": round(float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))), 4),
                      f"only_{na}_correct": b01, f"only_{nb}_correct": b10,
                      "p_mcnemar_exact": round(mcnemar_exact(b01, b10), 5)}
    return {"a": {"name": na, "file": fa}, "b": {"name": nb, "file": fb}, "n_common": len(ids), **res}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("NAME=A", "NAME=B"))
    ap.add_argument("--name", default="")
    ap.add_argument("--gold", default=None)
    ap.add_argument("--kb", default=str(Path(__file__).with_name("tech_preconditions.json")))
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    kb = KB(a.kb)
    gold_map = None
    if a.gold:
        gold_map = {g["sample_id"]: {t[:5] for t in g.get("techniques") or []} for g in read_jsonl(a.gold)}
        gold_map = {k: v for k, v in gold_map.items() if v}
    res = []
    for x, y in a.pair:
        na, fa = x.split("=", 1)
        nb, fb = y.split("=", 1)
        res.append(compare(na, fa, nb, fb, gold_map, kb, a.nboot, a.seed))
    out = {"name": a.name, "gold": a.gold or "gold_techniques trong file ket qua", "nboot": a.nboot, "comparisons": res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for r in res:
        print(a.name, r["a"]["name"], "vs", r["b"]["name"], json.dumps(r.get("all")))


if __name__ == "__main__":
    main()
