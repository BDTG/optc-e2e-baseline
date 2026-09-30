import argparse
import collections
import json
import zlib
from pathlib import Path

from sklearn.metrics import roc_auc_score

DEFAULT_EDGES = [1, 2, 3, 4, 5, 6, 8, 11, 21, 51, 101]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def bin_of(n, edges):
    b = 0
    for i, e in enumerate(edges):
        if n >= e:
            b = i
    return b


def order(ids, seed):
    return sorted(ids, key=lambda s: (zlib.crc32(f"{seed}:{s}".encode()), s))


def allocate(n, mal_bins, ben_bins):
    cap = {b: min(len(mal_bins.get(b, [])), len(ben_bins.get(b, []))) for b in mal_bins}
    total_cap = sum(cap.values())
    n = min(n, total_cap)
    weight = {b: len(v) for b, v in mal_bins.items() if cap[b] > 0}
    quota = {b: 0 for b in cap}
    left = n
    while left > 0 and weight:
        wsum = sum(weight.values())
        give = {b: max(1, int(left * w / wsum)) for b, w in weight.items()}
        for b in sorted(give, key=lambda k: -weight[k]):
            g = min(give[b], cap[b] - quota[b], left)
            quota[b] += g
            left -= g
            if left == 0:
                break
        weight = {b: w for b, w in weight.items() if quota[b] < cap[b]}
    return quota


def matched(pool, n, edges, seed, tag):
    mal_bins, ben_bins = collections.defaultdict(list), collections.defaultdict(list)
    for r in pool:
        (mal_bins if r["label"] else ben_bins)[bin_of(r["n_events"], edges)].append(r["id"])
    for d in (mal_bins, ben_bins):
        for b in d:
            d[b] = order(d[b], f"{seed}:{tag}:{b}")
    q = allocate(n, mal_bins, ben_bins)
    ids = []
    for b, k in q.items():
        ids += mal_bins[b][:k] + ben_bins[b][:k]
    return ids, q


def random_balanced(pool, n, seed, tag):
    mal = order([r["id"] for r in pool if r["label"]], f"{seed}:{tag}:m")[:n]
    ben = order([r["id"] for r in pool if not r["label"]], f"{seed}:{tag}:b")[:n]
    return mal + ben


def describe(ids, idx, edges, env):
    rs = [idx[(env, i)] for i in ids]
    y = [r["label"] for r in rs]
    hist = {"mal": collections.Counter(), "ben": collections.Counter()}
    for r in rs:
        hist["mal" if r["label"] else "ben"][bin_of(r["n_events"], edges)] += 1
    auc_len = roc_auc_score(y, [r["n_events"] for r in rs]) if len(set(y)) > 1 else None
    auc_chr = roc_auc_score(y, [r["n_chars"] for r in rs]) if len(set(y)) > 1 else None
    return {"n": len(rs), "mal": sum(y), "ben": len(y) - sum(y),
            "with_technique": sum(r["has_tech"] for r in rs if r["label"]),
            "nontrivial_evidence": sum(r["nontrivial"] for r in rs if r["label"]),
            "auc_length_events": auc_len, "auc_length_chars": auc_chr,
            "bins": {k: {str(edges[b]): v for b, v in sorted(c.items())} for k, c in hist.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--n_dev", type=int, default=200)
    ap.add_argument("--n_calib", type=int, default=1000)
    ap.add_argument("--n_nontrivial", type=int, default=500)
    ap.add_argument("--edges", type=int, nargs="+", default=DEFAULT_EDGES)
    ap.add_argument("--test_env", default="REAL")
    ap.add_argument("--calib_env", default="LAB")
    a = ap.parse_args(argv)
    edges = sorted(set(a.edges))

    idx = {}
    for r in read_jsonl(a.data):
        n_ev = len(r["events"])
        gt = r.get("evidence_gt_events") or []
        idx[(r["env"], r["sample_id"])] = {
            "id": r["sample_id"], "env": r["env"], "label": int(r["label"]), "n_events": n_ev,
            "n_chars": len(r.get("text", "")), "has_tech": int(bool(r.get("techniques_base"))),
            "nontrivial": int(bool(gt) and len(gt) < n_ev),
        }
    envs = collections.Counter((r["env"], r["label"]) for r in idx.values())
    if not any(e == a.test_env for e, _ in envs) or not any(e == a.calib_env for e, _ in envs):
        raise SystemExit(f"env missing, found {dict(envs)}; rerun convert with LAB/REAL file names")

    test_pool = [r for r in idx.values() if r["env"] == a.test_env]
    lab_pool = [r for r in idx.values() if r["env"] == a.calib_env]

    test_m, q_test = matched(test_pool, a.n_test, edges, a.seed, "test")
    used_test = set(test_m)
    test_r = random_balanced(test_pool, a.n_test, a.seed, "test_random")
    nt_pool = [r["id"] for r in test_pool if r["label"] and r["nontrivial"] and r["id"] not in used_test]
    nontriv = order(nt_pool, f"{a.seed}:nontrivial")[:a.n_nontrivial]

    dev_m, q_dev = matched(lab_pool, a.n_dev, edges, a.seed, "dev")
    used_dev = set(dev_m)
    calib_pool = [r["id"] for r in lab_pool if r["label"] and r["has_tech"] and r["id"] not in used_dev]
    calib = order(calib_pool, f"{a.seed}:calib")[:a.n_calib]

    splits = {
        f"{a.test_env}_test_matched": test_m,
        f"{a.test_env}_test_random": test_r,
        f"{a.test_env}_nontrivial": nontriv,
        f"{a.calib_env}_dev_matched": dev_m,
        f"{a.calib_env}_calib": calib,
    }
    assert not (set(test_m) & set(nontriv))
    assert not (set(dev_m) & set(calib))

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    member = collections.defaultdict(list)
    for name, ids in splits.items():
        senv = a.test_env if name.startswith(a.test_env) else a.calib_env
        for i in ids:
            member[(senv, i)].append(name)
    handles = {name: open(out / f"{name}.jsonl", "w", encoding="utf-8") for name in splits}
    try:
        for line_rec in read_jsonl(a.data):
            for name in member.get((line_rec["env"], line_rec["sample_id"]), []):
                handles[name].write(json.dumps(line_rec, ensure_ascii=False) + "\n")
    finally:
        for h in handles.values():
            h.close()

    manifest = {
        "source": str(a.data), "seed": a.seed, "edges": edges, "pool": {f"{e}|{y}": n for (e, y), n in sorted(envs.items())},
        "requested": {"n_test": a.n_test, "n_dev": a.n_dev, "n_calib": a.n_calib, "n_nontrivial": a.n_nontrivial},
        "splits": {name: describe(ids, idx, edges, a.test_env if name.startswith(a.test_env) else a.calib_env)
                   for name, ids in splits.items()},
        "overlap_test_matched_random": len(set(test_m) & set(test_r)),
    }
    (out / "splits_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: {x: v[x] for x in ("n", "mal", "ben", "with_technique", "nontrivial_evidence",
                                              "auc_length_events", "auc_length_chars")}
                      for k, v in manifest["splits"].items()}, indent=1))


if __name__ == "__main__":
    main()
