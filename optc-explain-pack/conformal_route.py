"""Dinh tuyen bang tin hieu conformal: giu ket qua SLM nho khi tap conformal chi co 1 technique,
nguoc lai escalate len model lon (vd 7B). So sanh nhom giu lai voi nhom escalate.

  python conformal_route.py --calib q05_LAB_calib_eve.jsonl --test q05_REAL_eve.jsonl --big q7b_REAL_eve.jsonl \
      --weighted --calib_data LAB_calib.jsonl --test_data REAL_test_matched.jsonl --out route.json \
      --lat_small q05_cpu_eve.jsonl --lat_big q7b_cpu_eve.jsonl
--lat_small/--lat_big: latency lay tu file do tren CPU (theo mau neu co, con lai dung trung binh cua file do).
"""
import argparse
import json
from pathlib import Path

from conformal import WeightedQuantile, domain_weights, quantile, rows
from eve import read_jsonl


def mean(x):
    return round(sum(x) / len(x), 4) if x else None


def latency_source(path):
    if not path:
        return lambda r, _: r.get("latency_ms", 0)
    d = {r["sample_id"]: r.get("latency_ms", 0) for r in read_jsonl(path)}
    m = sum(d.values()) / len(d)
    return lambda _, sid: d.get(sid, m)


def route(test_all, big, gold, qfn, rule, lat_s, lat_b):
    keep, esc = [], []
    for r in test_all:
        p = r.get("technique_probs") or {}
        q = qfn(r["sample_id"])
        S = {t for t, v in p.items() if 1.0 - v <= q}
        go = len(S) > 1 if rule == "ambiguous" else len(S) != 1
        (esc if go and r["sample_id"] in big else keep).append(r)
    ttp = lambda rs: [r for r in rs if r["sample_id"] in gold]
    hit = lambda r: int((r.get("technique") or "")[:5] in gold[r["sample_id"]])
    k, e = ttp(keep), ttp(esc)
    small_e = [hit(r) for r in e]
    big_e = [hit(big[r["sample_id"]]) for r in e]
    sys_hits = [hit(r) for r in k] + big_e
    lat = [lat_s(r, r["sample_id"]) for r in keep] + \
        [lat_s(r, r["sample_id"]) + lat_b(big[r["sample_id"]], r["sample_id"]) for r in esc]
    return {"rule": rule, "n_alerts": len(test_all), "escalate_rate": round(len(esc) / len(test_all), 4),
            "n_ttp": len(k) + len(e), "n_ttp_kept": len(k), "n_ttp_escalated": len(e),
            "acc_kept_small": mean([hit(r) for r in k]), "acc_escalated_small": mean(small_e),
            "acc_escalated_big": mean(big_e), "acc_system": mean(sys_hits),
            "recovered_by_big": sum(int(b and not s) for s, b in zip(small_e, big_e)),
            "harmed_by_big": sum(int(s and not b) for s, b in zip(small_e, big_e)),
            "latency_ms_mean_system": round(sum(lat) / len(lat), 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--big", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--calib_data", default=None)
    ap.add_argument("--test_data", default=None)
    ap.add_argument("--lat_small", default=None, help="file ket qua SLM nho do tren CPU (latency_ms)")
    ap.add_argument("--lat_big", default=None, help="file ket qua model lon do tren CPU (latency_ms)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    lat_s, lat_b = latency_source(a.lat_small), latency_source(a.lat_big)
    calib = rows(a.calib)
    test_all = list(read_jsonl(a.test))
    big = {r["sample_id"]: r for r in read_jsonl(a.big)}
    gold = {r["sample_id"]: {t[:5] for t in r["gold_techniques"]} for r in test_all
            if r.get("label") and r.get("gold_techniques")}
    hit = lambda r: int((r.get("technique") or "")[:5] in gold[r["sample_id"]])
    tt = [r for r in test_all if r["sample_id"] in gold]
    base = {"small_only": {"acc": mean([hit(r) for r in tt]),
                           "latency_ms_mean": round(sum(lat_s(r, r["sample_id"]) for r in test_all) / len(test_all), 1)},
            "big_only": {"acc": mean([hit(big[r["sample_id"]]) for r in tt if r["sample_id"] in big]),
                         "latency_ms_mean": round(sum(lat_b(r, r["sample_id"]) for r in big.values()) / len(big), 1)}}
    res = {"calib": a.calib, "test": a.test, "big": a.big, "latency_small": a.lat_small or a.test,
           "latency_big": a.lat_big or a.big, "baselines": base, "unweighted": [], "weighted": []}
    cal = [r["score"] for r in calib if r["inscope"]]
    for al in a.alphas:
        q = quantile(cal, al)
        for rule in ("ambiguous", "not_singleton"):
            res["unweighted"].append({"alpha": al, "qhat": q, **route(test_all, big, gold, lambda _: q, rule, lat_s, lat_b)})
    if a.weighted:
        ct = {d["sample_id"]: d.get("text", "") for d in read_jsonl(a.calib_data)}
        td = {d["sample_id"]: d.get("text", "") for d in read_jsonl(a.test_data)}
        cids = [r["id"] for r in calib if r["inscope"] and r["id"] in ct]
        tids = [r["sample_id"] for r in test_all if r["sample_id"] in td]
        wc, wt = domain_weights(cids, tids, ct, td)
        cl = [r for r in calib if r["inscope"] and r["id"] in wc]
        wq = WeightedQuantile([r["score"] for r in cl], [wc[r["id"]] for r in cl])
        for al in a.alphas:
            for rule in ("ambiguous", "not_singleton"):
                res["weighted"].append({"alpha": al, **route(test_all, big, gold, lambda i, al=al: wq(al, wt.get(i, 1.0)), rule,
                                                             lat_s, lat_b)})
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(base))
    for k in ("unweighted", "weighted"):
        for x in res[k]:
            print(k, {y: x[y] for y in ("alpha", "rule", "escalate_rate", "acc_kept_small", "acc_escalated_small",
                                          "acc_escalated_big", "acc_system", "latency_ms_mean_system")})


if __name__ == "__main__":
    main()
