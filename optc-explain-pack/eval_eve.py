import argparse
import collections
import json
import random
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score

from eve import read_jsonl
from kb import KB


def safe(f, *a):
    try:
        v = f(*a)
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def boot(values, fn, n=1000, seed=0):
    if not values or n <= 0:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        s = [values[rng.randrange(len(values))] for _ in values]
        v = fn(s)
        if v is not None:
            stats.append(v)
    if not stats:
        return None
    stats.sort()
    return [round(stats[int(0.025 * (len(stats) - 1))], 4), round(stats[int(0.975 * (len(stats) - 1))], 4)]


def auc_of(pairs):
    ys = [y for y, _ in pairs]
    if len(set(ys)) < 2:
        return None
    return roc_auc_score(ys, [s for _, s in pairs])


def pick_threshold(dev):
    pairs = [(int(r["label"]), r["verdict_margin"]) for r in dev if r.get("label") is not None]
    cands = sorted({s for _, s in pairs})
    best, thr = -1, 0.0
    P = sum(y for y, _ in pairs) or 1
    N = (len(pairs) - sum(y for y, _ in pairs)) or 1
    for c in cands:
        tp = sum(1 for y, s in pairs if y and s >= c)
        fp = sum(1 for y, s in pairs if not y and s >= c)
        j = tp / P - fp / N
        if j > best:
            best, thr = j, c
    return thr


def length_baselines(rs, data):
    ys, ne, nc = [], [], []
    for r in rs:
        if r.get("label") is None:
            continue
        ys.append(int(r["label"]))
        ne.append(r.get("n_events") or 0)
        nc.append(len((data.get(r["sample_id"]) or {}).get("text", "")))
    if len(set(ys)) < 2:
        return {}
    return {"auc_length_events": float(roc_auc_score(ys, ne)),
            "auc_length_chars": float(roc_auc_score(ys, nc)) if any(nc) else None}


def verdict_metrics(rs, thr, nboot):
    pairs = [(int(r["label"]), r["verdict_margin"]) for r in rs if r.get("label") is not None]
    if not pairs:
        return {}
    y = [a for a, _ in pairs]
    s = [b for _, b in pairs]
    pred = [int(v >= thr) for v in s]
    tp = sum(1 for a, b in zip(y, pred) if a and b)
    fp = sum(1 for a, b in zip(y, pred) if not a and b)
    fn = sum(1 for a, b in zip(y, pred) if a and not b)
    tn = sum(1 for a, b in zip(y, pred) if not a and not b)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(pairs), "pos": sum(y), "threshold": thr,
        "auc": safe(auc_of, pairs), "auc_ci": boot(pairs, lambda x: auc_of(x), nboot),
        "ap": safe(average_precision_score, y, s) if len(set(y)) > 1 else None,
        "acc": (tp + tn) / len(y), "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "tpr": rec, "fpr": fp / (fp + tn) if fp + tn else None,
        "tie_rate": mean([int(abs(r["verdict_margin_raw"]) < 1e-6) for r in rs]),
        "length_only_auc": safe(auc_of, [(int(r["label"]), r.get("n_events", 0)) for r in rs if r.get("label") is not None]),
    }


def ttp_metrics(rs, kb, nboot):
    m = [r for r in rs if r.get("label") and r.get("gold_techniques")]
    if not m:
        return {}
    gold = [set(r["gold_techniques"]) for r in m]
    freq = collections.Counter(t for g in gold for t in g)
    maj = freq.most_common(1)[0][0]
    hit = [int((r.get("technique") or "")[:5] in g) for r, g in zip(m, gold)]
    top3 = []
    for r, g in zip(m, gold):
        p = r.get("technique_probs") or {}
        top = sorted(p, key=p.get, reverse=True)[:3] if p else ([r["technique"]] if r.get("technique") else [])
        top3.append(int(bool({t[:5] for t in top} & g)))
    scope = [int(any(kb.covers(t) for t in g)) for g in gold]
    in_scope = [h for h, s in zip(hit, scope) if s]
    cand = [int(bool(set(r.get("kb_candidates") or []) & g)) for r, g in zip(m, gold)]
    cand_in_scope = [c for c, s in zip(cand, scope) if s]
    return {
        "n": len(m), "acc_top1": mean(hit), "acc_top1_ci": boot(hit, mean, nboot), "acc_top3": mean(top3),
        "majority_technique": maj, "acc_majority": mean([int(maj in g) for g in gold]),
        "kb_scope": mean(scope), "acc_top1_in_scope": mean(in_scope), "n_in_scope": len(in_scope),
        "kb_recall_in_scope": mean(cand_in_scope),
        "undetermined_rate": mean([int(r.get("status") == "undetermined") for r in m]),
        "no_prediction_rate": mean([int(not r.get("technique")) for r in m]),
    }


def evidence_metrics(rs, data, gold, nboot):
    out = {}
    has_ev = [r for r in rs if r.get("technique")]
    out["entailment_ok"] = mean([int(r.get("entailment_ok", False)) for r in has_ev])
    out["minimal_ok"] = mean([int(r.get("minimal_ok", False)) for r in has_ev])
    gen = [r for r in rs if "n_evidence_generated" in r]
    if gen:
        g = sum(r["n_evidence_generated"] for r in gen)
        out["generated_evidence_fabricated"] = sum(r["n_evidence_fabricated"] for r in gen) / g if g else None
        out["generated_no_parseable_evidence"] = mean([int(r["n_evidence_generated"] == 0) for r in gen])
        v = [r["technique_valid_id"] for r in gen if r.get("technique_valid_id") is not None]
        out["generated_valid_technique_id"] = mean([int(x) for x in v])
        out["unsupported_technique"] = mean([int(bool(r.get("technique")) and not r.get("entailment_ok")) for r in gen])
    rows, rows_nt, base_nt = [], [], []
    for r in rs:
        d = data.get(r["sample_id"])
        if not d or not r.get("label"):
            continue
        gt = set(d.get("evidence_gt_events") or [])
        if not gt:
            continue
        pe = {it["event"] for it in (r.get("evidence") or [])}
        p = len(pe & gt) / len(pe) if pe else 0.0
        q = len(pe & gt) / len(gt)
        rows.append((p, q))
        n_ev = len(d["events"])
        if n_ev >= 3 and len(gt) / n_ev <= 0.5:
            rows_nt.append((p, q))
            base_nt.append((len(gt) / n_ev, 1.0))
    if rows:
        out["hint_event_precision"] = mean([a for a, _ in rows])
        out["hint_event_precision_ci"] = boot([a for a, _ in rows], mean, nboot)
        out["hint_event_recall"] = mean([b for _, b in rows])
        out["hint_n"] = len(rows)
    if rows_nt:
        out["hint_nontrivial_precision"] = mean([a for a, _ in rows_nt])
        out["hint_nontrivial_recall"] = mean([b for _, b in rows_nt])
        out["hint_nontrivial_n"] = len(rows_nt)
        out["baseline_all_events_nontrivial_precision"] = mean([a for a, _ in base_nt])
    if gold:
        gp, gr, gt_hit = [], [], []
        for r in rs:
            g = gold.get(r["sample_id"])
            if not g or not g.get("evidence"):
                continue
            G = {(e["event"], e["field"]) for e in g["evidence"]}
            P = {(e["event"], e["field"]) for e in (r.get("evidence") or [])}
            gp.append(len(P & G) / len(P) if P else 0.0)
            gr.append(len(P & G) / len(G))
            if g.get("techniques"):
                gt_hit.append(int((r.get("technique") or "")[:5] in {t[:5] for t in g["techniques"]}))
        if gp:
            out["gold_field_precision"] = mean(gp)
            out["gold_field_precision_ci"] = boot(gp, mean, nboot)
            out["gold_field_recall"] = mean(gr)
            out["gold_n"] = len(gp)
            out["gold_technique_acc"] = mean(gt_hit)
    return out


def latency(rs):
    lat = sorted(r["latency_ms"] for r in rs if r.get("latency_ms") is not None)
    if not lat:
        return {}
    return {"mean_ms": mean(lat), "p50_ms": lat[len(lat) // 2], "p90_ms": lat[int(0.9 * (len(lat) - 1))]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--kb", default=str(Path(__file__).with_name("tech_preconditions.json")))
    ap.add_argument("--gold", default=None)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    kb = KB(a.kb)
    data = {d["sample_id"]: d for d in read_jsonl(a.data)}
    gold = {g["sample_id"]: g for g in read_jsonl(a.gold)} if a.gold else {}
    thr = pick_threshold(list(read_jsonl(a.dev))) if a.dev else 0.0
    table = []
    for path in a.results:
        rs = list(read_jsonl(path))
        if not rs:
            continue
        table.append({
            "file": path, "mode": rs[0].get("mode"), "model": rs[0].get("model"), "n": len(rs),
            "verdict": {**verdict_metrics(rs, thr, a.bootstrap), **length_baselines(rs, data)},
            "ttp": ttp_metrics(rs, kb, a.bootstrap),
            "evidence": evidence_metrics(rs, data, gold, a.bootstrap),
            "status": dict(collections.Counter(r.get("status") for r in rs)),
            "latency": latency(rs),
        })
    txt = json.dumps(table, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    print("\nmode        auc    auc_len fpr    ttp1   maj    ttp1_kb  kb_rec  entail  min    hintP  hintP_nt  gen_fab")
    for t in table:
        v, p, e = t["verdict"], t["ttp"], t["evidence"]
        f = lambda x: "  -   " if x is None else f"{x:.3f} "
        print(f"{t['mode']:<11} {f(v.get('auc'))} {f(v.get('auc_length_only_baseline'))} {f(v.get('fpr'))} {f(p.get('acc_top1'))} {f(p.get('acc_majority'))} "
              f"{f(p.get('acc_top1_in_scope'))}   {f(p.get('kb_recall_in_scope'))} {f(e.get('entailment_ok'))} "
              f"{f(e.get('minimal_ok'))} {f(e.get('hint_event_precision'))} {f(e.get('hint_nontrivial_precision'))}    "
              f"{f(e.get('generated_evidence_fabricated'))}")


if __name__ == "__main__":
    main()
