"""Mo ta tap attack_data: nguon, so mau, phan bo technique, ty le trong KB, trung voi REAL test.

  python attackdata_stats.py --data attackdata.jsonl --manifest attackdata.jsonl.manifest.json \
      --real REAL_test_matched.jsonl --out attackdata_stats.json
"""
import argparse
import collections
import json
from pathlib import Path

from eve import read_jsonl
from kb import KB


def pct(a, b):
    return round(a / b, 4) if b else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--real", required=True, help="REAL test (vd REAL_test_matched.jsonl) de do trung lap")
    ap.add_argument("--kb", default=str(Path(__file__).with_name("tech_preconditions.json")))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    kb = KB(a.kb)
    rows = list(read_jsonl(a.data))
    man = json.loads(Path(a.manifest).read_text(encoding="utf-8")) if a.manifest else {}

    tech = collections.Counter()
    sub = collections.Counter()
    src = collections.Counter()
    ds = collections.Counter()
    n_ev = []
    eids = collections.Counter()
    in_kb = in_c = gold_in_c = 0
    for r in rows:
        base = sorted(set(r.get("techniques_base") or []))
        tech.update(base)
        sub.update(t for t in r.get("techniques") or [] if "." in t)
        src[r.get("label_source")] += 1
        ds[r.get("dataset")] += 1
        n_ev.append(len(r["events"]))
        eids.update(e.get("eid") for e in r["events"])
        cands = {w.technique for w in kb.witnesses(r["events"])}
        cov = {t for t in base if kb.covers(t)}
        in_kb += bool(cov)
        in_c += bool(cands)
        gold_in_c += bool(cov & cands)

    real = list(read_jsonl(a.real))
    real_tech = collections.Counter(t for r in real if r.get("label") for t in set(r.get("techniques_base") or []))
    real_text = {r.get("text") for r in real}
    shared = sorted(set(tech) & set(real_tech))
    alerts_shared = sum(bool(set(r.get("techniques_base") or []) & set(real_tech)) for r in rows)
    n = len(rows)
    n_ev.sort()
    out = {
        "source": {"repo": "https://github.com/splunk/attack_data", "selection_mode": man.get("mode"),
                   "datasets_selected": man.get("datasets_selected") if isinstance(man.get("datasets_selected"), int)
                   else len(man.get("datasets_selected") or []) or None,
                   "datasets_with_alerts": len(ds), "file_stats": man.get("file_stats"),
                   "label": "nhan technique muc file tu .yml cua dataset (doc lap voi nhan LLM cua AD-GEN)"},
        "n_alerts": n, "label_malicious": sum(int(r.get("label") or 0) for r in rows),
        "by_label_source": dict(src),
        "events_per_alert": {"min": n_ev[0], "median": n_ev[n // 2], "max": n_ev[-1], "mean": round(sum(n_ev) / n, 2)},
        "sysmon_eid": dict(sorted(eids.items(), key=lambda x: -x[1])),
        "n_techniques": len(tech), "n_subtechniques": len(sub),
        "technique_distribution": dict(tech.most_common()),
        "top1_technique_share": pct(tech.most_common(1)[0][1], n) if tech else None,
        "kb": {"kb_version": kb.version, "kb_techniques": kb.techniques,
               "alerts_technique_in_kb": in_kb, "kb_scope": pct(in_kb, n),
               "alerts_nonempty_C": in_c, "nonempty_C_rate": pct(in_c, n),
               "kb_recall_in_scope": pct(gold_in_c, in_kb),
               "techniques_in_kb": sorted(t for t in tech if kb.covers(t))},
        "overlap_real_test": {"real_file": a.real, "real_n": len(real),
                              "real_malicious_techniques": len(real_tech),
                              "shared_techniques": shared, "n_shared_techniques": len(shared),
                              "alerts_with_technique_seen_in_real": alerts_shared,
                              "share_alerts_with_technique_seen_in_real": pct(alerts_shared, n),
                              "techniques_only_in_attackdata": sorted(set(tech) - set(real_tech)),
                              "exact_text_duplicates_with_real": sum(r.get("text") in real_text for r in rows)},
    }
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_alerts", "by_label_source", "n_techniques")}, indent=1))
    print(json.dumps(out["kb"], indent=1))
    print(json.dumps({k: v for k, v in out["overlap_real_test"].items() if k != "techniques_only_in_attackdata"}, indent=1))


if __name__ == "__main__":
    main()
