import argparse
import collections
import json
import statistics


def pct(a, b):
    return round(a / b, 4) if b else None


def q(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="P1/Output/data/adgen-v2.jsonl")
    ap.add_argument("--out", default="P1/Output/results_phase2/adgen-hint-stats.json")
    a = ap.parse_args()

    by_env = collections.defaultdict(lambda: collections.Counter())
    verdicts, risks = collections.Counter(), collections.Counter()
    field_cnt, n_events_all = collections.Counter(), 0
    n_ev, n_chars = [], []
    per_tech = collections.defaultdict(lambda: collections.Counter())
    mal = ben = 0
    mal_tech = mal_cov = mal_subset = mal_overlap = 0
    ben_hint = ben_inline = 0
    ev_gt_sizes, ev_gt_frac = [], []
    lab_not_in_hints = collections.Counter()

    with open(a.data, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            env, y = o["env"], o["label"]
            by_env[env]["mal" if y else "ben"] += 1
            verdicts[str(o.get("verdict"))] += 1
            risks[str(o.get("risk_level"))] += 1
            evs = o["events"]
            n_ev.append(len(evs))
            n_chars.append(len(o["text"]))
            for e in evs:
                n_events_all += 1
                for k in e["fields"]:
                    field_cnt[k] += 1
            inline = {h[:5] for e in evs for h in e["hints"]}
            rec_h = {str(h)[:5] for h in o.get("sysmon_hints", [])}
            hints = inline | rec_h
            lab = set(o["techniques_base"])
            if y:
                mal += 1
                if not lab:
                    continue
                mal_tech += 1
                gt = o["evidence_gt_events"]
                if gt:
                    mal_cov += 1
                    ev_gt_sizes.append(len(gt))
                    ev_gt_frac.append(len(gt) / max(len(evs), 1))
                if lab <= hints:
                    mal_subset += 1
                if lab & hints:
                    mal_overlap += 1
                for t in lab:
                    per_tech[t]["n"] += 1
                    per_tech[t]["covered"] += int(any(t in {h[:5] for h in evs[i]["hints"]} for i in gt))
                    per_tech[t]["in_hints"] += int(t in hints)
                    if t not in hints:
                        lab_not_in_hints[t] += 1
            else:
                ben += 1
                ben_hint += int(bool(hints))
                ben_inline += int(bool(inline))

    res = {
        "n": mal + ben,
        "by_env": {k: dict(v) for k, v in by_env.items()},
        "verdicts": dict(verdicts.most_common()),
        "risk_levels": dict(risks.most_common()),
        "malicious": {
            "n": mal,
            "with_technique": mal_tech,
            "evidence_gt_coverage": pct(mal_cov, mal_tech),
            "evidence_events_median": statistics.median(ev_gt_sizes) if ev_gt_sizes else None,
            "evidence_events_frac_median": round(statistics.median(ev_gt_frac), 4) if ev_gt_frac else None,
            "label_subset_of_hints": pct(mal_subset, mal_tech),
            "label_overlap_hints": pct(mal_overlap, mal_tech),
            "label_techs_not_in_hints_top": dict(lab_not_in_hints.most_common(10)),
        },
        "benign": {
            "n": ben,
            "has_any_hint": pct(ben_hint, ben),
            "has_inline_hint": pct(ben_inline, ben),
        },
        "per_technique": {t: {"n": c["n"], "coverage": pct(c["covered"], c["n"]),
                              "in_hints": pct(c["in_hints"], c["n"])}
                          for t, c in sorted(per_tech.items(), key=lambda kv: -kv[1]["n"])[:25]},
        "events_per_sample": {"p50": q(n_ev, .5), "p90": q(n_ev, .9), "p99": q(n_ev, .99), "max": max(n_ev) if n_ev else None},
        "text_chars": {"p50": q(n_chars, .5), "p90": q(n_chars, .9), "p99": q(n_chars, .99)},
        "field_presence": {k: pct(v, n_events_all) for k, v in field_cnt.most_common()},
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    m = res["malicious"]
    print(json.dumps(res, indent=1, ensure_ascii=False)[:4000])
    print("GATE evidence_gt_coverage =", m["evidence_gt_coverage"],
          "| benign_has_hint =", res["benign"]["has_any_hint"],
          "| label_subset_of_hints =", m["label_subset_of_hints"])


if __name__ == "__main__":
    main()
