"""Mo rong adgen_hint_stats: them coverage_by_env, evidence_nontrivial_detail, length_by_label.
Giu nguyen cac chi so goc (evidence_gt_coverage, benign_has_hint, label_subset_of_hints)."""
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


def med(xs):
    return round(statistics.median(xs), 4) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="P1/Output/data/adgen-v2.jsonl")
    ap.add_argument("--out", default="P1/Output/results_phase2/adgen-hint-stats-env.json")
    a = ap.parse_args()

    by_env = collections.defaultdict(lambda: collections.Counter())
    verdicts, risks = collections.Counter(), collections.Counter()
    mal = ben = 0

    # coverage_by_env: cac chi so GATE tach theo env
    env_cov = collections.defaultdict(lambda: collections.Counter())  # env -> {mal_tech, mal_cov, mal_subset, ben, ben_hint}

    # evidence_nontrivial_detail: trong so mal co gt evidence, evidence co "khu tru" event khong
    ev_with_gt = 0
    ev_all_events = 0        # gt bao TAT CA event (frac==1) -> tam thuong
    ev_nontrivial = 0        # gt la tap con thuc su (0<frac<1) -> co khu tru
    ev_sizes, ev_fracs = [], []
    ev_nontrivial_sizes = []

    # length_by_label
    len_ev = {0: [], 1: []}
    len_ch = {0: [], 1: []}

    with open(a.data, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            env, y = o["env"], o["label"]
            by_env[env]["mal" if y else "ben"] += 1
            verdicts[str(o.get("verdict"))] += 1
            risks[str(o.get("risk_level"))] += 1
            evs = o["events"]
            len_ev[y].append(len(evs))
            len_ch[y].append(len(o["text"]))

            inline = {h[:5] for e in evs for h in e["hints"]}
            rec_h = {str(h)[:5] for h in o.get("sysmon_hints", [])}
            hints = inline | rec_h
            lab = set(o["techniques_base"])
            gt = o.get("evidence_gt_events") or []

            if y:
                mal += 1
                if not lab:
                    continue
                env_cov[env]["mal_tech"] += 1
                if gt:
                    env_cov[env]["mal_cov"] += 1
                    ev_with_gt += 1
                    frac = len(gt) / max(len(evs), 1)
                    ev_sizes.append(len(gt))
                    ev_fracs.append(frac)
                    if frac >= 1.0:
                        ev_all_events += 1
                    else:
                        ev_nontrivial += 1
                        ev_nontrivial_sizes.append(len(gt))
                if lab <= hints:
                    env_cov[env]["mal_subset"] += 1
            else:
                ben += 1
                env_cov[env]["ben"] += 1
                env_cov[env]["ben_hint"] += int(bool(hints))

    coverage_by_env = {}
    for env, c in sorted(env_cov.items()):
        coverage_by_env[env] = {
            "mal_with_technique": c["mal_tech"],
            "evidence_gt_coverage": pct(c["mal_cov"], c["mal_tech"]),
            "label_subset_of_hints": pct(c["mal_subset"], c["mal_tech"]),
            "benign_n": c["ben"],
            "benign_has_hint": pct(c["ben_hint"], c["ben"]),
        }

    evidence_nontrivial_detail = {
        "mal_with_gt_evidence": ev_with_gt,
        "gt_covers_all_events": ev_all_events,
        "gt_localizes_subset_nontrivial": ev_nontrivial,
        "nontrivial_rate_among_gt": pct(ev_nontrivial, ev_with_gt),
        "gt_size_median": med(ev_sizes),
        "gt_size_p90": q(ev_sizes, .9),
        "gt_frac_median": med(ev_fracs),
        "gt_frac_p90": round(q(ev_fracs, .9), 4) if ev_fracs else None,
        "nontrivial_gt_size_median": med(ev_nontrivial_sizes),
    }

    length_by_label = {}
    for y, name in ((0, "benign"), (1, "malicious")):
        length_by_label[name] = {
            "n": len(len_ev[y]),
            "events": {"p50": q(len_ev[y], .5), "p90": q(len_ev[y], .9),
                       "p99": q(len_ev[y], .99), "max": max(len_ev[y]) if len_ev[y] else None},
            "text_chars": {"p50": q(len_ch[y], .5), "p90": q(len_ch[y], .9), "p99": q(len_ch[y], .99)},
        }

    res = {
        "n": mal + ben,
        "by_env": {k: dict(v) for k, v in by_env.items()},
        "coverage_by_env": coverage_by_env,
        "evidence_nontrivial_detail": evidence_nontrivial_detail,
        "length_by_label": length_by_label,
    }
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
