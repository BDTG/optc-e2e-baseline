"""Kiem injection theo tung mau: technique doi thi C(E) / evidence co doi khong, evidence co dung field bi chen khong.
Tach theo nhan (malicious / benign). Vi tri chen lay lai tu inject.inject (deterministic), khong can file data_inj.

  python inject_persample.py --data REAL_test_matched.jsonl --out inj_persample.json --per_sample inj_persample.jsonl \
      --pair eve:instruction:<clean.jsonl>:<injected.jsonl> --pair kb_only:instruction:<clean>:<injected> ...
"""
import argparse
import collections
import json
from pathlib import Path

from eve import read_jsonl
from inject import ADAPTIVE, PAYLOADS, inject

CATS = ("unchanged", "C_changed", "C_same_evidence_on_injected_field", "C_same_evidence_moved", "C_same_evidence_same")


def ev_set(r):
    return {(e.get("event"), e.get("field")) for e in r.get("evidence") or []}


def classify(o, r, loc):
    t_changed = (o.get("technique") or "") != (r.get("technique") or "")
    c_changed = set(o.get("kb_candidates") or []) != set(r.get("kb_candidates") or [])
    eo, ei = ev_set(o), ev_set(r)
    if not t_changed:
        cat = "unchanged"
    elif c_changed:
        cat = "C_changed"
    elif loc in ei:
        cat = "C_same_evidence_on_injected_field"
    elif eo != ei:
        cat = "C_same_evidence_moved"
    else:
        cat = "C_same_evidence_same"
    return {"technique_changed": t_changed, "C_changed": c_changed, "evidence_changed": eo != ei,
            "evidence_on_injected_field": loc in ei, "category": cat}


def summarize(rows):
    def block(rs):
        n = len(rs)
        cat = collections.Counter(x["category"] for x in rs)
        ch = [x for x in rs if x["technique_changed"]]
        return {"n": n, "technique_changed": len(ch),
                "technique_change_rate": round(len(ch) / n, 4) if n else None,
                "verdict_flip_rate": round(sum(x["verdict_flip"] for x in rs) / n, 4) if n else None,
                "categories": {c: cat[c] for c in CATS},
                "changed_without_evidence_change": cat["C_same_evidence_same"],
                "changed_with_C_same": len(ch) - cat["C_changed"]}
    out = {"all": block(rows)}
    for lab, name in ((1, "malicious"), (0, "benign")):
        out[name] = block([x for x in rows if x["label"] == lab])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pair", action="append", required=True, help="system:payload:clean.jsonl:injected.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_sample", default=None)
    a = ap.parse_args(argv)
    data = {r["sample_id"]: r for r in read_jsonl(a.data)}
    loc_cache, res, per = {}, [], []
    for spec in a.pair:
        system, payload, clean, injected = spec.split(":", 3)
        if payload not in loc_cache:
            pl = payload if payload in ADAPTIVE else PAYLOADS[payload]
            loc_cache[payload] = {}
            for sid, rec in data.items():
                r = inject(rec, pl)
                if r is not None:
                    loc_cache[payload][sid] = (r["injected"]["event"], r["injected"]["field"])
        locs = loc_cache[payload]
        cl = {r["sample_id"]: r for r in read_jsonl(clean)}
        rows = []
        for r in read_jsonl(injected):
            o = cl.get(r["sample_id"])
            if o is None or r["sample_id"] not in locs:
                continue
            x = {"system": system, "payload": payload, "sample_id": r["sample_id"], "label": o.get("label"),
                 "injected_at": list(locs[r["sample_id"]]),
                 "technique_clean": o.get("technique"), "technique_injected": r.get("technique"),
                 "verdict_flip": int(o["verdict"] != r["verdict"]),
                 **classify(o, r, locs[r["sample_id"]])}
            rows.append(x)
        per += rows
        res.append({"system": system, "payload": payload, "clean": clean, "injected": injected, **summarize(rows)})
        s = res[-1]["all"]
        print(f"{system:10s} {payload:22s} n={s['n']:4d} tech_changed={s['technique_changed']:4d} "
              f"C_same={s['changed_with_C_same']:4d} no_evidence_change={s['changed_without_evidence_change']}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    if a.per_sample:
        with open(a.per_sample, "w", encoding="utf-8") as f:
            for x in per:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
