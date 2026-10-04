"""Clause doi chung cheo field: KB co giu dung ngu nghia field khong, hay chi khop tu khoa.
Sinh bien the KB tu mot file tech_preconditions.json, giu nguyen gia tri dieu kien va eid:
  anyfield     - moi dieu kien duoc thu tren MOI field cua event (khop tu khoa bat ke field)
  cross<seed>  - moi ten field duoc doi sang mot field KHAC (hoan vi khong diem bat dong, ca trong cmp)

  python kb_field_control.py --kb tech_preconditions.json --data REAL_test_matched.jsonl --outdir kb_ctrl --seeds 0 1 2
Sau do chay `eve.py kbstats` / `eve.py run --mode kb_only` voi tung --kb trong outdir.
"""
import argparse
import copy
import json
import random
from pathlib import Path

from eve import read_jsonl


def kb_fields(kb):
    names = set()
    for t in kb["techniques"].values():
        for c in t["clauses"]:
            for k in c.get("all", {}):
                names.update(k.split("|"))
            for m in c.get("cmp", []):
                names.update((m["a"], m["b"]))
    return names


def data_fields(paths):
    names = set()
    for p in paths:
        for rec in read_jsonl(p):
            for e in rec["events"]:
                names.update(e.get("fields", {}))
    return names


def derangement(items, seed):
    items = sorted(items)
    rng = random.Random(seed)
    while True:
        perm = items[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(items, perm)):
            return dict(zip(items, perm))


def remap(kb, fn_key, fn_cmp, tag):
    out = copy.deepcopy(kb)
    out["version"] = f"{kb.get('version')}-{tag}"
    out["note"] = f"Bien the doi chung '{tag}' sinh boi kb_field_control.py tu KB {kb.get('version')}."
    for t in out["techniques"].values():
        for c in t["clauses"]:
            c["all"] = {fn_key(k): v for k, v in c.get("all", {}).items()}
            if c.get("cmp"):
                c["cmp"] = [fn_cmp(m) for m in c["cmp"]]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--data", nargs="+", required=True, help="du lieu de lay tap ten field co that")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args(argv)
    kb = json.loads(Path(a.kb).read_text(encoding="utf-8"))
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    kf, df = kb_fields(kb), data_fields(a.data)
    universe = sorted(kf | df)
    any_key = "|".join(universe)
    variants = {"anyfield": remap(kb, lambda k: any_key, lambda m: m, "anyfield")}
    for s in a.seeds:
        perm = derangement(universe, s)
        variants[f"cross{s}"] = remap(
            kb, lambda k, p=perm: "|".join(p[n] for n in k.split("|")),
            lambda m, p=perm: {**m, "a": p[m["a"]], "b": p[m["b"]]}, f"cross{s}")
        (out / f"cross{s}_mapping.json").write_text(json.dumps({k: perm[k] for k in sorted(kf)}, indent=1),
                                                   encoding="utf-8")
    for name, v in variants.items():
        (out / f"kb_{name}.json").write_text(json.dumps(v, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = {"kb": a.kb, "kb_fields": sorted(kf), "n_universe": len(universe), "variants": sorted(variants)}
    (out / "variants.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("n_universe", "variants")}))


if __name__ == "__main__":
    main()
