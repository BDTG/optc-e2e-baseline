"""Chuyen output annotate (eve mode json cua 7B teacher) -> gold.jsonl cho eval_eve --gold.
Moi dong gold: {sample_id, techniques:[T####], evidence:[{event, field}], verdict}.
Day la gold do model gan (silver), doc lap voi co che EVE constrained (mode json = unconstrained generative)."""
import argparse
import json
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", required=True, help="output eve mode json cua model annotator")
    ap.add_argument("--out", default="results/gold.jsonl")
    ap.add_argument("--source", default="qwen2.5-7b-json")
    a = ap.parse_args()
    n = n_tech = n_ev = 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in read_jsonl(a.annot):
            t = r.get("technique_sub") or r.get("technique")
            techs = [t] if t else []
            ev = [{"event": e["event"], "field": e["field"]} for e in (r.get("evidence") or [])]
            f.write(json.dumps({"sample_id": r["sample_id"], "techniques": techs, "evidence": ev,
                                "verdict": r.get("verdict_gen"), "source": a.source}, ensure_ascii=False) + "\n")
            n += 1
            n_tech += int(bool(techs))
            n_ev += int(bool(ev))
    print(json.dumps({"n": n, "with_technique": n_tech, "with_evidence": n_ev, "out": a.out}, indent=1))


if __name__ == "__main__":
    main()
