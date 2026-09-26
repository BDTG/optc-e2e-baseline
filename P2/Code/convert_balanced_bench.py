"""Chuyen adgen-v2.jsonl -> schema bench cu (nid/label/technique_id/parent_chain/event_seq)
de chay lai 6 run explain voi verdict THAT (2 lop).
Sample: 1000 ben + 1000 mal CO TECHNIQUE, env REAL, seed 42, deterministic."""
import json, random, collections

SRC = "P1/Output/data/adgen-v2.jsonl"
OUT = "P1/Output/data/adgen-balanced-bench.jsonl"
N_MAL = 1000
N_BEN = 1000
SEED = 42


def msg_of(e):
    parts = [e.get("title", "")]
    for k, v in (e.get("fields") or {}).items():
        v = str(v)[:200]
        parts.append(f"cmd: {v}" if k == "CommandLine" else f"{k}: {v}")
    return " | ".join(parts)


def convert(o):
    y = int(o["label"])
    techs = [t for t in (o.get("techniques") or []) if isinstance(t, str) and t[:1] == "T"]
    evs = o.get("events") or []
    return {
        "nid": o["sample_id"],
        "label": y,
        "technique_id": ", ".join(techs) if (y == 1 and techs) else "none",
        "parent_chain": [{"eid": e.get("idx", 0), "msg": msg_of(e)} for e in evs],
        "event_seq": [e.get("eid", 0) for e in evs],
        "src": "adgen-v2",
        "env": o.get("env", ""),
    }


def main():
    pool = {0: [], 1: []}
    cache = {}
    for line in open(SRC, encoding="utf-8"):
        o = json.loads(line)
        if o.get("env") != "REAL":
            continue
        y = int(o["label"])
        if y == 1 and not [t for t in (o.get("techniques") or []) if isinstance(t, str) and t[:1] == "T"]:
            continue
        pool[y].append(o["sample_id"])
        cache[o["sample_id"]] = o
    print("pool REAL (mal loc co-tech):", {k: len(v) for k, v in pool.items()}, flush=True)
    rng = random.Random(SEED)
    pick = []
    for k, n in ((0, N_BEN), (1, N_MAL)):
        ids = sorted(pool[k])
        rng.shuffle(ids)
        assert len(ids) >= n, f"pool lop {k} chi co {len(ids)}"
        pick += ids[:n]
    rows = [convert(cache[i]) for i in pick]
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lb = collections.Counter(r["label"] for r in rows)
    print(json.dumps({"out": OUT, "n": len(rows), "label": dict(lb)}, indent=1))


if __name__ == "__main__":
    main()
