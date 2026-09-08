"""AD-GEN -> tier-2 chain format (giong sd-chains.jsonl) + tap nho bench.
Adapter: narrative Sysmon -> chain list [{msg}], label 1 (attack/mal verdict) hoac 0 (benign).
Chi dung label: verdict (benign/suspicious/malicious) + mitre_techniques GT.
TAP NHO truoc (per-le): 200 mau can bang de bench nhanh."""
import json, re, sys, collections

SRC = [
    ("P1/Output/data/adgen/LAB.jsonl", "LAB"),
    ("P1/Output/data/adgen/REAL.jsonl", "REAL"),
]
OUT = "P1/Output/data/adgen-chains.jsonl"
SAMPLE_OUT = "P1/Output/data/adgen-chains-balanced.jsonl"
N_PER_CLASS = 100
SEED = 42

def parse_narrative(text):
    """Narrative AD-GEN: '<Entity>... [Timeline: T0] [T0 + 0.123s] -> Event 1 (User: X): <body>'
    Body chua cac dong 'Key: Value' ngan cach ' | '. Rut image + cmd + ten event thanh msg."""
    events = re.split(r"\[T0? ?\+ ?[\d.]+s\] -> ", text)
    chain = []
    for ev in events[1:]:
        m = re.match(r"Event (\d+) \(User: ([^)]+)\): (.*)", ev, re.DOTALL)
        if not m:
            continue
        eid, user, body = m.group(1), m.group(2), m.group(3)
        img = re.search(r"Image: ([^|]+)", body)
        cmdline = re.search(r"CommandLine: ([^|]+)", body)
        etype = re.search(r"EventType: ([^|]+)", body)
        name = etype.group(1).strip() if etype else f"Event{eid}"
        parts = [name]
        if img:
            parts.append(img.group(1).strip())
        if cmdline:
            parts.append("cmd: " + cmdline.group(1).strip())
        chain.append({"eid": int(eid), "msg": " | ".join(parts)[:300]})
    return chain

def main():
    n_in = n_out = 0
    recs = []
    stats = collections.Counter()
    for path, env in SRC:
        try:
            f = open(path, encoding="utf-8")
        except FileNotFoundError:
            print(f"MISS {path}", flush=True)
            continue
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                # xuong dong bi xuyen vao string (Windows download) -> dong phu; bo qua an toan
                stats["bad_json_line"] += 1
                continue
            lab = o.get("label", {})
            verdict = str(lab.get("verdict", "benign")).lower()
            mal = 0 if verdict in ("benign", "normal", "low") else 1
            techs = [t for t in (lab.get("mitre_techniques") or []) if str(t).startswith("T")]
            chain = parse_narrative(o.get("narrative", ""))
            if not chain:
                stats[f"{env}:empty_chain"] += 1
                continue
            recs.append({
                "nid": o.get("sample_id"),
                "label": mal,
                "self_label": chain[-1]["msg"] if chain else None,
                "parent_chain": chain,
                "event_seq": [c["eid"] for c in chain],
                "src": f"adgen_{env}",
                "technique_id": "+".join(techs) if techs else "",
            })
            stats[f"{env}:{'mal' if mal else 'ben'}"] += 1
            n_out += 1
        f.close()
    mal = [r for r in recs if r["label"] == 1]
    ben = [r for r in recs if r["label"] == 0]
    with open(OUT, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import random
    random.seed(SEED)
    s = random.sample(mal, min(N_PER_CLASS, len(mal))) + random.sample(ben, min(N_PER_CLASS, len(ben)))
    random.shuffle(s)
    with open(SAMPLE_OUT, "w", encoding="utf-8") as f:
        for r in s:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"in={n_in} out={n_out} (mal {len(mal)} / ben {len(ben)})", flush=True)
    print(dict(stats.most_common(12)), flush=True)
    print(f"SAVED {OUT} + {SAMPLE_OUT} (n={len(s)})", flush=True)

if __name__ == "__main__":
    sys.exit(main())
