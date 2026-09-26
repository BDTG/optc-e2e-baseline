import argparse
import collections
import json
import re
import sys
import zlib
import random
from pathlib import Path

EVENT_SPLIT = re.compile(r"\[T0 \+ ([\d.]+)s\] -> ")
EVENT_HEAD = re.compile(r"Event (\d+) \(User: ([^)]*)\): (.*)", re.DOTALL)
HINT = re.compile(r"\[hint:(T\d{4}(?:\.\d{3})?)\]\s*")
TECH = re.compile(r"T\d{4}(?:\.\d{3})?")

KEEP_FIELDS = [
    "EventType", "Image", "CommandLine", "ParentImage", "ParentCommandLine",
    "SourceImage", "TargetImage", "GrantedAccess", "CallTrace",
    "TargetObject", "Details", "NewName",
    "TargetFilename", "ImageLoaded", "Signed", "Signature",
    "PipeName", "QueryName", "QueryResults",
    "DestinationIp", "DestinationPort", "DestinationHostname", "Protocol",
    "StartModule", "StartFunction", "IntegrityLevel", "OriginalFileName",
]
SPAN_FIELDS = [
    "Image", "CommandLine", "ParentImage", "ParentCommandLine", "SourceImage",
    "TargetImage", "GrantedAccess", "TargetObject", "Details", "TargetFilename",
    "ImageLoaded", "PipeName", "QueryName", "DestinationIp", "DestinationPort",
    "DestinationHostname", "StartFunction",
]
BENIGN_VERDICTS = {"benign", "normal", "low", "none"}


def base(t):
    return t[:5]


def parse_fields(body):
    parts = [p.strip() for p in body.split(" | ")]
    title = parts[0].rstrip(":").strip() if parts else ""
    fields = {}
    for p in parts[1:]:
        if ": " in p:
            k, v = p.split(": ", 1)
            k = k.strip()
            if k in KEEP_FIELDS and v.strip():
                fields[k] = v.strip()
    return title, fields


def parse_narrative(text):
    pieces = EVENT_SPLIT.split(text)
    events = []
    for i in range(1, len(pieces) - 1, 2):
        t = float(pieces[i])
        m = EVENT_HEAD.match(pieces[i + 1].strip())
        if not m:
            continue
        eid, user, body = int(m.group(1)), m.group(2).strip(), m.group(3)
        hints = HINT.findall(body)
        body = HINT.sub("", body)
        title, fields = parse_fields(body)
        events.append({"idx": len(events), "t": t, "eid": eid, "user": user,
                       "title": title, "fields": fields, "hints": hints})
    return events


def render_event(e, max_val):
    kv = [f"{k}: {v[:max_val]}" for k, v in e["fields"].items()]
    return f"[E{e['idx']}] Event {e['eid']} {e['title']} | " + " | ".join(kv)


def render(events, max_val=200, max_chars=6000):
    lines = [render_event(e, max_val) for e in events]
    out, n = [], 0
    for ln in lines:
        if n + len(ln) + 1 > max_chars:
            out.append(f"[... {len(lines) - len(out)} events truncated]")
            break
        out.append(ln)
        n += len(ln) + 1
    return "\n".join(out)


def spans(events):
    res = []
    for e in events:
        for k in SPAN_FIELDS:
            v = e["fields"].get(k)
            if v:
                res.append({"event": e["idx"], "field": k, "value": v[:200]})
    return res


def is_malicious(lab):
    v = str(lab.get("verdict", "")).strip().lower()
    if v:
        return int(v not in BENIGN_VERDICTS)
    r = str(lab.get("risk_level", "Low")).strip().lower()
    return int(r != "low")


def convert(rec):
    lab = rec.get("label") or {}
    if isinstance(lab, str):
        try:
            lab = json.loads(lab)
        except json.JSONDecodeError:
            lab = {}
    events = parse_narrative(rec.get("narrative", ""))
    if not events:
        return None
    techs = [t for t in (lab.get("mitre_techniques") or []) if TECH.fullmatch(str(t))]
    tbase = sorted({base(t) for t in techs})
    ev_gt = [e["idx"] for e in events if {base(h) for h in e["hints"]} & set(tbase)]
    return {
        "sample_id": rec.get("sample_id"),
        "env": rec.get("environment", ""),
        "label": is_malicious(lab),
        "verdict": lab.get("verdict"),
        "risk_level": lab.get("risk_level"),
        "techniques": techs,
        "techniques_base": tbase,
        "tactics": [t for t in (lab.get("mitre_tactics") or []) if str(t).startswith("TA")],
        "actions": [a.get("tool_name") for a in (lab.get("recommended_actions") or []) if isinstance(a, dict)],
        "sysmon_hints": rec.get("sysmon_hints") or [],
        "events": events,
        "text": render(events),
        "spans": spans(events),
        "evidence_gt_events": ev_gt,
    }


def iter_jsonl(path, stats=None):
    buf = ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if buf:
                try:
                    yield json.loads(buf + line, strict=False)
                    buf = ""
                    if stats is not None:
                        stats["rejoined"] += 1
                    continue
                except json.JSONDecodeError:
                    pass
            try:
                obj = json.loads(line, strict=False)
                if buf and stats is not None:
                    stats["dropped"] += 1
                buf = ""
                yield obj
            except json.JSONDecodeError:
                buf = buf + line if buf else line
                if buf.count("\n") > 200:
                    buf = ""
                    if stats is not None:
                        stats["dropped"] += 1


def iter_hf():
    from datasets import load_dataset
    for r in load_dataset("namhop88/AD-GEN", split="train"):
        yield r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", action="store_true")
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--out", default="P1/Output/data/adgen-v2.jsonl")
    ap.add_argument("--balanced_out", default="P1/Output/data/adgen-v2-balanced.jsonl")
    ap.add_argument("--n_per_class", type=int, default=500)
    ap.add_argument("--balanced_env", default="REAL")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--env_map", nargs="*", default=[],
                    help="path_substring=ENV — gan env theo TEN FILE khi export khong phan biet "
                         "(AD-GEN LAB.jsonl/REAL.jsonl deu ghi environment='LAB'), vd: LAB.jsonl=LAB REAL.jsonl=REAL")
    a = ap.parse_args()

    rd = collections.Counter()
    env_map = []
    for spec in a.env_map:
        k, _, v = spec.partition("=")
        if k and v:
            env_map.append((k, v))

    def iter_inputs():
        if a.hf:
            for r in iter_hf():
                yield "", r
        else:
            for p in a.inputs:
                for r in iter_jsonl(p, rd):
                    yield p, r

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = n_empty = 0
    env_remap = collections.Counter()
    pool = {0: [], 1: []}
    with open(a.out, "w", encoding="utf-8") as f:
        for path, rec in iter_inputs():
            n_in += 1
            o = convert(rec)
            if o is None:
                n_empty += 1
                continue
            for k, v in env_map:
                if k in path:
                    if o["env"] != v:
                        env_remap[f"{o['env']}->{v}"] += 1
                    o["env"] = v
                    break
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            n_out += 1
            if o["env"] == a.balanced_env:
                pool[o["label"]].append(o["sample_id"])
    rng = random.Random(a.seed)
    pick = set()
    for k in (0, 1):
        ids = sorted(pool[k], key=lambda s: zlib.crc32(s.encode()))
        rng.shuffle(ids)
        pick.update(ids[:a.n_per_class])
    with open(a.out, encoding="utf-8") as fi, open(a.balanced_out, "w", encoding="utf-8") as fo:
        for line in fi:
            if json.loads(line)["sample_id"] in pick:
                fo.write(line)
    print(json.dumps({"in": n_in, "out": n_out, "empty": n_empty,
                      "env_remap": dict(env_remap),
                      "pool_" + a.balanced_env: {"ben": len(pool[0]), "mal": len(pool[1])},
                      "balanced": len(pick), "read": dict(rd)}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
