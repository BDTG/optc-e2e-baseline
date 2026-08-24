"""
export_dataset.py — Buoc chuan bi du lieu cho fine-tune SLM.
Doc COMISET tho -> dung cay tien trinh -> moi cay thanh 1 mau
{text, label, meta:{host,ts,size}}. Cung schema voi export_optc_ecar.py.
Xuat JSONL de script train doc lai (khong phai parse 50M event moi lan train).

Chay: python export_dataset.py --data dataset_comillas2.json --out train_data.jsonl
"""
import argparse
import json
import re
import time
from collections import defaultdict
from datetime import datetime

from utils import basename

FIELD_MAP = {
    "host": ["host_name", "Host_name", "Computer", "hostname", "beat_name"],
    "guid": ["process_guid", "Process_guid", "ProcessGuid"],
    "pguid": ["process_parent_guid", "Process_parent_guid", "ParentProcessGuid"],
    "eid": ["event_id", "EventID", "event_original_code"],
    "image": ["process_name", "Process_name", "process_path", "Image"],
    "cmd": ["CommandLine", "command_line", "process_command_line"],
    "target_image": ["TargetImage", "target_image"],
    "target_file": ["TargetFilename", "TargetObject", "target_filename"],
    "pipe": ["PipeName", "pipe_name"],
    "time": ["@timestamp", "event_original_time", "event_recorded_time"],
    "log": ["log_name", "source_name", "_index"],
    "tech_id": ["rule_technique_id", "Rule_technique_id", "RuleName", "rule_name"],
}
REDTEAM_TECHNIQUES = {
    "T1003", "T1003.001", "T1003.002", "T1003.003", "T1550.002",
    "T1021", "T1021.002", "T1055", "T1490", "T1047",
    "T1059", "T1059.001", "T1543", "T1036", "T1204",
}
TECH_REGEX = re.compile(r"T\d{4}(?:\.\d{3})?", re.IGNORECASE)


def get(rec, key):
    if not isinstance(rec, dict):
        return None
    d = rec.get("_source", rec)
    for cand in FIELD_MAP[key]:
        if isinstance(d, dict) and cand in d and d[cand] not in (None, ""):
            return d[cand]
        # d la rec khi khong co "_source" -> khoi kiem tra lai, da xet o nhanh tren
        if d is not rec and cand in rec and rec[cand] not in (None, ""):
            return rec[cand]
    return None


def parse_time(t):
    if t is None:
        return 0.0
    if isinstance(t, (int, float)):
        return float(t)
    s = str(t)
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return 0.0


def norm_tech(t):
    if not t:
        return None
    m = TECH_REGEX.search(str(t))
    return m.group(0).upper() if m else None


def is_mal(rec, trust_all):
    tid = norm_tech(get(rec, "tech_id"))
    if tid is None:
        return 0
    if trust_all:
        return 1
    base = tid.split(".")[0]
    return 1 if (tid in REDTEAM_TECHNIQUES or base in REDTEAM_TECHNIQUES) else 0


def event_token(rec, with_cmd):
    """Token phang cho mot event. Neu with_cmd, them command-line rut gon (RQ4)."""
    eid = str(get(rec, "eid") or "?")
    img = basename(get(rec, "image"))
    tgt = ""
    ti, tf, pipe = get(rec, "target_image"), get(rec, "target_file"), get(rec, "pipe")
    if ti:
        tgt = basename(ti)
    elif pipe:
        tgt = "pipe:" + basename(pipe)
    elif tf:
        tgt = "file:" + basename(tf)
    tok = f"{img}|E{eid}|{tgt}"
    if with_cmd:
        cmd = str(get(rec, "cmd") or "").strip().lower()
        if cmd:
            # rut gon command-line: bo path dai, giu args
            cmd = re.sub(r'"?[a-z]:\\[^"]*\\', "", cmd)[:80]
            tok += f"|{cmd}"
    return tok


def _iter(path):
    """Uu tien ijson (streaming, khong load ca file JSON long vao RAM). Neu KHONG
    import duoc ijson, fallback sang doc NDJSON tung dong. Chi bat loi quanh viec
    import/mo file — neu ijson da bat dau doc roi loi giua chung (file hong), de loi
    do lo ra ngoai thay vi am tham doc lai tu dau (se dem trung moi record da doc)."""
    try:
        import ijson
    except ImportError:
        print("[parse] khong co ijson, fallback doc NDJSON tung dong", flush=True)
        ijson = None

    if ijson is not None:
        print("[parse] doc bang ijson (streaming)", flush=True)
        with open(path, "rb") as f:
            for _k, v in ijson.kvitems(f, ""):
                yield v
        return

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--trust-all-labels", action="store_true")
    ap.add_argument("--min-mal-events", type=int, default=10)
    ap.add_argument("--max-events", type=int, default=300)
    ap.add_argument("--with-cmd", action="store_true",
                    help="them command-line vao token (cho RQ4)")
    args = ap.parse_args()

    nodes = {}
    n = ns = ng = 0
    t_start = time.time()
    print(f"[parse] doc {args.data} ...", flush=True)
    for rec in _iter(args.data):
        n += 1
        if args.limit and n > args.limit:
            break
        if n % 2_000_000 == 0:
            elapsed = time.time() - t_start
            print(f"[parse] {n:,} events, {len(nodes):,} nodes... "
                  f"({n/max(elapsed,1e-9):,.0f} event/s, {elapsed:.0f}s)", flush=True)
        if "sysmon" not in str(get(rec, "log") or "").lower():
            ns += 1
            continue
        g = get(rec, "guid")
        if not g:
            ng += 1
            continue
        if g not in nodes:
            nodes[g] = {"guid": g, "pguid": get(rec, "pguid"),
                        "host": get(rec, "host") or "unknown",
                        "toks": [], "n_mal": 0, "ts": parse_time(get(rec, "time"))}
        nd = nodes[g]
        if len(nd["toks"]) < args.max_events:
            nd["toks"].append(event_token(rec, args.with_cmd))
        if is_mal(rec, args.trust_all_labels):
            nd["n_mal"] += 1
    print(f"[parse] events={n:,} nodes={len(nodes):,} skip_ns={ns:,} skip_ng={ng:,}", flush=True)

    # dung cay
    children = defaultdict(list)
    for g, nd in nodes.items():
        if nd["pguid"] and nd["pguid"] in nodes:
            children[nd["pguid"]].append(g)
    roots = [g for g, nd in nodes.items() if not (nd["pguid"] and nd["pguid"] in nodes)]

    written = pos = 0
    with open(args.out, "w", encoding="utf-8") as fo:
        for root in roots:
            members, stack, seen = [], [(root, 0)], set()
            while stack:
                g, d = stack.pop()
                if g in seen:
                    continue
                seen.add(g)
                members.append((g, d))
                for c in children.get(g, []):
                    stack.append((c, d + 1))
            toks = []
            n_mal = 0
            for g, _d in sorted(members, key=lambda x: x[1]):
                toks.extend(nodes[g]["toks"])
                n_mal += nodes[g]["n_mal"]
            if not toks:
                continue
            label = 1 if n_mal >= args.min_mal_events else 0
            rec_out = {"text": " ".join(toks), "label": label,
                       "meta": {"host": nodes[root]["host"], "ts": nodes[root]["ts"],
                                "size": len(members)}}
            fo.write(json.dumps(rec_out) + "\n")
            written += 1
            pos += label
    print(f"[out] viet {written:,} cay -> {args.out}  (doc={pos:,}, {100*pos/max(written,1):.2f}%)")


if __name__ == "__main__":
    main()
