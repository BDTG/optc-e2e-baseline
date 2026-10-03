import argparse
import collections
import glob
import html
import json
import re
import sys
import zlib
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eve import render

DEFAULT_TECHNIQUES = ["T1003", "T1036", "T1047", "T1055", "T1059", "T1218", "T1490", "T1543", "T1553", "T1574",
                      "T1547", "T1562", "T1070", "T1053", "T1021", "T1546", "T1112", "T1548"]
KEEP_FIELDS = {
    "EventType", "Image", "CommandLine", "ParentImage", "ParentCommandLine", "SourceImage", "TargetImage",
    "GrantedAccess", "CallTrace", "TargetObject", "Details", "NewName", "TargetFilename", "ImageLoaded",
    "Signed", "Signature", "PipeName", "QueryName", "QueryResults", "DestinationIp", "DestinationPort",
    "DestinationHostname", "Protocol", "StartModule", "StartFunction", "IntegrityLevel", "OriginalFileName",
}
TITLES = {1: "Process Create", 3: "Network connection", 5: "Process terminated", 6: "Driver loaded",
          7: "Image loaded", 8: "CreateRemoteThread", 9: "RawAccessRead", 10: "Process accessed",
          11: "File created", 12: "Registry object added or deleted", 13: "Registry value set",
          14: "Registry object renamed", 15: "File stream created", 17: "Pipe Created", 18: "Pipe Connected",
          19: "WmiEventFilter", 20: "WmiEventConsumer", 21: "WmiEventConsumerToFilter", 22: "Dns query",
          23: "File Delete", 25: "Process Tampering", 26: "File Delete logged"}
AGENT_RE = re.compile(r"\\(splunk[^\\]*|sysmon64|sysmon|msmpeng|mpcmdrun|nissrv|senseir|mssense)\.exe$", re.I)
ROOT_RE_DEFAULT = r"invoke-atomictest|\\atomicredteam\\|\\atomics\\|atomic-red-team"
EVENT_RE = re.compile(r"<Event\b.*?</Event>", re.S)
EID_RE = re.compile(r"<EventID[^>]*>\s*(\d+)\s*</EventID>")
DATA_RE = re.compile(r"<Data\s+Name\s*=\s*['\"]([^'\"]+)['\"]\s*(?:/>|>(.*?)</Data>)", re.S)
USERID_RE = re.compile(r"<Security\s+UserID\s*=\s*['\"]([^'\"]+)['\"]")
PROVIDER_RE = re.compile(r"<Provider\s+Name\s*=\s*['\"]([^'\"]+)['\"]")
RULE_TECH_RE = re.compile(r"technique_id=(T\d{4}(?:\.\d{3})?)", re.I)
MAX_EVENTS = 149


def is_lfs_pointer(path):
    try:
        with open(path, "rb") as f:
            return f.read(40).startswith(b"version https://git-lfs")
    except OSError:
        return True


def parse_time(s):
    if not s:
        return None
    s = s.strip().replace("T", " ").rstrip("Z")
    if "." in s:
        head, frac = s.split(".", 1)
        s = f"{head}.{frac[:6]}"
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def parse_events(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out = []
    for m in EVENT_RE.finditer(text):
        block = m.group(0)
        prov = PROVIDER_RE.search(block)
        if prov and "sysmon" not in prov.group(1).lower():
            continue
        eid_m = EID_RE.search(block)
        if not eid_m:
            continue
        data = {k: html.unescape(v or "").strip() for k, v in DATA_RE.findall(block)}
        uid = USERID_RE.search(block)
        out.append({"eid": int(eid_m.group(1)), "data": data, "user": data.get("User") or (uid.group(1) if uid else ""),
                    "time": parse_time(data.get("UtcTime"))})
    return out


def group_key(e):
    d = e["data"]
    if e["eid"] == 10:
        return d.get("SourceProcessGUID") or d.get("SourceProcessGuid")
    if e["eid"] == 8:
        return d.get("SourceProcessGuid") or d.get("SourceProcessGUID")
    return d.get("ProcessGuid")


def group_image(evs):
    for e in evs:
        d = e["data"]
        img = d.get("Image") or d.get("SourceImage")
        if img:
            return img
    return ""


def group_signature(evs):
    for e in evs:
        if e["eid"] == 1:
            d = e["data"]
            return (d.get("Image", "").lower(), re.sub(r"\s+", " ", d.get("CommandLine", "").lower()).strip())
    return (group_image(evs).lower(), "")


def load_yml(path):
    try:
        d = yaml.safe_load(open(path, encoding="utf-8"))
    except Exception:
        return None
    return d if isinstance(d, dict) else None


def sysmon_files(yml_path, d):
    base = Path(yml_path).parent
    out = []
    for x in d.get("datasets") or []:
        if not isinstance(x, dict):
            continue
        src = str(x.get("source", "")) + " " + str(x.get("sourcetype", ""))
        if "sysmon" in src.lower() and "linux" not in src.lower():
            p = str(x.get("path", ""))
            out.append(base / Path(p).name)
    for url in d.get("dataset") or []:
        u = str(url)
        if "sysmon" in u.lower() and "linux" not in u.lower():
            out.append(base / u.rsplit("/", 1)[-1])
    seen, res = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            res.append(p)
    return res


def select_datasets(root, techniques):
    want = {t[:5] for t in techniques}
    res = []
    for y in sorted(glob.glob(str(Path(root) / "datasets" / "attack_techniques" / "*" / "*" / "*.yml"))):
        d = load_yml(y)
        if not d:
            continue
        techs = [str(t) for t in (d.get("mitre_technique") or d.get("technique") or [])]
        techs = [t for t in techs if re.fullmatch(r"T\d{4}(\.\d{3})?", t)]
        if not techs or not ({t[:5] for t in techs} & want):
            continue
        files = sysmon_files(y, d)
        if files:
            res.append({"yml": y, "techniques": techs, "files": files, "directory": Path(y).parent.name})
    return res


def cmd_list(a):
    root = Path(a.root)
    for ds in select_datasets(root, a.techniques):
        for f in ds["files"]:
            print(str(f.relative_to(root)).replace("\\", "/"))


def build_groups(events):
    groups = collections.defaultdict(list)
    parent = {}
    for e in events:
        k = group_key(e)
        if not k:
            continue
        groups[k].append(e)
        if e["eid"] == 1:
            pg = e["data"].get("ParentProcessGuid")
            if pg:
                parent[k] = pg
    return groups, parent


def descendants(roots, parent):
    children = collections.defaultdict(set)
    for c, p in parent.items():
        children[p].add(c)
    out, stack = set(), list(roots)
    while stack:
        n = stack.pop()
        for c in children.get(n, ()):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def to_record(sample_id, techniques, evs, label_source, dataset):
    evs = sorted(evs, key=lambda e: (e["time"] or datetime.min, e["eid"]))[:MAX_EVENTS]
    t0 = next((e["time"] for e in evs if e["time"]), None)
    events = []
    for i, e in enumerate(evs):
        d = e["data"]
        fields = {k: v for k, v in d.items() if k in KEEP_FIELDS and v and v != "-"}
        hints = RULE_TECH_RE.findall(d.get("RuleName", ""))
        dt = (e["time"] - t0).total_seconds() if e["time"] and t0 else 0.0
        events.append({"idx": i, "t": round(dt, 3), "eid": e["eid"], "user": e["user"],
                       "title": TITLES.get(e["eid"], f"Event {e['eid']}"), "fields": fields, "hints": hints})
    rec = {
        "sample_id": sample_id, "env": "ATTACKDATA", "label": 1, "verdict": "malicious",
        "techniques": techniques, "techniques_base": sorted({t[:5] for t in techniques}),
        "tactics": [], "actions": [], "sysmon_hints": sorted({h for e in events for h in e["hints"]}),
        "events": events, "evidence_gt_events": [], "label_source": label_source, "dataset": dataset,
    }
    rec["text"] = render(events)
    rec["spans"] = [{"event": e["idx"], "field": k, "value": v[:200]} for e in events for k, v in e["fields"].items()]
    return rec


def cmd_convert(a):
    root = Path(a.root)
    root_re = re.compile(a.root_regex, re.I)
    dsets = select_datasets(root, a.techniques)
    parsed, stats = [], collections.Counter()
    image_files = collections.Counter()
    for ds in dsets:
        for f in ds["files"]:
            if not f.exists():
                stats["file_missing"] += 1
                continue
            if is_lfs_pointer(f):
                stats["file_lfs_pointer"] += 1
                continue
            evs = parse_events(f)
            if not evs:
                stats["file_no_sysmon_events"] += 1
                continue
            stats["file_parsed"] += 1
            groups, parent = build_groups(evs)
            sigs = {group_signature(g) for g in groups.values() if group_image(g)}
            for sg in sigs:
                image_files[sg] += 1
            parsed.append((ds, f, groups, parent, len(evs)))
    n_files = max(len(parsed), 1)
    background = {im for im, c in image_files.items() if c / n_files >= a.bg_frac and n_files >= a.bg_min_files}

    per_tech = collections.Counter()
    src_count = collections.Counter()
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_files = []
    with open(out_path, "w", encoding="utf-8") as fo:
        for ds, f, groups, parent, n_ev in parsed:
            roots = [k for k, g in groups.items()
                     if any(e["eid"] == 1 and root_re.search(e["data"].get("CommandLine", "")) for e in g)]
            if roots and a.mode in ("auto", "tree"):
                keys = descendants(roots, parent) - set(roots)
                source = "tree"
            elif a.mode in ("auto", "rare"):
                keys = {k for k, g in groups.items() if group_signature(g) not in background}
                source = "rare"
            else:
                keys, source = set(), "none"
            keys = {k for k in keys if not AGENT_RE.search(group_image(groups[k]))}
            ordered = sorted(keys, key=lambda k: (min((e["time"] or datetime.min) for e in groups[k]), k))
            techs = ds["techniques"]
            base = sorted({t[:5] for t in techs})
            kept = 0
            for k in ordered:
                if kept >= a.max_per_file or all(per_tech[t] >= a.max_per_technique for t in base):
                    break
                evs = groups[k]
                if len(evs) < a.min_events:
                    continue
                sid = f"ATD_{techs[0]}_{zlib.crc32(str(f).encode()):08x}_{zlib.crc32(k.encode()):08x}"
                rec = to_record(sid, techs, evs, source, str(f.relative_to(root)).replace("\\", "/"))
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
                for t in base:
                    per_tech[t] += 1
                src_count[source] += 1
            manifest_files.append({"file": str(f.relative_to(root)).replace("\\", "/"), "techniques": techs,
                                   "events": n_ev, "groups": len(groups), "roots": len(roots),
                                   "label_source": source, "alerts": kept})
    manifest = {
        "root": str(root), "mode": a.mode, "root_regex": a.root_regex, "bg_frac": a.bg_frac,
        "datasets_selected": len(dsets), "file_stats": dict(stats), "alerts_per_technique": dict(sorted(per_tech.items())),
        "alerts_by_label_source": dict(src_count),
        "background_signatures": [list(x) for x in sorted(background)],
        "files": manifest_files,
    }
    Path(str(out_path) + ".manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("datasets_selected", "file_stats", "alerts_per_technique",
                                               "alerts_by_label_source")}, indent=1))
    if stats["file_lfs_pointer"]:
        print(f"WARNING {stats['file_lfs_pointer']} files are LFS pointers; run: python attackdata_to_eve.py list "
              f"--root {a.root} > lfs_files.txt, then git lfs pull --include=<each path>", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("list")
    l.add_argument("--root", required=True)
    l.add_argument("--techniques", nargs="+", default=DEFAULT_TECHNIQUES)
    l.set_defaults(func=cmd_list)
    c = sub.add_parser("convert")
    c.add_argument("--root", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--techniques", nargs="+", default=DEFAULT_TECHNIQUES)
    c.add_argument("--mode", choices=("auto", "tree", "rare"), default="auto")
    c.add_argument("--root_regex", default=ROOT_RE_DEFAULT)
    c.add_argument("--bg_frac", type=float, default=0.3)
    c.add_argument("--bg_min_files", type=int, default=5)
    c.add_argument("--min_events", type=int, default=1)
    c.add_argument("--max_per_file", type=int, default=20)
    c.add_argument("--max_per_technique", type=int, default=60)
    c.set_defaults(func=cmd_convert)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
