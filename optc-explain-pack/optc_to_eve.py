"""STT44: chuyen alert OpTC (alerts-enriched-v2.jsonl, do thi node/op sau cascade Velox)
sang schema adgen-v2 de chay eve.py. Moi canh event_seq (src,op,dst) -> 1 event kieu Sysmon,
GIU src_node/dst_node (bo prefix node_) de map nguoc ve GT node (uuid_index_map.index_id).
Path OpTC "/Device/HarddiskVolumeN/..." duoc chuan hoa ve "C:\\..." de clause KB (Sysmon) kich duoc.
Khong co nhan technique cho OpTC -> techniques_base rong; danh gia chi o muc evidence-vs-GT-node."""
import argparse
import json
import re
from pathlib import Path

NODE = re.compile(r"^node_", re.I)
NETFLOW = re.compile(r"netflow\s+(\S+)\s*->\s*(\S+)\s*\(([^)]*)\)")
DEV = re.compile(r"(?i)^\\device\\harddiskvolume\d+\\")
SYSROOT = re.compile(r"(?i)^%systemroot%\\")
PATHF = {"Image", "ParentImage", "SourceImage", "TargetImage", "TargetFilename", "ImageLoaded"}


def strip_node(x):
    return NODE.sub("", str(x or ""))


def norm_path(v):
    v = v.replace("/", "\\")
    v = DEV.sub("c:\\\\", v)
    v = SYSROOT.sub("c:\\\\windows\\\\", v)
    return v


def parse_msg(msg):
    m = (msg or "").strip()
    if m.startswith("subject"):
        body = m[len("subject"):].strip()
        cmd = None
        if " | cmd:" in body:
            path, cmd = body.split(" | cmd:", 1)
            path, cmd = path.strip(), cmd.strip()
        else:
            path = body
        return "subject", {"path": path, "cmd": (None if cmd in (None, "None", "") else cmd)}, path
    if m.startswith("file"):
        return "file", {"path": m[len("file"):].strip()}, m[len("file"):].strip()
    if m.startswith("netflow"):
        g = NETFLOW.search(m)
        if g:
            return "netflow", {"src": g.group(1), "dst": g.group(2), "proto": g.group(3)}, g.group(2)
        return "netflow", {"raw": m}, m
    return "other", {"raw": m}, m


def ip_port(x):
    if not x or x in ("None:None", "None"):
        return None, None
    i = x.rfind(":")
    if i < 0:
        return x, None
    return x[:i], x[i + 1:]


def build_event(idx, e):
    op = (e.get("op") or "").upper()
    st, sf, _ = parse_msg(e.get("src_msg"))
    dt, df, _ = parse_msg(e.get("dst_msg"))
    src_node, dst_node = strip_node(e.get("src")), strip_node(e.get("dst"))
    F, eid, title = {}, 0, op.title()

    if st == "subject" and dt == "subject" and op == "CREATE":
        eid, title = 1, "Process Create"
        F["ParentImage"] = sf["path"]; F["Image"] = df["path"]
        if df.get("cmd"):
            F["CommandLine"] = df["cmd"]
        if sf.get("cmd"):
            F["ParentCommandLine"] = sf["cmd"]
    elif st == "subject" and dt == "subject" and op in ("OPEN", "READ"):
        eid, title = 10, "Process Access"
        F["SourceImage"] = sf["path"]; F["TargetImage"] = df["path"]
    elif "netflow" in (st, dt) and op in ("START", "MESSAGE", "OPEN"):
        eid, title = 3, "Network Connection"
        subj = sf if st == "subject" else (df if dt == "subject" else None)
        net = df if dt == "netflow" else sf
        if subj:
            F["Image"] = subj["path"]
        ip, port = ip_port(net.get("dst"))
        if ip:
            F["DestinationIp"] = ip
        if port:
            F["DestinationPort"] = port
    elif dt == "file" and op in ("WRITE", "CREATE", "MODIFY", "RENAME", "READ", "DELETE"):
        eid, title = 11, "File " + op.title()
        if st == "subject":
            F["Image"] = sf["path"]
        F["TargetFilename"] = df["path"]
    elif op == "TERMINATE":
        eid, title = 5, "Process Terminate"
        F["Image"] = (sf.get("path") if st == "subject" else df.get("path")) or ""
    else:
        if st == "subject":
            F["Image"] = sf["path"]
        if dt == "subject":
            F["TargetImage"] = df["path"]
        if dt == "file":
            F["TargetFilename"] = df["path"]

    for k in list(F):
        if k in PATHF and isinstance(F[k], str):
            F[k] = norm_path(F[k])
    F = {k: v for k, v in F.items() if v not in (None, "", "None")}
    return {"idx": idx, "eid": eid, "title": title, "fields": F, "hints": [],
            "op": op, "src_node": src_node, "dst_node": dst_node}


def convert(alert):
    seq = alert.get("event_seq") or []
    events = [build_event(i, e) for i, e in enumerate(seq)]
    events = [e for e in events if e["fields"]]
    for i, e in enumerate(events):
        e["idx"] = i
    if not events:
        return None
    text_lines = [f"[E{e['idx']}] {e['title']} | " +
                  " | ".join(f"{k}: {str(v)[:200]}" for k, v in e["fields"].items()) for e in events]
    return {
        "sample_id": f"OPTC_{strip_node(alert.get('nid'))}",
        "env": "OPTC", "label": 1,
        "verdict": None, "risk_level": None,
        "techniques": [], "techniques_base": [], "tactics": [], "sysmon_hints": [],
        "events": events,
        "text": "\n".join(text_lines)[:6000],
        "spans": [{"event": e["idx"], "field": k, "value": str(v)[:200]}
                  for e in events for k, v in e["fields"].items()],
        "evidence_gt_events": [],
        "nid": strip_node(alert.get("nid")), "rank": alert.get("rank"), "score": alert.get("score"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda o: o.get("rank", 1 << 30))
    if a.top:
        rows = rows[:a.top]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n_out = n_empty = n_ev = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for al in rows:
            o = convert(al)
            if o is None:
                n_empty += 1
                continue
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            n_out += 1
            n_ev += len(o["events"])
    print(json.dumps({"in": len(rows), "out": n_out, "empty": n_empty,
                      "events_total": n_ev, "events_per_alert": round(n_ev / max(n_out, 1), 1),
                      "out_file": a.out}, indent=1))


if __name__ == "__main__":
    main()
