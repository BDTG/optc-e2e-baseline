"""STT44 buoc 5: do evidence EVE co trung GT node (DARPA) khong.
Map: evidence {event_idx} -> converted events[event_idx] -> src_node/dst_node (index_id) -> GT set.
Dem TRUNG khi src HOAC dst la GT node (theo lua chon)."""
import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="output eve/kb_only tren optc")
    ap.add_argument("--data", required=True, help="optc_*.jsonl (co src_node/dst_node)")
    ap.add_argument("--gt_map", default="../P0/gt/uuid_index_map.csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    gt = set(r["index_id"] for r in csv.DictReader(open(a.gt_map, encoding="utf-8")))
    data = {o["sample_id"]: o for o in read_jsonl(a.data)}
    # GT node hien dien trong tap alert nay (mau so recall)
    present = set()
    for o in data.values():
        for e in o["events"]:
            for n in (e["src_node"], e["dst_node"]):
                if n in gt:
                    present.add(n)

    res = list(read_jsonl(a.results))
    ev_total = ev_hit = 0
    alerts_with_ev = alerts_hit = 0
    surfaced = set()
    for r in res:
        evs = r.get("evidence") or []
        o = data.get(r["sample_id"])
        if not o:
            continue
        alerts_with_ev += int(bool(evs))
        hit_this = False
        for it in evs:
            ev_total += 1
            ei = it.get("event")
            if ei is None or ei >= len(o["events"]):
                continue
            e = o["events"][ei]
            nodes = {e["src_node"], e["dst_node"]}
            g = nodes & gt
            if g:
                ev_hit += 1
                hit_this = True
                surfaced |= g
        alerts_hit += int(hit_this)

    def pct(x, y):
        return round(x / y, 4) if y else None

    out = {
        "results": a.results, "data": a.data,
        "n_alerts": len(res), "gt_total": len(gt), "gt_present_in_set": len(present),
        "alerts_with_evidence": alerts_with_ev,
        "evidence_items_total": ev_total, "evidence_items_hit_gt": ev_hit,
        "evidence_precision": pct(ev_hit, ev_total),
        "alerts_with_ev_hitting_gt": alerts_hit,
        "alert_evidence_precision": pct(alerts_hit, alerts_with_ev),
        "gt_nodes_surfaced": len(surfaced),
        "node_recall_vs_present": pct(len(surfaced), len(present)),
        "node_recall_vs_all_gt": pct(len(surfaced), len(gt)),
        "hit_rule": "src OR dst is GT",
    }
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
