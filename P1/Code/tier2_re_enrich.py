"""
Re-enrich alerts with real msg from nid2msg_cache.pkl
Input: alerts_enriched_partial.jsonl (2250, generic labels) + nid2msg_cache.pkl
Output: alerts_enriched_v2.jsonl with self_label/msg enriched
"""
import json, pickle, os, re
from pathlib import Path

CACHE_PATH = r"D:\OpTC-thesis\data\nid2msg_cache.pkl"
INPUT_PATH = r"D:\OpTC-thesis\P1\Output\alerts_enriched_partial.jsonl"
OUTPUT_PATH = r"D:\OpTC-thesis\P1\Output\alerts_enriched_v2.jsonl"
GT_SCORES_PATH = r"D:\OpTC-thesis\P1\Output\gt_and_scores.json"

def load_cache(path):
    print(f"Loading cache {path} ...")
    with open(path, 'rb') as f:
        cache = pickle.load(f)
    print(f"  cache size: {len(cache)} entries")
    # sample
    for k in list(cache.keys())[:3]:
        print(f"  sample {k}: {str(cache[k])[:200]}")
    return cache

def nid_to_int(nid_str):
    # nid like "2561" or "node_2561"
    if isinstance(nid_str, int):
        return nid_str
    s = str(nid_str)
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else None

def get_msg(nid, cache):
    # Try multiple key forms
    nid_int = nid_to_int(nid)
    # direct int
    if nid_int is not None and nid_int in cache:
        return str(cache[nid_int])
    # str form
    if str(nid) in cache:
        return str(cache[str(nid)])
    # node_ prefix
    key = f"node_{nid_int}" if nid_int is not None else None
    if key and key in cache:
        return str(cache[key])
    return None

def enrich_alert(alert, cache):
    nid = alert.get("nid")
    nid_int = nid_to_int(nid)
    # self_label enrichment
    msg = get_msg(nid, cache)
    if msg:
        alert["self_label"] = msg
        alert["self_label_src"] = "cache"
    else:
        # keep original but mark missing
        alert["self_label_src"] = "miss"
        # debug: keep original
    # parent_chain enrichment
    new_chain = []
    for p in alert.get("parent_chain", []):
        node_raw = p.get("node", "")
        # node like "node_2561" -> extract id
        op = p.get("op", "")
        node_msg = get_msg(node_raw, cache)
        entry = {"node": node_raw, "op": op}
        if node_msg:
            entry["msg"] = node_msg
        else:
            entry["msg"] = None
        new_chain.append(entry)
    alert["parent_chain"] = new_chain
    # event_seq enrichment
    new_seq = []
    for e in alert.get("event_seq", []):
        src = e.get("src","")
        dst = e.get("dst","")
        op = e.get("op","")
        src_msg = get_msg(src, cache)
        dst_msg = get_msg(dst, cache)
        entry = {"src": src, "op": op, "dst": dst}
        if src_msg:
            entry["src_msg"] = src_msg
        if dst_msg:
            entry["dst_msg"] = dst_msg
        new_seq.append(entry)
    alert["event_seq"] = new_seq
    return alert

def main():
    cache = load_cache(CACHE_PATH)
    # stats for missing
    alerts = []
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            alerts.append(json.loads(line))
    print(f"Loaded {len(alerts)} alerts from {INPUT_PATH}")
    # also load gt to check coverage
    with open(GT_SCORES_PATH) as f:
        gt_data = json.load(f)
    gt_nids = set(gt_data["gt_nids"])
    print(f"GT total {len(gt_nids)}")
    # enrich
    enriched = []
    miss_self = 0
    miss_parent = 0
    miss_event = 0
    total_parent = 0
    total_event = 0
    for a in alerts:
        enriched_a = enrich_alert(a, cache)
        enriched.append(enriched_a)
        if enriched_a.get("self_label_src") == "miss":
            miss_self += 1
        for p in enriched_a.get("parent_chain",[]):
            total_parent += 1
            if p.get("msg") is None:
                miss_parent += 1
        for e in enriched_a.get("event_seq",[]):
            total_event += 1
            # count if both src_msg and dst_msg missing? just count dst
            if e.get("src_msg") is None and e.get("dst_msg") is None:
                miss_event += 1
    print(f"Enriched {len(enriched)}")
    print(f"  miss self_label: {miss_self}/{len(enriched)} ({miss_self/len(enriched)*100:.1f}%)")
    print(f"  miss parent msg: {miss_parent}/{total_parent} ({miss_parent/max(total_parent,1)*100:.1f}%)")
    print(f"  miss event both: {miss_event}/{total_event} ({miss_event/max(total_event,1)*100:.1f}%)")
    # show samples: 2 GT and 2 benign
    print("\n--- Sample enriched GT alerts ---")
    for a in enriched:
        if str(a.get("nid")) in gt_nids:
            print(json.dumps(a, ensure_ascii=False)[:1500])
            # just 2
            break
    print("\n--- Sample enriched benign alerts ---")
    for a in enriched:
        if str(a.get("nid")) not in gt_nids:
            print(json.dumps(a, ensure_ascii=False)[:1500])
            break
    # write output
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out:
        for a in enriched:
            out.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"\nWrote {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
