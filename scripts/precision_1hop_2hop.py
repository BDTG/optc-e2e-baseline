# -*- coding: utf-8 -*-
"""GT UUIDs (113, tu DB full map) -> tim trong pilot DB optc_051 qua node_uuid
-> so sanh voi alerts MAGIC v3 (index pilot). Tinh precision@0hop/1hop."""
import torch
import csv

import psycopg2

SCORES = '/data/artifacts/evaluation/evaluation/3b2acf17e906e6eebace6e6761cbcfbb9340dcca78843ee99bfe5e8019f6bc3f/optc_h051/precision_recall_dir/scores_model_epoch_11.pkl'
MAP = '/tmp/uuid_index_map.csv'

d = torch.load(SCORES, map_location='cpu', weights_only=False)
nodes = d['nodes']
y_truth = d['y_truth']
y_pred = d['y_preds']

tp = set(n for n, yt, yp in zip(nodes, y_truth, y_pred) if yt == 1 and yp == 1)
fp = set(n for n, yt, yp in zip(nodes, y_truth, y_pred) if yt == 0 and yp == 1)
fn = set(n for n, yt, yp in zip(nodes, y_truth, y_pred) if yt == 1 and yp == 0)

# GT UUIDs
gt_uuids = set()
with open(MAP, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        gt_uuids.add(r['gt_uuid'])
print(f"GT UUIDs: {len(gt_uuids)}")

# Tim GT UUIDs trong pilot DB (node_uuid = gt_uuid? hoac qua uuid_index_map table)
conn = psycopg2.connect(database='optc_051', host='pg-pids', user='postgres',
                        password='postgres', port=5432)
cur = conn.cursor()

# node_uuid trong pilot DB
found = {}
for table in ['subject_node_table', 'file_node_table', 'netflow_node_table']:
    cur.execute(f"SELECT node_uuid, index_id FROM {table}")
    for u, i in cur.fetchall():
        if u in gt_uuids:
            found[u] = (table, i)

print(f"GT UUIDs co mat trong pilot DB: {len(found)}/{len(gt_uuids)}")

gt_pilot_idx = set(i for _, (_, i) in found.items())
print(f"  -> index pilot: {sorted(gt_pilot_idx)[:20]}")

# So sanh voi alerts
tp_in_gt = tp & gt_pilot_idx
fp_in_gt = fp & gt_pilot_idx
fn_in_gt = fn & gt_pilot_idx

total_alerts = len(tp) + len(fp)
prec = len(tp_in_gt) / total_alerts if total_alerts else 0
rec = len(tp_in_gt) / len(gt_pilot_idx) if gt_pilot_idx else 0
print(f"\n=== precision@0hop (GT nodes co trong graph) ===")
print(f"TP la GT: {len(tp_in_gt)} | alerts: {total_alerts}")
print(f"precision: {prec:.6f}")
print(f"recall: {rec:.4f} ({len(tp_in_gt)}/{len(gt_pilot_idx)})")
print(f"FP la GT (bi flag nhung khong phai GT time-window): {len(fp_in_gt)}")
print(f"FN la GT (GT nodes bi miss): {len(fn_in_gt)} -> {sorted(fn_in_gt)}")

# 1-hop: GT nodes + neighbors trong event_table (edges)
# load edges src_index_id, dst_index_id
cur.execute("SELECT DISTINCT src_index_id, dst_index_id FROM event_table")
edges = cur.fetchall()
print(f"\nedges: {len(edges)}")

# 1-hop cua GT nodes
one_hop = set()
for s, d_ in edges:
    s, d_ = int(s), int(d_)
    if s in gt_pilot_idx:
        one_hop.add(d_)
    if d_ in gt_pilot_idx:
        one_hop.add(s)
print(f"1-hop neighbors: {len(one_hop)} (khong tinh GT nodes)")
alerts = tp | fp
prec_1hop = len(alerts & one_hop) / len(alerts) if alerts else 0
print(f"alerts trong 1-hop: {len(alerts & one_hop)}/{len(alerts)} -> precision@1hop: {prec_1hop:.6f}")

# 2-hop
two_hop = set(one_hop)
for s, d_ in edges:
    s, d_ = int(s), int(d_)
    if s in one_hop:
        two_hop.add(d_)
    if d_ in one_hop:
        two_hop.add(s)
two_hop -= gt_pilot_idx
print(f"2-hop (them): {len(two_hop - one_hop)}")
prec_2hop = len(alerts & two_hop) / len(alerts) if alerts else 0
print(f"alerts trong 2-hop: {len(alerts & two_hop)}/{len(alerts)} -> precision@2hop: {prec_2hop:.6f}")

conn.close()
