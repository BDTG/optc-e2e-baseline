# -*- coding: utf-8 -*-
"""Calib MAGIC subset: precision@1hop/2hop o cac threshold van hanh (FPR co dinh)."""
import torch
import numpy as np
import psycopg2

SCORES = '/data/artifacts_sub/evaluation/evaluation/ef05e2ae0d9907ac9c05e6742bdd434de16e7a3252cc25438d6e42462c2c60a9/optc_h051/precision_recall_dir/scores_model_epoch_11.pkl'

d = torch.load(SCORES, map_location='cpu', weights_only=False)
scores = np.array(d['pred_scores'], dtype=float)
y = np.array(d['y_truth'], dtype=int)
nodes = np.array(d['nodes'], dtype=int)

# GT index ids (113)
gt_idx = set(int(n) for n in d['node2attacks'].keys())
print(f"GT nodes: {len(gt_idx)}")

# 1-hop/2-hop tu event_table subset
conn = psycopg2.connect(database='optc_051_sub', host='pg-pids', user='postgres',
                        password='postgres', port=5432)
cur = conn.cursor()
cur.execute("SELECT DISTINCT src_index_id, dst_index_id FROM event_table")
edges = [(int(s), int(d_)) for s, d_ in cur.fetchall()]
conn.close()
print(f"edges: {len(edges)}")

one_hop = set()
for s, d_ in edges:
    if s in gt_idx:
        one_hop.add(d_)
    if d_ in gt_idx:
        one_hop.add(s)
two_hop = set(one_hop)
for s, d_ in edges:
    if s in one_hop:
        two_hop.add(d_)
    if d_ in one_hop:
        two_hop.add(s)
two_hop -= gt_idx
print(f"1-hop: {len(one_hop)} | 2-hop: {len(two_hop)}")

# Threshold theo FPR target (tren test)
from sklearn.metrics import roc_curve
fpr_grid = [0.0001, 0.001, 0.01, 0.05, 0.1]
fpr, tpr, th = roc_curve(y, scores)
print(f"\n{'FPR':>8} {'th':>10} {'tp':>5} {'fp':>8} {'rec':>7} {'prec':>9} {'prec@1hop':>10} {'prec@2hop':>10}")
for ft in fpr_grid:
    idx = np.argmin(np.abs(fpr - ft))
    t = th[idx]
    pred = (scores >= t).astype(int)
    tp = ((pred==1)&(y==1)).sum(); fp = ((pred==1)&(y==0)).sum()
    rec = tp/len(gt_idx)
    prec = tp/(tp+fp) if tp+fp else 0
    alerts = set(int(n) for n in nodes[pred==1])
    p1 = len(alerts & one_hop)/len(alerts) if alerts else 0
    p2 = len(alerts & two_hop)/len(alerts) if alerts else 0
    print(f"{ft:>8.4f} {t:>10.3f} {tp:>5} {fp:>8} {rec:>7.4f} {prec:>9.6f} {p1:>10.5f} {p2:>10.5f}")

# Threshold hien tai (train_distance tu val)
cur_th = scores[np.array(d['y_preds'], dtype=int)==1].min() if np.array(d['y_preds']).sum() else 0
pred = (scores >= cur_th).astype(int)
alerts = set(int(n) for n in nodes[pred==1])
tp = ((pred==1)&(y==1)).sum(); fp = ((pred==1)&(y==0)).sum()
print(f"\nTHRESHOLD HIEN TAI (val train_distance): {cur_th:.3f} -> tp={tp} fp={fp} alerts={len(alerts)}")
print(f"  precision@0hop={tp/len(alerts):.6f} prec@1hop={len(alerts&one_hop)/len(alerts):.5f} prec@2hop={len(alerts&two_hop)/len(alerts):.5f}")
