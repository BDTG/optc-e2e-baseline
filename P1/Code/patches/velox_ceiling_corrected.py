"""Corrected Velox precision ceiling (explicit column names)."""
import sys, os
sys.path.insert(0, "/home/vung2/optec-l6/velox/src")
os.chdir("/home/vung2/optec-l6/velox/src")
import argparse
args = argparse.Namespace(model="velox", dataset="optc_h051", cpu=False, from_weights=False, seed=0,
                          force_restart=[], restart_from_scratch=False, tuned=True,
                          tuning_mode=False, tuning_file_path=None, experiment="no_experiment")
from config import get_yml_cfg
cfg = get_yml_cfg(args)
from provnet_utils import init_database_connection
import torch
import numpy as np

cur, connect = init_database_connection(cfg)
nid2msg = {}
cur.execute("SELECT index_id, path, cmd FROM subject_node_table;")
for iid, path, cmd in cur.fetchall():
    nid2msg[int(iid)] = f"subject {path} | cmd: {cmd}"
cur.execute("SELECT index_id, local_ip, local_port, remote_ip, remote_port, protocol FROM netflow_node_table;")
for iid, lip, lp, rip, rp, proto in cur.fetchall():
    nid2msg[int(iid)] = f"netflow {lip}:{lp} -> {rip}:{rp} ({proto})"
cur.execute("SELECT index_id, path FROM file_node_table;")
for iid, path in cur.fetchall():
    nid2msg[int(iid)] = f"file {path}"
print(f"label map: {len(nid2msg)} nodes")

VD = "/home/vung2/optec-l6/velox/artifacts/detection/evaluation/9e5abdde8f6e786d10de031c927ac59c88249d11ce4fc1b54f69b4af334fc999/optc_h051/precision_recall_dir"
results = torch.load(os.path.join(VD, "result_model_epoch_1.pth"), weights_only=False)
nids = list(results.keys()); scores = [float(results[n]["score"]) for n in nids]
y_arr = np.array([int(results[n]["y_true"]) for n in nids])
order = np.argsort(-np.array(scores))

def flat_msg(nid):
    return nid2msg.get(int(nid))

for K in [2000, 5000, 10000, 50000]:
    top = order[:K]
    y_hat = np.zeros(len(nids), dtype=int); y_hat[top] = 1
    tp_nids = [str(nids[i]) for i in range(len(nids)) if y_hat[i] == 1 and y_arr[i] == 1]
    fp_nids = [str(nids[i]) for i in range(len(nids)) if y_hat[i] == 1 and y_arr[i] == 0]
    all_true = [str(nids[i]) for i in range(len(nids)) if y_arr[i] == 1]
    tp_msgs = set(m for n in tp_nids if (m := flat_msg(n)) is not None)
    gt_msgs = set(m for n in all_true if (m := flat_msg(n)) is not None)
    ctp = cgt = uni = miss = 0
    for fp in fp_nids:
        m = flat_msg(fp)
        if m is None: miss += 1
        elif m in tp_msgs: ctp += 1
        elif m in gt_msgs: cgt += 1
        else: uni += 1
    ceil = len(tp_nids) / max(len(tp_nids) + ctp, 1)
    print("-" * 70)
    print(f"k={K}: TP={len(tp_nids)} FP={len(fp_nids)} | collide_TP={ctp} ({100*ctp/max(len(fp_nids),1):.1f}%) collide_GT={cgt} unique={uni}")
    print(f"  CORRECTED CEILING PRECISION = {len(tp_nids)}/{len(tp_nids)+ctp} = {100*ceil:.2f}%")
