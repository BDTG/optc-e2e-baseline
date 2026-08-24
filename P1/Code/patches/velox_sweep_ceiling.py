import os, sys, torch, argparse, glob
import numpy as np
sys.path.insert(0, "/home/vung2/optec-l6/velox/src")
os.chdir("/home/vung2/optec-l6/velox/src")
args = argparse.Namespace(model="velox", dataset="optc_h051", cpu=False, from_weights=False, seed=0,
                          force_restart=[], restart_from_scratch=False, tuned=True,
                          tuning_mode=False, tuning_file_path=None, experiment="no_experiment")
from config import get_yml_cfg
cfg = get_yml_cfg(args)
from provnet_utils import init_database_connection
from FP_check import get_node_infos

VD = "/home/vung2/optec-l6/velox/artifacts/detection/evaluation/9e5abdde8f6e786d10de031c927ac59c88249d11ce4fc1b54f69b4af334fc999/optc_h051/precision_recall_dir"

res_files = sorted(glob.glob(os.path.join(VD, "result_model_epoch_*.pth")))
print("result files:", [os.path.basename(f) for f in res_files])

cur, connect = init_database_connection(cfg)
indexid2msg = get_node_infos(cur)
print(f"indexid2msg: {len(indexid2msg)} entries")

def flat_msg(nid):
    return indexid2msg.get(str(nid))

target = os.path.join(VD, "result_model_epoch_1.pth")
if not os.path.exists(target):
    target = res_files[-1]
results = torch.load(target, weights_only=False)
print(f"Loaded {os.path.basename(target)}: {len(results)} nodes")

nids, scores, y_trues = [], [], []
for nid, r in results.items():
    nids.append(nid); scores.append(float(r["score"])); y_trues.append(int(r["y_true"]))
n_pos = sum(y_trues)
print(f"Nodes: {len(nids)} | Malicious: {n_pos}")

scores_arr = np.array(scores); y_arr = np.array(y_trues)
order = np.argsort(-scores_arr)
sorted_y = y_arr[order]

print("=" * 78)
print("SWEEP NGUONG VELOX (test set)")
print("=" * 78)
print(f"{'k':>8} | {'TP@k':>5} | {'recall@k':>9} | {'prec@k':>8}")
for k in [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000]:
    if k > len(nids): break
    tp_k = int(sorted_y[:k].sum())
    print(f"{k:>8} | {tp_k:>5} | {tp_k/n_pos:>9.4f} | {tp_k/k:>8.4f}")

for K in [2000, 5000, 10000]:
    top_idx = order[:K]
    y_hat = np.zeros(len(nids), dtype=int); y_hat[top_idx] = 1
    tp_nids = [str(nids[i]) for i in range(len(nids)) if y_hat[i] == 1 and y_arr[i] == 1]
    fp_nids = [str(nids[i]) for i in range(len(nids)) if y_hat[i] == 1 and y_arr[i] == 0]
    all_true = [str(nids[i]) for i in range(len(nids)) if y_arr[i] == 1]
    tp_msgs = set(m for nid in tp_nids if (m := flat_msg(nid)) is not None)
    gt_msgs = set(m for nid in all_true if (m := flat_msg(nid)) is not None)
    ctp = cgt = uni = miss = 0
    for fp in fp_nids:
        m = flat_msg(fp)
        if m is None: miss += 1
        elif m in tp_msgs: ctp += 1
        elif m in gt_msgs: cgt += 1
        else: uni += 1
    ceil = len(tp_nids) / max(len(tp_nids) + ctp, 1)
    print("-" * 70)
    print(f"k={K}: TP={len(tp_nids)} FP={len(fp_nids)} | collide_TP={ctp} collide_GT={cgt} unique={uni} miss={miss}")
    print(f"  CEILING PRECISION = {len(tp_nids)}/{len(tp_nids)+ctp} = {100*ceil:.2f}%")

torch.save({"order": order, "scores": scores_arr, "y": y_arr, "nids": nids},
           os.path.join(VD, "threshold_sweep.pth"))
print("SAVED threshold_sweep.pth")
