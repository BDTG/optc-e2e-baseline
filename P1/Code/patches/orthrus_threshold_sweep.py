import os, sys, torch, argparse
sys.path.insert(0, "/home/vung2/optec-l6/orthrus/src")
os.chdir("/home/vung2/optec-l6/orthrus/src")
args = argparse.Namespace(model="orthrus", dataset="OPTC_H051", cpu=False, from_weights=False, seed=0)
from config import get_yml_cfg
cfg = get_yml_cfg(args)
from provnet_utils import init_database_connection, gen_nodeid2msg

D = "/home/vung2/optec-l6/orthrus/artifacts/detection/evaluation/73454a4b6b28d2b5f902861f332ccfbd941a929e996f25978a568be77bdff831/OPTC_H051/precision_recall_dir"
cur, connect = init_database_connection(cfg)
indexid2msg = gen_nodeid2msg(cur)

def flat_msg(nid):
    v = indexid2msg.get(nid)
    if v is None: v = indexid2msg.get(str(nid))
    if v is None and str(nid).isdigit(): v = indexid2msg.get(int(nid))
    if v is None: return None
    if isinstance(v, dict):
        return " | ".join(f"{k} {m}" for k, m in v.items())
    return str(v)

results = torch.load(os.path.join(D, "result_model_epoch_1.pth"), weights_only=False)

# Extract scores + labels
nids, scores, y_trues = [], [], []
for nid, r in results.items():
    nids.append(nid); scores.append(float(r["score"])); y_trues.append(int(r["y_true"]))
n_total = len(nids)
n_pos = sum(y_trues)
print(f"Nodes: {n_total} | Malicious (GT): {n_pos}")

import numpy as np
scores_arr = np.array(scores); y_arr = np.array(y_trues)

# SWEEP: thresholds = every unique score value (descending)
order = np.argsort(-scores_arr)
sorted_scores = scores_arr[order]
sorted_y = y_arr[order]

print("=" * 78)
print("SWEEP NGUONG ORTHRUS (test set, 322679 nodes, 114 malicious)")
print("recall@k = TP trong top-k score / 114  (k = alert budget)")
print("=" * 78)
print(f"{'k':>8} | {'TP@k':>5} | {'recall@k':>9} | {'prec@k':>8} | {'FP@k':>7}")
print("-" * 78)
sweep = {}
for k in [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3406, 5000, 10000, 50000]:
    tp_k = int(sorted_y[:k].sum())
    rec = tp_k / n_pos
    prec = tp_k / k
    sweep[k] = (tp_k, rec, prec)
    print(f"{k:>8} | {tp_k:>5} | {rec:>9.4f} | {prec:>8.4f} | {k - tp_k:>7}")

# Best F1 threshold + full confusion at a few operating points
print()
print("=" * 78)
print("CONFUSION TAI CAC DIEM NGUONG (score > thr => alert)")
print("=" * 78)
for thr_q in [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
    thr = np.quantile(scores_arr, 1 - thr_q)
    y_hat = (scores_arr > thr).astype(int)
    tp = int(((y_hat == 1) & (y_arr == 1)).sum())
    fp = int(((y_hat == 1) & (y_arr == 0)).sum())
    fn = n_pos - tp
    rec = tp / n_pos
    prec = tp / max(tp + fp, 1)
    print(f"alert rate {thr_q*100:>5.2f}% (~{int(thr_q*n_total):>6} alerts) | thr={thr:.4f} | TP={tp:>3} FP={fp:>6} FN={fn:>3} | recall={rec:.4f} prec={prec:.4f}")

# Save sweep results for precision ceiling at high-recall point
torch.save({"sweep": sweep, "sorted_scores": sorted_scores, "sorted_y": sorted_y},
           os.path.join(D, "threshold_sweep.pth"))
print()
print("SAVED threshold_sweep.pth")
