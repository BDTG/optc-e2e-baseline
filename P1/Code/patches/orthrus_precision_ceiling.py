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

nids, scores, y_trues = [], [], []
for nid, r in results.items():
    nids.append(nid); scores.append(float(r["score"])); y_trues.append(int(r["y_true"]))

import numpy as np
scores_arr = np.array(scores); y_arr = np.array(y_trues)

# Operating point: top-10k alerts (recall 88.6%)
K = 10000
order = np.argsort(-scores_arr)
top_idx = order[:K]
y_hat = np.zeros(len(nids), dtype=int)
y_hat[top_idx] = 1

tp_nids = [str(nids[i]) for i in range(len(nids)) if y_hat[i] == 1 and y_arr[i] == 1]
fp_nids = [str(nids[i]) for i in range(len(nids)) if y_hat[i] == 1 and y_arr[i] == 0]
all_true_nids = [str(nids[i]) for i in range(len(nids)) if y_arr[i] == 1]
print(f"Operating point k={K}: TP={len(tp_nids)} FP={len(fp_nids)} GT={len(all_true_nids)}")
print(f"recall={len(tp_nids)/len(all_true_nids):.4f} prec={len(tp_nids)/K:.4f}")

# Build msg sets
tp_msgs, gt_msgs = set(), set()
for nid in all_true_nids:
    m = flat_msg(nid)
    if m is not None:
        gt_msgs.add(m)
for nid in tp_nids:
    m = flat_msg(nid)
    if m is not None:
        tp_msgs.add(m)

fp_collide_tp, fp_collide_gt, fp_unique, missing = set(), set(), set(), 0
for fp in fp_nids:
    m = flat_msg(fp)
    if m is None:
        missing += 1
        continue
    if m in tp_msgs:
        fp_collide_tp.add(fp)
    elif m in gt_msgs:
        fp_collide_gt.add(fp)
    else:
        fp_unique.add(fp)

n_fp = len(fp_nids)
n_ctp, n_cgt, n_uni = len(fp_collide_tp), len(fp_collide_gt), len(fp_unique)
print("=" * 70)
print(f"TRAN PRECISION @ k={K} (SLM xem path+cmd, phan loai FP)")
print("=" * 70)
print(f"Tong FP:                          {n_fp}")
print(f"  FP trung msg voi TP:            {n_ctp}  ({100*n_ctp/max(n_fp,1):.2f}%)  <- KHONG tach duoc")
print(f"  FP trung msg voi GT (khong TP): {n_cgt}  ({100*n_cgt/max(n_fp,1):.2f}%)  <- kho, GT bi miss")
print(f"  FP msg doc nhat:                {n_uni}  ({100*n_uni/max(n_fp,1):.2f}%)  <- de tach nhat")
print(f"  FP khong co msg (bo qua):       {missing}")

tp_kept = len(tp_nids)
fp_floor = n_ctp
ceil_prec = tp_kept / max(tp_kept + fp_floor, 1)
print("-" * 70)
print(f"Neu SLM hoan hao (giu het {tp_kept} TP, loai moi FP tach duoc):")
print(f"  FP con lai toi thieu:  {fp_floor}")
print(f"  TRAN PRECISION:        {tp_kept}/{tp_kept + fp_floor} = {100*ceil_prec:.2f}%")
print("=" * 70)

# Also compute for k=2000 (prec 0.6%) and k=5000
for K2 in [2000, 5000]:
    top_idx2 = order[:K2]
    yh2 = np.zeros(len(nids), dtype=int); yh2[top_idx2] = 1
    tp2 = sum(1 for i in range(len(nids)) if yh2[i] == 1 and y_arr[i] == 1)
    fp2 = K2 - tp2
    # quick msg collision for this k
    tp2_nids = [str(nids[i]) for i in range(len(nids)) if yh2[i] == 1 and y_arr[i] == 1]
    fp2_nids = [str(nids[i]) for i in range(len(nids)) if yh2[i] == 1 and y_arr[i] == 0]
    tp2_msgs = set(m for nid in tp2_nids if (m := flat_msg(nid)) is not None)
    ctp2 = sum(1 for fp in fp2_nids if (m := flat_msg(fp)) is not None and m in tp2_msgs)
    ceil2 = tp2 / max(tp2 + ctp2, 1)
    print(f"k={K2}: TP={tp2} FP={fp2} | FP_collide_TP={ctp2} | CEILING={100*ceil2:.2f}%")

out = {
    "k": K,
    "tp_nids": tp_nids,
    "fp_collide_tp": sorted(fp_collide_tp),
    "fp_collide_gt_only": sorted(fp_collide_gt),
    "fp_unique": sorted(fp_unique),
    "ceiling_precision": ceil_prec,
}
torch.save(out, os.path.join(D, "precision_ceiling_k10000.pth"))
print("SAVED precision_ceiling_k10000.pth")
