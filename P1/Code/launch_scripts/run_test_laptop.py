"""
Laptop test-only runner: skips graph build + training, goes straight to
orthrus_gnn_testing.main + evaluation.main. CPU-only (no GPU on this laptop).
"""
import sys, os, argparse, time
os.environ["ORTHRUS_NO_DB"] = "1"
os.environ.setdefault("NID2MSG_CACHE", r"D:\orthrus_laptop\nid2msg_cache.pkl")

ROOT = r"D:\orthrus_laptop\optec-l6\orthrus"
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

args = argparse.Namespace(
    model="orthrus", dataset="OPTC_H051", cpu=True,
    run_from_training=True, seed=0, from_weights=False,
    force_restart=[], restart_from_scratch=False)
from config import get_yml_cfg
cfg = get_yml_cfg(args)

import torch, pickle
t0 = time.time()
print(f"torch {torch.__version__} cpu={args.cpu}", flush=True)
print("[cache] loading nodeid2msg...", flush=True)
with open(os.environ["NID2MSG_CACHE"], "rb") as f:
    nodeid2msg = pickle.load(f)
print(f"  {len(nodeid2msg)} labels in {time.time()-t0:.1f}s", flush=True)

from detection import orthrus_gnn_testing, evaluation

print("=== TESTING PHASE ===", flush=True)
orthrus_gnn_testing.main(cfg)
t1 = time.time()
print(f"=== testing done in {(t1-t0)/60:.1f} min ===", flush=True)

print("=== EVALUATION PHASE ===", flush=True)
evaluation.main(cfg)
t2 = time.time()
print(f"=== evaluation done in {(t2-t1)/60:.1f} min ===", flush=True)
print(f"TOTAL: {(t2-t0)/60:.1f} min", flush=True)
