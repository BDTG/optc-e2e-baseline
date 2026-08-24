"""measure_cpu.py — Do latency + RAM tren CPU (endpoint KHONG GPU).

Day la con so THAT cho RQ2. train.py do latency tren GPU (luc train);
paper can latency CPU vi muc tieu la thiet bi phan cung han hep khong GPU.

Chay tren may CHI DUNG CPU (hoac ep CUDA_VISIBLE_DEVICES=""):
  CUDA_VISIBLE_DEVICES="" python measure_cpu.py --adapter ./slm-ckpt \
      --base Qwen/Qwen2.5-3B --data train_data.jsonl --n 200 --quant int4
"""
from __future__ import annotations

import argparse
import time
import statistics

import psutil
import torch

from utils import make_run_dir, read_jsonl, save_run_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="model goc (vd Qwen/Qwen2.5-3B)")
    ap.add_argument("--adapter", default=None, help="thu muc LoRA adapter (neu co)")
    ap.add_argument("--data", required=True, help="JSONL de lay mau do latency")
    ap.add_argument("--n", type=int, default=200, help="so mau do")
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=None,
                    help="gioi han so CPU thread (mo phong endpoint yeu)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"[cpu] torch threads = {torch.get_num_threads()}")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    proc = psutil.Process()
    mem_before = proc.memory_info().rss / 1e9

    print(f"[cpu] tai model tren CPU (fp32/bf16, khong quant vi bitsandbytes can GPU)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=2, trust_remote_code=True, torch_dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    mem_after = proc.memory_info().rss / 1e9
    print(f"[cpu] RAM model ~ {mem_after - mem_before:.2f} GB (tong RSS {mem_after:.2f} GB)")

    rows = read_jsonl(args.data)[:args.n]

    # warmup
    with torch.no_grad():
        enc = tok(rows[0]["text"], truncation=True, max_length=args.max_seq_length,
                  return_tensors="pt")
        model(**enc)

    lat = []
    with torch.no_grad():
        for r in rows:
            enc = tok(r["text"], truncation=True, max_length=args.max_seq_length,
                      return_tensors="pt")
            t0 = time.perf_counter()
            model(**enc)
            lat.append((time.perf_counter() - t0) * 1000)

    peak_mem = proc.memory_info().rss / 1e9
    p50, p90, mean = statistics.median(lat), sorted(lat)[int(0.9 * len(lat))], statistics.mean(lat)
    print("\n===== LATENCY CPU (endpoint khong GPU) =====")
    print(f"  model        : {args.base}")
    print(f"  so mau        : {len(lat)}")
    print(f"  latency p50  : {p50:.1f} ms")
    print(f"  latency p90  : {p90:.1f} ms")
    print(f"  latency mean : {mean:.1f} ms")
    print(f"  RAM peak     : {peak_mem:.2f} GB")
    print(f"  throughput   : {1000/mean:.1f} mau/giay")
    print("\n[note] day la con so RQ2 that. Ghi vao bang Pareto cung voi AUC/recall tu train.py.")

    run_dir = make_run_dir("measure_cpu", model=args.base.split("/")[-1],
                            adapter=(args.adapter or "none").rstrip("/").split("/")[-1],
                            quant=None, threads=args.threads)
    save_run_metrics(run_dir, base=args.base, adapter=args.adapter, n=len(lat),
                      max_seq_length=args.max_seq_length, threads=args.threads,
                      latency_p50_ms=round(p50, 2), latency_p90_ms=round(p90, 2),
                      latency_mean_ms=round(mean, 2), ram_peak_gb=round(peak_mem, 3),
                      throughput_per_s=round(1000 / mean, 2))
    print(f"[run] metrics.json -> {run_dir}")


if __name__ == "__main__":
    main()
