"""bench_cpu_mem.py — RAM footprint CPU tach rieng: weight + KV cache (yeu cau Viec 3).
Moi model chay trong PROCESS RIENG (tranh allocator drift cua torch).
Usage: python bench_cpu_mem.py            # ca 3 size
       python bench_cpu_mem.py --one M    # do 1 model (subprocess)
"""
import argparse, json, os, subprocess, sys, time

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
          "Qwen/Qwen2.5-1.5B-Instruct"]


def measure_one(model_name):
    import psutil, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import importlib.util
    T = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("pe", os.path.join(T, "product_explain.py"))
    pe = importlib.util.module_from_spec(spec)
    sys.modules["pe"] = pe
    spec.loader.exec_module(pe)

    proc = psutil.Process()
    rss0 = proc.memory_info().rss / 1e9
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    rss_load = proc.memory_info().rss / 1e9

    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    kv_heads = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 1))
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    # KV cache (fp32): 2 (K+V) * layers * kv_heads * head_dim * ctx * 4 bytes
    kv_mb = lambda ctx: 2 * cfg.num_hidden_layers * kv_heads * hd * ctx * 4 / 1e6

    txt = "Event10 | C:\\Windows\\System32\\svchost.exe | CreateKey | Event10 | C:\\Windows\\System32\\svchost.exe"
    peak = rss_load
    t0 = time.time()
    for _ in range(3):
        try:
            pe.explain(txt, model, tok)
        except Exception:
            pass
        peak = max(peak, proc.memory_info().rss / 1e9)
    dt = (time.time() - t0) / 3
    return {"model": model_name, "params_b": round(n_params / 1e9, 3),
            "rss_base_gb": round(rss0, 2), "rss_after_load_gb": round(rss_load, 2),
            "weight_mem_gb": round(rss_load - rss0, 2),
            "rss_peak_infer_gb": round(peak, 2),
            "kv_mb_ctx512": round(kv_mb(512), 1), "kv_mb_ctx2048": round(kv_mb(2048), 1),
            "latency_s_per_sample": round(dt, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", default=None)
    a = ap.parse_args()
    if a.one:
        print("JSON:" + json.dumps(measure_one(a.one)), flush=True)
        return
    out = []
    for m in MODELS:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", m],
                           capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("JSON:")]
        if line:
            rec = json.loads(line[0][5:])
            out.append(rec)
            print(m.split("/")[-1], "->", rec["weight_mem_gb"], "GB weight, peak",
                  rec["rss_peak_infer_gb"], "GB", flush=True)
        else:
            print(m, "FAIL", r.stdout[-200:], r.stderr[-200:], flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpu-mem.json"), "w"), indent=1)
    print("SAVED cpu-mem.json", flush=True)


main()
