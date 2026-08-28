"""SLM explain with full system metrics (CPU/RAM/GPU/VRAM/disk/IO).
Test multiple models in batch=8 setting.
"""
import json, time, os, psutil, pynvml, threading
import numpy as np, re, torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Config
# First entry uses LoRA checkpoint. Base models (without LoRA) for comparison.
MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "P1/Output/models/lora-05b/checkpoint-113", True),  # with LoRA
    ("Qwen/Qwen2.5-0.5B-Instruct", None, False),  # base only
    ("HuggingFaceTB/SmolLM2-360M-Instruct", None, False),
    ("Qwen/Qwen3-0.6B", None, False),
    ("Qwen/Qwen2.5-1.5B-Instruct", None, False),
]
SAMPLE_SIZE = 100  # subset of S1 for fast iteration
MAX_NEW = 120
BATCH = 8

DATA="P1/Output/data/alerts-enriched-v2.jsonl"
GT="P1/Output/data/gt_and_scores.json"

# --- metrics collector (background thread) ---
class SysMetrics:
    def __init__(self):
        self.running=False
        self.samples=[]
        pynvml.nvmlInit()
        self.handle=pynvml.nvmlDeviceGetHandleByIndex(0)
    def start(self):
        self.running=True
        self.t0=time.time()
        threading.Thread(target=self._collect, daemon=True).start()
    def _collect(self):
        proc=psutil.Process(os.getpid())
        while self.running:
            try:
                cpu=psutil.cpu_percent(interval=0.5)
                ram=psutil.virtual_memory().percent
                gpu=pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                mem=pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                vram_used=mem.used/(1024**3)
                vram_total=mem.total/(1024**3)
                temp=pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                proc_cpu=proc.cpu_percent()
                proc_rss=proc.memory_info().rss/(1024**3)
                self.samples.append({
                    "t":time.time()-self.t0, "cpu_pct":cpu, "ram_pct":ram,
                    "gpu_pct":gpu, "vram_used_gb":vram_used, "vram_total_gb":vram_total,
                    "gpu_temp_c":temp, "proc_cpu_pct":proc_cpu, "proc_rss_gb":proc_rss})
            except Exception as e:
                pass
    def stop(self):
        self.running=False
        if not self.samples: return {}
        s=self.samples
        def stat(key):
            vals=[x[key] for x in s if key in x]
            return {"mean":float(np.mean(vals)),"max":float(np.max(vals)),
                    "min":float(np.min(vals))}
        return {"duration_sec":s[-1]["t"],"n_samples":len(s),
                **{k:stat(k) for k in ["cpu_pct","ram_pct","gpu_pct","vram_used_gb",
                                       "gpu_temp_c","proc_cpu_pct","proc_rss_gb"]}}
    def __del__(self):
        try: pynvml.nvmlShutdown()
        except: pass

# --- load data + filter S1 ---
gt=set(json.load(open(GT))["gt_nids"])
alerts=[json.loads(l) for l in open(DATA, encoding='utf-8') if l.strip()]
def has_cmd(r):
    chain=r.get("parent_chain",[]) or []
    for c in chain:
        msg=c.get("msg","") or ""
        if "| cmd:" in msg:
            v=msg.split("| cmd:",1)[1].strip()
            if v and v.lower()!="none": return True
    return False
def has_chain(r): return len(r.get("parent_chain",[]) or [])>=2
def has_events(r): return len(r.get("event_seq",[]) or [])>=3
s1=[a for a in alerts if all([has_cmd(a), has_chain(a), has_events(a)])]
s1=s1[:SAMPLE_SIZE]  # subset for speed

def build_chain_text(a):
    chain=a.get("parent_chain",[]) or []
    seq=a.get("event_seq",[]) or []
    parts=[c.get("msg","") for c in chain[-5:]]
    parts+=[f"{e.get('src_msg','')} -> {e.get('dst_msg','')}" for e in seq[-5:]]
    return " | ".join([p for p in parts if p])[:800]

PROMPT = (
    "You are a security analyst. Given the provenance chain of a process, "
    "explain what this process is doing and classify it.\n"
    "Chain: {chain}\n"
    "Output: Verdict, Subject, Action, TTP, Explanation"
)

# System info snapshot
pynvml.nvmlInit()
sys_info={
    "platform": psutil.WINDOWS if hasattr(psutil,'WINDOWS') else "linux",
    "cpu_count_physical": psutil.cpu_count(logical=False),
    "cpu_count_logical": psutil.cpu_count(logical=True),
    "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
    "ram_total_gb": psutil.virtual_memory().total/(1024**3),
    "gpu_name": pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0)),
    "python": __import__('sys').version.split()[0],
    "torch": torch.__version__,
    "transformers": __import__('transformers').__version__,
}
print("SYSTEM:", json.dumps(sys_info, indent=2), flush=True)

results_all=[]
for model_name, ckpt, use_lora in MODELS:
    print(f"\n=== Loading {model_name} (lora={use_lora}) ===", flush=True)
    tok=AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16,
        device_map="auto")
    if use_lora and os.path.isdir(ckpt):
        try:
            model=PeftModel.from_pretrained(base, ckpt)
            print("  LoRA loaded", flush=True)
        except Exception as e:
            print(f"  LoRA fail: {e}, using base", flush=True)
            model=base
    else:
        model=base
    model.eval()

    prompts=[PROMPT.format(chain=build_chain_text(a)) for a in s1]
    labels=[1 if str(a["nid"]) in gt else 0 for a in s1]

    # Start metrics
    metrics=SysMetrics()
    metrics.start()
    t_start=time.time()
    latencies=[]
    n_correct=0
    n_ttp=0
    n_hallu=0
    n_loop=0
    gens_sample=[]

    for i in range(0, len(prompts), BATCH):
        batch_p=prompts[i:i+BATCH]
        batch_y=labels[i:i+BATCH]
        t0=time.time()
        try:
            enc=tok(batch_p, return_tensors="pt", truncation=True,
                    max_length=480, padding=True).to(model.device)
            out=model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                              pad_token_id=tok.eos_token_id)
            gens=[tok.decode(out[k][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
                  for k in range(len(batch_p))]
        except Exception as e:
            gens=[f"ERROR: {e}"]*len(batch_p)
        elapsed=time.time()-t0
        latencies.extend([elapsed/len(batch_p)]*len(batch_p))
        for j,g in enumerate(gens):
            v_match=re.search(r"Verdict:\s*(MALICIOUS|BENIGN)", g, re.I)
            slm_v=1 if v_match and v_match.group(1).upper()=="MALICIOUS" else 0
            if slm_v == batch_y[j]: n_correct += 1
            if re.search(r"TTP:\s*T\d{4}", g, re.I): n_ttp += 1
            out_exe=set(re.findall(r"\b[a-z]+\.exe\b", g.lower()))
            if out_exe: n_hallu += 1
            if g.count('Subject:') > 3: n_loop += 1
            if i < 16:  # save first 2 batches as sample
                gens_sample.append({"nid":s1[i+j]["nid"],"is_mal":batch_y[j],"gen":g})
    metrics_summary=metrics.stop()
    total=time.time()-t_start

    summary={
        "model":model_name,"lora":use_lora,
        "n":len(prompts),"batch":BATCH,
        "verdict_accuracy":n_correct/len(prompts),
        "ttp_rate":n_ttp/len(prompts),
        "hallucination_rate":n_hallu/len(prompts),
        "loop_rate":n_loop/len(prompts),
        "latency_mean_sec":float(np.mean(latencies)),
        "latency_p95_sec":float(np.percentile(latencies,95)),
        "latency_min_sec":float(np.min(latencies)),
        "latency_max_sec":float(np.max(latencies)),
        "total_time_sec":float(total),
        "decisions_per_sec":float(len(prompts)/total),
        "decisions_per_hour":float(len(prompts)/total*3600),
        "system_metrics":metrics_summary,
    }
    print(f"\n=== {model_name} ===", flush=True)
    for k,v in summary.items():
        if k=="system_metrics":
            print(f"  system_metrics:", flush=True)
            for k2,v2 in v.items():
                print(f"    {k2}: {v2}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)
    results_all.append({"summary":summary,"samples":gens_sample})

    del model, base, tok
    torch.cuda.empty_cache()

with open("P1/Output/results_phase2/slm-explain-metrics.json","w") as f:
    json.dump({"system":sys_info,"runs":results_all}, f, indent=2)
print(f"\nSAVED slm-explain-metrics.json", flush=True)