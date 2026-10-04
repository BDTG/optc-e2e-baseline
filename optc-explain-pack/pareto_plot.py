import argparse
import collections
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = {
    "ttp_acc_top1": ("ttp", "acc_top1", "TTP top-1"),
    "ttp_acc_in_scope": ("ttp", "acc_top1_in_scope", "TTP top-1 (in KB scope)"),
    "evidence_entailment": ("evidence", "entailment_ok", "Evidence entailment rate"),
    "verdict_auc": ("verdict", "auc", "Verdict AUC"),
}


def short_model(m):
    if not m:
        return "no-model"
    m = str(m).rsplit("/", 1)[-1]
    for k in ("0.5B", "1.5B", "7B"):
        if k in m:
            return k
    return m[:14]


def load_points(paths, metric, label_map):
    sec, key, _ = METRICS[metric]
    pts = []
    for p in paths:
        for t in json.loads(Path(p).read_text(encoding="utf-8")):
            q = (t.get(sec) or {}).get(key)
            lat = (t.get("latency") or {}).get("mean_ms")
            if q is None or lat is None or q != q or lat != lat:
                continue
            name = label_map.get(t["file"]) or f"{t.get('mode')} ({short_model(t.get('model'))})"
            pts.append({"name": name, "file": t["file"], "quality": float(q), "latency_ms": max(float(lat), 0.01),
                        "mode": t.get("mode"), "model": t.get("model")})
    seen, uniq = set(), []
    for x in pts:
        if x["file"] not in seen:
            seen.add(x["file"])
            uniq.append(x)
    nc = collections.Counter(x["name"] for x in uniq)
    for x in uniq:
        if nc[x["name"]] > 1:
            x["name"] = f'{x["name"]} [{Path(x["file"]).stem}]'
    return uniq


def pareto(pts):
    front = []
    for x in pts:
        dominated = any(y["latency_ms"] <= x["latency_ms"] and y["quality"] >= x["quality"]
                        and (y["latency_ms"] < x["latency_ms"] or y["quality"] > x["quality"]) for y in pts)
        x["pareto"] = not dominated
        if not dominated:
            front.append(x)
    return sorted(front, key=lambda x: x["latency_ms"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--metric", choices=sorted(METRICS), default="ttp_acc_top1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--title", default=None)
    ap.add_argument("--baseline", type=float, default=None)
    ap.add_argument("--baseline_label", default="majority")
    ap.add_argument("--device", default=None, help="ghi chu thiet bi do latency tren truc X, vd 'GPU RTX 5090'")
    a = ap.parse_args(argv)
    label_map = dict(x.split("=", 1) for x in a.labels if "=" in x)
    pts = load_points(a.evals, a.metric, label_map)
    if not pts:
        raise SystemExit("no points with both metric and latency")
    front = pareto(pts)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "mode", "model", "quality", "latency_ms", "pareto", "file"])
        w.writeheader()
        for x in sorted(pts, key=lambda x: x["latency_ms"]):
            w.writerow({k: x.get(k) for k in w.fieldnames})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, x in enumerate(sorted(pts, key=lambda x: (x["latency_ms"], x["quality"]))):
        ax.scatter(x["latency_ms"], x["quality"], s=60 if x["pareto"] else 35,
                   color="#1f3864" if x["pareto"] else "#9aa5b1", zorder=3)
        ax.annotate(x["name"], (x["latency_ms"], x["quality"]), textcoords="offset points",
                    xytext=(6, 4 + 9 * (i % 3) - 9), fontsize=8)
    ax.plot([x["latency_ms"] for x in front], [x["quality"] for x in front], color="#1f3864", lw=1.2, zorder=2)
    if a.baseline is not None:
        ax.axhline(a.baseline, ls="--", color="#c0504d", lw=1)
        ax.text(min(x["latency_ms"] for x in pts), a.baseline, f" {a.baseline_label}",
                color="#c0504d", fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Mean latency per alert (ms, log scale" + (f", {a.device})" if a.device else ")"))
    ax.set_ylabel(METRICS[a.metric][2])
    ax.set_title(a.title or f"Quality vs latency: {METRICS[a.metric][2]}")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(json.dumps({"points": len(pts), "pareto": [x["name"] for x in front], "png": str(out),
                      "csv": str(out.with_suffix(".csv"))}, indent=1))


if __name__ == "__main__":
    main()
