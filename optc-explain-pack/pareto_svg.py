"""Ve lai bieu do Pareto (quality vs latency) thanh SVG de doc + xuat PNG qua Edge/Chrome headless.

Doc CSV do pareto_plot.py sinh ra (name, mode, model, quality, latency_ms, pareto, file);
cot tuy chon ci_lo, ci_hi -> ve thanh CI doc.
  python pareto_svg.py --csv results/adgen/stt58_pareto/pareto_ttp.csv --baseline 0.229 \
      --note "Latency: SLM on GPU, tf-idf on CPU" --out results/adgen/stt58_pareto/pareto_ttp_v2.svg
"""
import argparse
import csv
import html
import math
import shutil
import subprocess
from pathlib import Path

FONT = "Segoe UI, Inter, Helvetica Neue, Arial, sans-serif"
# mode -> (ten ngan tren nhan, nhom legend, mau)
MODES = {
    "eve": ("EVE", "EVE (constrained + KB witness)", "#2563eb"),
    "kb_only": ("KB-only", "KB rules only", "#7c3aed"),
    "extractive": ("Extractive tf-idf", "Extractive baseline (tf-idf + LR)", "#059669"),
    "json_enum": ("JSON-enum", "Constrained JSON (enum)", "#d97706"),
    "anchored": ("Anchored", "Unconstrained generation", "#94a3b8"),
    "json": ("JSON", "Unconstrained generation", "#94a3b8"),
    "free": ("Free-text", "Unconstrained generation", "#94a3b8"),
}
SHAPES = [("0.5B", "circle", "Qwen2.5-0.5B"), ("1.5B", "triangle", "Qwen2.5-1.5B"),
          ("7B", "diamond", "Qwen2.5-7B"), ("", "square", "no LLM")]


def size_of(model):
    m = (model or "").rsplit("/", 1)[-1]
    for k in ("0.5B", "1.5B", "7B"):
        if k in m:
            return k
    return ""


def plot_default_name(mode, model):
    """Ten pareto_plot.py tu sinh khi khong co --labels (khop short_model ben do)."""
    m = str(model).rsplit("/", 1)[-1] if model else ""
    short = next((k for k in ("0.5B", "1.5B", "7B") if k in m), m[:14]) if m else "no-model"
    return f"{mode} ({short})"


def fmt_ms(v):
    return f"{v / 1000:g} s" if v >= 1000 else f"{v:g} ms"


def text_w(s, size):
    return sum(0.62 if (c.isupper() or c.isdigit()) else 0.3 if c in " .,:;|" else 0.52 for c in s) * size


class Layout:
    """Dat nhan tham lam: thu lan luot cac vi tri, bo vi tri de len hop da chiem / ra ngoai vung ve."""

    def __init__(self, bounds):
        self.boxes, self.bounds = [], bounds

    def free(self, b):
        x0, y0, x1, y1 = self.bounds
        if b[0] < x0 or b[1] < y0 or b[2] > x1 or b[3] > y1:
            return False
        return not any(b[0] < o[2] and b[2] > o[0] and b[1] < o[3] and b[3] > o[1] for o in self.boxes)

    def add(self, b):
        self.boxes.append(b)


def marker(p, x, y):
    shape = next((s for k, s, _ in SHAPES if k == p["size"]), "circle")
    fo = "1" if p["pareto"] else "0.8"
    st = 'stroke="#0f172a" stroke-width="2.2"' if p["pareto"] else 'stroke="#ffffff" stroke-width="1.5"'
    c = p["color"]
    if shape == "diamond":
        r = 9.5
        return f'<polygon points="{x},{y - r} {x + r},{y} {x},{y + r} {x - r},{y}" fill="{c}" fill-opacity="{fo}" {st}/>'
    if shape == "triangle":
        r = 9
        return (f'<polygon points="{x},{y - r} {x + r * .87:.1f},{y + r / 2} {x - r * .87:.1f},{y + r / 2}" '
                f'fill="{c}" fill-opacity="{fo}" {st}/>')
    if shape == "square":
        r = 7
        return f'<rect x="{x - r}" y="{y - r}" width="{2 * r}" height="{2 * r}" rx="2" fill="{c}" fill-opacity="{fo}" {st}/>'
    return f'<circle cx="{x}" cy="{y}" r="7.5" fill="{c}" fill-opacity="{fo}" {st}/>'


def render(pts, a):
    W, H = 1000, 640
    L, R, T, B = 86, 36, 92, 150
    pw, ph = W - L - R, H - T - B
    lo = math.log10(min(p["lat"] for p in pts)) - 0.35
    hi = math.log10(max(p["lat"] for p in pts)) + 0.35
    ymax = math.ceil(max([p.get("ci_hi") or p["q"] for p in pts] + [a.baseline or 0]) * 10 + 0.5) / 10

    def X(v):
        return L + (math.log10(v) - lo) / (hi - lo) * pw

    def Y(q):
        return T + ph - q / ymax * ph

    o = []
    e = o.append
    e(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">')
    e(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    e(f'<text x="{L}" y="38" font-size="22" font-weight="600" fill="#0f172a">{html.escape(a.title)}</text>')
    e(f'<text x="{L}" y="62" font-size="14" fill="#475569">{html.escape(a.subtitle)}</text>')
    e(f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" fill="#f8fafc"/>')

    for k in range(int(round(ymax * 10)) + 1):
        q = k / 10
        y = Y(q)
        e(f'<line x1="{L}" x2="{L + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        e(f'<text x="{L - 10}" y="{y + 4.5:.1f}" font-size="13" text-anchor="end" fill="#475569">{q * 100:.0f}%</text>')
    for d in range(math.floor(lo), math.ceil(hi) + 1):
        for m in range(1, 10):
            v = m * 10 ** d
            if not (lo <= math.log10(v) <= hi):
                continue
            x = X(v)
            e(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{T + ph}" stroke="{"#e2e8f0" if m == 1 else "#f1f5f9"}"/>')
            if m == 1:
                e(f'<text x="{x:.1f}" y="{T + ph + 22}" font-size="13" text-anchor="middle" fill="#475569">{fmt_ms(v)}</text>')
    e(f'<line x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}" stroke="#94a3b8"/>')
    e(f'<line x1="{L}" x2="{L}" y1="{T}" y2="{T + ph}" stroke="#94a3b8"/>')
    e(f'<text x="{L + pw / 2}" y="{T + ph + 48}" font-size="14" text-anchor="middle" fill="#334155">'
      f'Mean latency per alert (log scale) \u2192 slower</text>')
    e(f'<text transform="translate(28 {T + ph / 2}) rotate(-90)" font-size="14" text-anchor="middle" '
      f'fill="#334155">{html.escape(a.ylabel)} \u2192 better</text>')

    lay = Layout((L + 2, T + 2, L + pw - 2, T + ph - 2))
    for p in pts:
        p["x"], p["y"] = round(X(p["lat"]), 1), round(Y(p["q"]), 1)
        lay.add((p["x"] - 11, p["y"] - 11, p["x"] + 11, p["y"] + 11))
        if p.get("ci_lo") is not None:
            lay.add((p["x"] - 6, Y(p["ci_hi"]) - 2, p["x"] + 6, Y(p["ci_lo"]) + 2))

    front = sorted((p for p in pts if p["pareto"]), key=lambda p: p["lat"])
    if front:
        path = [(X(front[0]["lat"]), Y(front[0]["q"]))]
        for p, n in zip(front, front[1:]):
            path += [(X(n["lat"]), Y(p["q"])), (X(n["lat"]), Y(n["q"]))]
        path.append((L + pw, Y(front[-1]["q"])))
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in path)
        e(f'<polygon points="{d} {L + pw},{T + ph} {path[0][0]:.1f},{T + ph}" fill="#2563eb" fill-opacity="0.05"/>')
        e(f'<polyline points="{d}" fill="none" stroke="#1e3a8a" stroke-width="1.8" stroke-dasharray="6 4"/>')
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            lay.add((min(x0, x1) - 3, min(y0, y1) - 3, max(x0, x1) + 3, max(y0, y1) + 3))
        tx = "Pareto frontier"
        tw = text_w(tx, 12)
        x0, x1, ly = path[0][0], path[1][0], path[0][1]
        cands = [(x0 + (x1 - x0) * f, ly + dy) for dy in (19, -9) for f in (0.5, 0.3, 0.7, 0.15, 0.85)]
        fx, fy = next(((x, y) for x, y in cands if lay.free((x - tw / 2 - 4, y - 14, x + tw / 2 + 4, y + 4))),
                      cands[0])
        e(f'<text x="{fx:.1f}" y="{fy:.1f}" font-size="12" text-anchor="middle" fill="#1e3a8a" font-style="italic">{tx}</text>')
        lay.add((fx - tw / 2 - 4, fy - 14, fx + tw / 2 + 4, fy + 4))

    if a.baseline is not None:
        y = Y(a.baseline)
        e(f'<line x1="{L}" x2="{L + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="5 4"/>')
        tx = f"{a.baseline_label} ({a.baseline * 100:.1f}%)"
        e(f'<text x="{L + 8}" y="{y - 7:.1f}" font-size="12" fill="#dc2626">{html.escape(tx)}</text>')
        lay.add((L + 6, y - 20, L + 12 + text_w(tx, 12), y - 2))
        lay.add((L, y - 3, L + pw, y + 3))

    eves = sorted((p for p in pts if p["mode"] == "eve"), key=lambda p: p["lat"])
    if len(eves) >= 2 and abs(eves[0]["q"] - eves[-1]["q"]) < 0.05:
        s, b = eves[0], eves[-1]
        x0, x1 = s["x"] + 14, b["x"] - 14
        y = min([s["y"], b["y"]] + [Y(p["ci_hi"]) for p in eves if p.get("ci_hi") is not None]) - 26
        e('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
          'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#2563eb"/></marker></defs>')
        e(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x0:.1f}" y2="{y:.1f}" stroke="#2563eb" stroke-width="1.5" marker-end="url(#ah)"/>')
        tx = f"same accuracy, {b['lat'] / s['lat']:.0f}\u00d7 faster with {s['label']}"
        cx = (x0 + x1) / 2
        e(f'<text x="{cx:.1f}" y="{y - 8:.1f}" font-size="13" text-anchor="middle" fill="#1d4ed8" font-weight="600" '
          f'paint-order="stroke" stroke="#f8fafc" stroke-width="4">{html.escape(tx)}</text>')
        tw = text_w(tx, 13)
        lay.add((min(x0, cx - tw / 2), y - 24, max(x1, cx + tw / 2), y + 4))

    labels = []
    for p in sorted(pts, key=lambda p: (not p["pareto"], p["y"])):
        tx = f'{p["label"]} \u00b7 {p["q"] * 100:.1f}%'
        fs = 13.5 if p["pareto"] else 12.5
        w, h = text_w(tx, fs), fs + 5
        placed = None
        for dist in (14, 32, 52, 76):
            dg = dist * .72
            cands = [(dist, -h / 2), (-dist - w, -h / 2), (-w / 2, -dist - h), (-w / 2, dist),
                     (dg, -dg - h), (dg, dg), (-dg - w, -dg - h), (-dg - w, dg)]
            for dx, dy in cands:
                bx = (p["x"] + dx, p["y"] + dy, p["x"] + dx + w, p["y"] + dy + h)
                if lay.free(bx):
                    gap = math.hypot(max(bx[0] - p["x"], 0, p["x"] - bx[2]), max(bx[1] - p["y"], 0, p["y"] - bx[3]))
                    placed = (bx, gap > 6)
                    break
            if placed:
                break
        if not placed:
            placed = ((p["x"] + 12, p["y"] - h / 2, p["x"] + 12 + w, p["y"] + h / 2), False)
        lay.add(placed[0])
        labels.append((p, tx, fs, placed))

    for p, _, _, (bx, leader) in labels:
        if leader:
            cx = min(max(p["x"], bx[0]), bx[2])
            cy = min(max(p["y"], bx[1]), bx[3])
            e(f'<line x1="{p["x"]}" y1="{p["y"]}" x2="{cx:.1f}" y2="{cy:.1f}" stroke="#94a3b8" stroke-width="1"/>')
    for p in pts:
        if p.get("ci_lo") is not None:
            y0, y1 = Y(p["ci_lo"]), Y(p["ci_hi"])
            e(f'<g stroke="{p["color"]}" stroke-width="1.4" stroke-opacity="0.55"><line x1="{p["x"]}" x2="{p["x"]}" '
              f'y1="{y0:.1f}" y2="{y1:.1f}"/><line x1="{p["x"] - 5}" x2="{p["x"] + 5}" y1="{y0:.1f}" y2="{y0:.1f}"/>'
              f'<line x1="{p["x"] - 5}" x2="{p["x"] + 5}" y1="{y1:.1f}" y2="{y1:.1f}"/></g>')
    for p in sorted(pts, key=lambda p: p["pareto"]):
        e(marker(p, p["x"], p["y"]))
    for p, tx, fs, (bx, _) in labels:
        wt, col = ("600", "#0f172a") if p["pareto"] else ("400", "#475569")
        e(f'<text x="{bx[0]:.1f}" y="{bx[3] - 5:.1f}" font-size="{fs}" font-weight="{wt}" fill="{col}" '
          f'paint-order="stroke" stroke="#f8fafc" stroke-width="4" stroke-linejoin="round">{html.escape(tx)}</text>')

    ly, x = T + ph + 80, L
    groups = []
    for p in pts:
        if p["group"] not in [g for g, _ in groups]:
            groups.append((p["group"], p["color"]))
    for g, c in groups:
        e(f'<rect x="{x}" y="{ly - 10}" width="12" height="12" rx="3" fill="{c}"/>')
        e(f'<text x="{x + 18}" y="{ly}" font-size="12.5" fill="#334155">{html.escape(g)}</text>')
        x += 18 + text_w(g, 12.5) + 24
    ly, x = ly + 26, L
    sizes = {p["size"] for p in pts}
    for k, _, name in SHAPES:
        if k in sizes:
            e(marker({"size": k, "color": "#64748b", "pareto": False}, x + 7, ly - 4))
            e(f'<text x="{x + 20}" y="{ly}" font-size="12.5" fill="#334155">{html.escape(name)}</text>')
            x += 20 + text_w(name, 12.5) + 24
    e(f'<circle cx="{x + 7}" cy="{ly - 4}" r="7" fill="#ffffff" stroke="#0f172a" stroke-width="2.2"/>')
    e(f'<text x="{x + 20}" y="{ly}" font-size="12.5" fill="#334155">dark outline = Pareto-optimal</text>')
    if any(p.get("ci_lo") is not None for p in pts):
        x += 20 + text_w("dark outline = Pareto-optimal", 12.5) + 24
        e(f'<line x1="{x + 7}" x2="{x + 7}" y1="{ly - 12}" y2="{ly + 4}" stroke="#64748b" stroke-width="1.4"/>')
        e(f'<text x="{x + 16}" y="{ly}" font-size="12.5" fill="#334155">95% bootstrap CI</text>')
    if a.note:
        e(f'<text x="{W - R}" y="{H - 12}" font-size="11.5" text-anchor="end" fill="#64748b">{html.escape(a.note)}</text>')
    e("</svg>")
    return "\n".join(o), W, H


def export_png(svg, png, w, h, scale):
    cands = [shutil.which("msedge"), shutil.which("chrome"), shutil.which("google-chrome"), shutil.which("chromium"),
             r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
             r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    exe = next((c for c in cands if c and Path(c).exists()), None)
    if not exe:
        return False
    if png.exists():
        png.unlink()
    try:
        subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={w},{h}",
                        f"--force-device-scale-factor={scale}", "--default-background-color=FFFFFFFF",
                        f"--screenshot={png.resolve()}", svg.resolve().as_uri()], capture_output=True, timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return png.exists()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help="duong dan .svg; PNG cung ten")
    ap.add_argument("--baseline", type=float, default=None)
    ap.add_argument("--baseline_label", default="Majority class")
    ap.add_argument("--title", default="TTP top-1 accuracy vs. latency per alert")
    ap.add_argument("--subtitle", default="LAB \u2192 REAL transfer \u00b7 REAL_test_matched (n = 1000 alerts)")
    ap.add_argument("--ylabel", default="TTP top-1 accuracy")
    ap.add_argument("--note", default="")
    ap.add_argument("--scale", type=float, default=2.0, help="he so phong to PNG (2 = 2000x1280)")
    ap.add_argument("--no_png", action="store_true")
    a = ap.parse_args(argv)

    pts = []
    with open(a.csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            short, group, color = MODES.get(r["mode"], (r["mode"], r["mode"], "#64748b"))
            sz = size_of(r["model"])
            name = (r.get("name") or "").strip()
            label = name if name and name != plot_default_name(r["mode"], r["model"]) else f"{short} {sz}".strip()
            ci = [float(r[k]) if r.get(k) not in (None, "") else None for k in ("ci_lo", "ci_hi")]
            pts.append({"mode": r["mode"], "size": sz, "q": float(r["quality"]), "lat": float(r["latency_ms"]),
                        "pareto": r["pareto"] == "True", "label": label,
                        "group": group, "color": color, "ci_lo": ci[0], "ci_hi": ci[1]})
    if not pts:
        raise SystemExit("CSV rong")
    svg, W, H = render(pts, a)
    out = Path(a.out).with_suffix(".svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print("svg:", out)
    if not a.no_png:
        png = out.with_suffix(".png")
        print("png:", png) if export_png(out, png, W, H, a.scale) else print("WARN: khong co Edge/Chrome; chi xuat SVG")


if __name__ == "__main__":
    main()
