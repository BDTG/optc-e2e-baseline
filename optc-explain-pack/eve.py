import argparse
import collections
import json
import math
import re
import sys
import time
from pathlib import Path

from kb import KB

SYSTEM = ("You are a SOC analyst. Explain the endpoint alert using only the events shown. "
          "Reply with one JSON object.")
USER = ("Alert events (Sysmon, one per line, [E#] is the event index):\n{text}\n\n"
        "Return JSON with keys: verdict (malicious or benign), evidence (list of \"E#.Field=value\" "
        "copied from the events), technique (MITRE ATT&CK technique ID and name).")
CF_TEXT = "N/A"
VERDICT_HEAD = '{"verdict": "'
TECH_HEAD = '{"verdict": "malicious", '
EVID_RE = re.compile(r'E(\d+)\.([A-Za-z]+)=([^"\]\n]+)')
TECH_RE = re.compile(r"\bT\d{4}(?:[._]\d{3})?(?!\d)")
EVENT_TAG_RE = re.compile(r"^\s*\[?E(\d+)\]?")
KV_RE = re.compile(r"([A-Za-z][A-Za-z0-9]*)\s*[:=]\s*([^|\n]+)")
ESC_RE = re.compile(r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})?')
VERDICT_RE = re.compile(r"\b(not\s+)?(malicious|benign)\b", re.I)
EVENT_KEYS = ("event", "eventindex", "event_index", "eventidx", "idx", "index", "e")
TEXT_KEYS = ("evidence", "value", "details", "description", "text")
MODES = ("eve", "kb_only", "anchored", "json_enum", "json", "free")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def render(events, max_val=200, max_chars=6000):
    out, n = [], 0
    for e in events:
        kv = " | ".join(f"{k}: {str(v)[:max_val]}" for k, v in e.get("fields", {}).items())
        ln = f"[E{e['idx']}] Event {e.get('eid')} {e.get('title', '')} | {kv}"
        if n + len(ln) + 1 > max_chars:
            out.append(f"[... {len(events) - len(out)} events truncated]")
            break
        out.append(ln)
        n += len(ln) + 1
    return "\n".join(out)


def loads_lenient(gen):
    """JSON dau tien trong output sinh; bo ```json, va dong ngoac neu bi cat do max_new_tokens."""
    s = gen.replace("```json", "").replace("```", "")
    i = s.find("{")
    if i < 0:
        return None, "text"
    s = s[i:]
    fixed = ESC_RE.sub(lambda m: m.group(0) if m.group(1) else "\\\\", s)
    for cand in (s, fixed):
        try:
            return json.JSONDecoder().raw_decode(cand)[0], "json"
        except ValueError:
            pass
    s = fixed
    stack, in_str, esc = [], False, False
    for ch in s:
        if in_str:
            esc = ch == "\\" and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    tail = s if in_str else s.rstrip().rstrip(",")
    if in_str:
        tail = (tail[:-1] if esc else tail) + '"'
    try:
        return json.loads(tail + "".join(reversed(stack))), "json_repaired"
    except ValueError:
        return None, "text"


def get_ci(d, key):
    return next((v for k, v in d.items() if k.lower() == key), None) if isinstance(d, dict) else None


def event_idx(v):
    m = re.search(r"\d+", str(v))
    return int(m.group()) if m else None


def evidence_claims(obj, gen):
    """(event|None, field, value) tu nhieu dang evidence ma SLM hay sinh."""
    claims = []

    def from_text(s, ev=None):
        hit = [(int(m.group(1)), m.group(2), m.group(3)) for m in EVID_RE.finditer(s)]
        if hit:
            return hit
        m = EVENT_TAG_RE.match(s)
        ev = int(m.group(1)) if m else ev
        return [(ev, k, v) for k, v in KV_RE.findall(s)]

    items = get_ci(obj, "evidence")
    if items is None:
        return from_text(gen) if obj is None else []
    for it in items if isinstance(items, list) else [items]:
        if isinstance(it, str):
            claims += from_text(it)
        elif isinstance(it, dict):
            ev = next((event_idx(v) for k, v in it.items() if k.lower() in EVENT_KEYS), None)
            fld, val = get_ci(it, "field"), get_ci(it, "value")
            if isinstance(fld, str) and val is not None:
                claims.append((ev, fld, str(val)))
                continue
            for k, v in it.items():
                if k.lower() in EVENT_KEYS:
                    continue
                if k.lower() in TEXT_KEYS and isinstance(v, str):
                    claims += from_text(v, ev)
                elif isinstance(v, (str, int, float)):
                    claims.append((ev, k, str(v)))
    return claims


def ev_str(item, max_val=160):
    return f"E{item['event']}.{item['field']}={str(item['value'])[:max_val]}"


def logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def softmax(d):
    if not d:
        return {}
    m = max(d.values())
    z = {k: math.exp(v - m) for k, v in d.items()}
    s = sum(z.values())
    return {k: v / s for k, v in z.items()}


def all_spans(events, max_spans, max_val=160):
    res = []
    for e in events:
        for k, v in e.get("fields", {}).items():
            if k in ("EventType", "Signature", "IntegrityLevel", "Protocol"):
                continue
            res.append({"event": e["idx"], "eid": e.get("eid"), "field": k, "value": str(v)[:max_val]})
            if len(res) >= max_spans:
                return res
    return res


def cap_witnesses(ws, cap):
    if len(ws) <= cap:
        return ws
    by_t = collections.defaultdict(list)
    for w in ws:
        by_t[w.technique].append(w)
    per = max(1, cap // len(by_t))
    out = [w for t in sorted(by_t) for w in by_t[t][:per]]
    return out[:max(cap, len(by_t))]


class Explainer:
    def __init__(self, scorer, kb, mode, max_witnesses=24, max_spans=48, topk_evidence=2,
                 max_new_tokens=384, valid_ids=None, attested_only=False):
        if mode not in MODES:
            raise ValueError(mode)
        self.s, self.kb, self.mode = scorer, kb, mode
        self.max_witnesses, self.max_spans, self.topk = max_witnesses, max_spans, topk_evidence
        self.max_new_tokens = max_new_tokens
        self.valid_ids = valid_ids
        self.attested_only = attested_only
        self.enum = dict(kb.enum)
        self.cf_prefix_v = scorer.chat_prefix(SYSTEM, USER.format(text=CF_TEXT), VERDICT_HEAD)
        a, b = scorer.score(self.cf_prefix_v, ['malicious"', 'benign"'])
        self.cf_margin = a - b

    def tech_label(self, t):
        return f"{t} {self.enum.get(t[:5], self.kb.names.get(t[:5], ''))}".strip()

    def pmi(self, prefix, cf_prefix, conts):
        a = self.s.score(prefix, conts)
        b = self.s.score(cf_prefix, conts)
        return [x - y for x, y in zip(a, b)], a

    def verdict(self, text):
        p = self.s.chat_prefix(SYSTEM, USER.format(text=text), VERDICT_HEAD)
        a, b = self.s.score(p, ['malicious"', 'benign"'])
        return a - b, (a - b) - self.cf_margin

    def run_eve(self, rec, text):
        ws = self.kb.witnesses(rec["events"])
        if self.attested_only:
            ws = [w for w in ws if w.attested]
        ws = cap_witnesses(ws, self.max_witnesses)
        if not ws:
            return {"technique": None, "technique_sub": None, "technique_probs": {}, "candidates": [],
                    "evidence": [], "witness_clause": None, "status": "undetermined", "n_witnesses": 0}
        base = self.s.chat_prefix(SYSTEM, USER.format(text=text), TECH_HEAD + '"evidence": ')
        cfp = self.s.chat_prefix(SYSTEM, USER.format(text=CF_TEXT), TECH_HEAD + '"evidence": ')
        conts = [json.dumps([ev_str(i) for i in w.as_items()]) + ', "technique": "' + self.tech_label(w.technique) + '"}'
                 for w in ws]
        pm, raw = self.pmi(base, cfp, conts)
        by_t = collections.defaultdict(list)
        for w, v in zip(ws, pm):
            by_t[w.technique].append(v)
        tscore = {t: logsumexp(v) for t, v in by_t.items()}
        probs = softmax(tscore)
        t_star = max(probs, key=probs.get)
        j = max((i for i, w in enumerate(ws) if w.technique == t_star), key=lambda i: pm[i])
        w = ws[j]
        return {"technique": t_star, "technique_sub": w.sub, "technique_probs": probs,
                "candidates": sorted(by_t), "evidence": w.as_items(), "witness_clause": w.clause,
                "evidence_attested": w.attested, "status": "entailed", "n_witnesses": len(ws),
                "witness_scores": [{"t": x.technique, "clause": x.clause, "event": x.event, "pmi": v, "raw": r}
                                   for x, v, r in zip(ws, pm, raw)]}

    def rank_enum(self, prefix_tail, text):
        ts = sorted(self.enum)
        p = self.s.chat_prefix(SYSTEM, USER.format(text=text), prefix_tail)
        cfp = self.s.chat_prefix(SYSTEM, USER.format(text=CF_TEXT), prefix_tail)
        pm, _ = self.pmi(p, cfp, [self.tech_label(t) + '"' for t in ts])
        return softmax(dict(zip(ts, pm)))

    def run_json_enum(self, rec, text):
        probs = self.rank_enum(TECH_HEAD + '"technique": "', text)
        t = max(probs, key=probs.get)
        return {"technique": t, "technique_sub": None, "technique_probs": probs, "candidates": sorted(self.enum),
                "evidence": [], "witness_clause": None, "status": "unconstrained"}

    def run_anchored(self, rec, text):
        spans = all_spans(rec["events"], self.max_spans)
        if not spans:
            return self.run_json_enum(rec, text)
        p = self.s.chat_prefix(SYSTEM, USER.format(text=text), TECH_HEAD + '"evidence": [')
        cfp = self.s.chat_prefix(SYSTEM, USER.format(text=CF_TEXT), TECH_HEAD + '"evidence": [')
        pm, _ = self.pmi(p, cfp, [json.dumps(ev_str(s)) + "]" for s in spans])
        order = sorted(range(len(spans)), key=lambda i: -pm[i])[:self.topk]
        chosen = [spans[i] for i in sorted(order)]
        tail = TECH_HEAD + '"evidence": ' + json.dumps([ev_str(s) for s in chosen]) + ', "technique": "'
        probs = self.rank_enum(tail, text)
        t = max(probs, key=probs.get)
        return {"technique": t, "technique_sub": None, "technique_probs": probs, "candidates": sorted(self.enum),
                "evidence": chosen, "witness_clause": None, "status": "unconstrained"}

    def parse_generation(self, rec, gen):
        fields = collections.defaultdict(dict)
        for e in rec["events"]:
            for k, v in e.get("fields", {}).items():
                fields[e["idx"]][k.lower()] = (k, str(v), e.get("eid"))
        obj, parse = loads_lenient(gen)

        def match(ev, fld, val):
            val = str(val).replace("\\\\", "\\").strip().strip('"').rstrip("\\").strip()
            if len(val) < 3:
                return None
            for i in ([ev] if ev in fields else list(fields)):
                hit = fields[i].get(str(fld).lower())
                if hit and (val.lower() in hit[1].lower() or hit[1].lower().startswith(val.lower())):
                    return {"event": i, "eid": hit[2], "field": hit[0], "value": hit[1]}
            return None

        items, seen, fabricated = [], set(), 0
        for ev, fld, val in evidence_claims(obj, gen):
            it = match(ev, fld, val)
            if it is None:
                fabricated += 1
            elif (it["event"], it["field"]) not in seen:
                seen.add((it["event"], it["field"]))
                items.append(it)

        tech_src = get_ci(obj, "technique")
        tech_txt = json.dumps(tech_src) if tech_src is not None else gen
        ids = TECH_RE.findall(tech_txt) or (TECH_RE.findall(gen) if tech_src is None else [])
        t = ids[0].replace("_", ".") if ids else None
        vsrc = get_ci(obj, "verdict")
        vm = VERDICT_RE.search(str(vsrc) if vsrc is not None else gen)
        vg = None if vm is None else ("benign" if vm.group(1) else vm.group(2).lower())
        valid = None if t is None else (t in self.valid_ids or t[:5] in self.valid_ids if self.valid_ids else True)
        return {"technique": t[:5] if t else None, "technique_sub": t if t and "." in t else None,
                "technique_probs": {}, "candidates": [], "evidence": items, "witness_clause": None,
                "status": "unconstrained", "generation": gen, "verdict_gen": vg, "parse": parse,
                "n_evidence_generated": len(items) + fabricated, "n_evidence_fabricated": fabricated,
                "technique_valid_id": valid}

    def run_gen(self, rec, text, json_mode):
        head = VERDICT_HEAD if json_mode else ""
        p = self.s.chat_prefix(SYSTEM, USER.format(text=text), head)
        gen, _ = self.s.generate(p, self.max_new_tokens)
        return self.parse_generation(rec, head + gen)

    def explain(self, rec):
        t0 = time.time()
        text = rec.get("text") or render(rec["events"])
        m_raw, m = self.verdict(text)
        if self.mode == "eve":
            out = self.run_eve(rec, text)
        elif self.mode == "anchored":
            out = self.run_anchored(rec, text)
        elif self.mode == "json_enum":
            out = self.run_json_enum(rec, text)
        else:
            out = self.run_gen(rec, text, self.mode == "json")
        t = out.get("technique")
        ev = out.get("evidence") or []
        out["entailment_ok"] = bool(t) and bool(ev) and self.kb.covers(t) and self.kb.entails(t, ev)
        out["minimal_ok"] = out["entailment_ok"] and self.kb.minimal(t, ev)
        ws_all = self.kb.witnesses(rec["events"])
        out["kb_candidates"] = sorted({w.technique for w in ws_all if w.attested or not self.attested_only})
        out.update({
            "sample_id": rec.get("sample_id"), "env": rec.get("env"), "mode": self.mode,
            "model": self.s.name, "label": rec.get("label"), "attested_only": self.attested_only,
            "gold_techniques": rec.get("techniques_base") or [],
            "verdict_margin_raw": m_raw, "verdict_margin": m,
            "verdict": "malicious" if m > 0 else "benign",
            "n_events": len(rec["events"]), "latency_ms": round(1000 * (time.time() - t0), 1),
        })
        return out


def load_valid_ids(path):
    if not path or not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return set(d.get("valid", [])) | {x[:5] for x in d.get("valid", [])}


def run_kb_only(rec, kb, rule, max_witnesses=24, attested_only=False):
    """Baseline xep hang deu (kb_only): chon technique bang luat thuan, khong model.
    Rule: first = witness o event som nhat; alpha = ma technique nho nhat;
    specific = co sub-technique truoc, roi nhieu item hon, roi event som hon.
    technique_probs deu tren C(E) (xep hang deu). Verdict luat: co witness thi malicious.
    """
    t0 = time.time()
    ws = kb.witnesses(rec["events"])
    if attested_only:
        ws = [w for w in ws if w.attested]
    ws = cap_witnesses(ws, max_witnesses)
    if not ws:
        out = {"technique": None, "technique_sub": None, "technique_probs": {}, "candidates": [],
               "evidence": [], "witness_clause": None, "status": "undetermined", "n_witnesses": 0}
    else:
        if rule == "alpha":
            pick = min(range(len(ws)), key=lambda i: (ws[i].technique, i))
        elif rule == "specific":
            pick = min(range(len(ws)), key=lambda i: (not bool(ws[i].sub), -len(ws[i].as_items()), ws[i].event, i))
        else:
            pick = min(range(len(ws)), key=lambda i: (ws[i].event, i))
        w = ws[pick]
        cands = sorted({x.technique for x in ws})
        out = {"technique": w.technique, "technique_sub": w.sub,
               "technique_probs": {t: 1.0 / len(cands) for t in cands},
               "candidates": cands, "evidence": w.as_items(), "witness_clause": w.clause,
               "evidence_attested": w.attested, "status": "entailed", "n_witnesses": len(ws)}
    t = out.get("technique")
    ev = out.get("evidence") or []
    out["entailment_ok"] = bool(t) and bool(ev) and kb.covers(t) and kb.entails(t, ev)
    out["minimal_ok"] = out["entailment_ok"] and kb.minimal(t, ev)
    ws_all = kb.witnesses(rec["events"])
    out["kb_candidates"] = sorted({w.technique for w in ws_all if w.attested or not attested_only})
    out.update({
        "sample_id": rec.get("sample_id"), "env": rec.get("env"), "mode": "kb_only",
        "kb_rule": rule, "model": None, "label": rec.get("label"), "attested_only": attested_only,
        "gold_techniques": rec.get("techniques_base") or [],
        "verdict_margin_raw": 1.0 if out["candidates"] else -1.0,
        "verdict_margin": 1.0 if out["candidates"] else -1.0,
        "verdict": "malicious" if out["candidates"] else "benign",
        "n_events": len(rec["events"]), "latency_ms": round(1000 * (time.time() - t0), 1),
    })
    return out


def cmd_run(a):
    kb = KB(a.kb)
    kb_only = a.mode == "kb_only"
    if kb_only:
        scorer = ex = None
        explain = lambda rec: run_kb_only(rec, kb, a.kb_rule, a.max_witnesses, a.attested_only)  # noqa: E731
    else:
        if not a.model:
            raise SystemExit("--model is required except for --mode kb_only")
        from slm import Scorer
        scorer = Scorer(a.model, device=a.device, dtype=a.dtype, batch_size=a.batch_size,
                        max_len=a.max_len, threads=a.threads, prefix_cache=not a.no_prefix_cache)
        ex = Explainer(scorer, kb, a.mode, a.max_witnesses, a.max_spans, a.topk_evidence,
                       a.max_new_tokens, load_valid_ids(a.attack_map), a.attested_only)
        explain = ex.explain
    done = set()
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if a.resume and out_path.exists():
        done = {r["sample_id"] for r in read_jsonl(out_path)}
    stats = collections.Counter()
    lat = []
    with open(out_path, "a" if a.resume else "w", encoding="utf-8") as f:
        for i, rec in enumerate(read_jsonl(a.data)):
            if a.limit and i >= a.limit:
                break
            if rec.get("sample_id") in done:
                continue
            if a.only_malicious and not rec.get("label"):
                continue
            try:
                r = explain(rec)
            except ValueError as e:
                if kb_only:
                    raise
                stats["skipped_" + type(e).__name__] += 1
                print(f"skip {rec.get('sample_id')}: {e}", file=sys.stderr)
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            stats[r["status"]] += 1
            stats["entailment_ok"] += int(r["entailment_ok"])
            stats["with_technique"] += int(bool(r.get("technique")))
            if not kb_only:
                stats["verdict_tie"] += int(abs(r["verdict_margin_raw"]) < 1e-6)
            stats["n"] += 1
            lat.append(r["latency_ms"])
            if a.progress and stats["n"] % a.progress == 0:
                print(f"[{stats['n']}] {dict(stats)} mean_ms={sum(lat) / len(lat):.0f}", flush=True)
    n = max(stats["n"], 1)
    summary = {"mode": a.mode, **({"kb_rule": a.kb_rule} if kb_only else {}),
               "model": None if kb_only else a.model, "n": stats["n"], "counts": dict(stats),
               "entailment_ok_rate_among_predicted": stats["entailment_ok"] / max(stats["with_technique"], 1),
               **({} if kb_only else {"verdict_tie_rate": stats["verdict_tie"] / n}),
               "latency_ms_mean": sum(lat) / len(lat) if lat else None,
               "latency_ms_p90": sorted(lat)[int(0.9 * (len(lat) - 1))] if lat else None,
               "cf_verdict_margin": None if kb_only else ex.cf_margin,
               "n_forward": 0 if kb_only else scorer.n_forward}
    Path(str(out_path) + ".summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    if summary.get("verdict_tie_rate", 0) > 0.01:
        print("WARNING verdict tie rate > 1%: check dtype/numerics", file=sys.stderr)


def cmd_kbstats(a):
    kb = KB(a.kb)
    per = collections.defaultdict(collections.Counter)
    tot = collections.Counter()
    clause_hits = collections.Counter()
    for i, rec in enumerate(read_jsonl(a.data)):
        if a.limit and i >= a.limit:
            break
        ws = kb.witnesses(rec["events"])
        cands = {w.technique for w in ws}
        for w in ws:
            clause_hits[f"{w.technique}:{w.clause}"] += 1
        y = rec.get("label")
        gold = set(rec.get("techniques_base") or [])
        tot["mal" if y else "ben"] += 1
        tot[("mal" if y else "ben") + "_nonempty_C"] += int(bool(cands))
        if y and gold:
            tot["mal_with_gold"] += 1
            in_kb = {t for t in gold if kb.covers(t)}
            tot["mal_gold_in_kb"] += int(bool(in_kb))
            tot["mal_gold_in_kb_and_C"] += int(bool(in_kb & cands))
            gt_ev = set(rec.get("evidence_gt_events") or [])
            if in_kb & cands and gt_ev:
                wev = {w.event for w in ws if w.technique in in_kb}
                tot["hint_gt_available"] += 1
                tot["witness_hits_hint_gt"] += int(bool(wev & gt_ev))
        for t in kb.techniques:
            g = t in gold
            c = t in cands
            if y:
                per[t]["mal_gold"] += int(g)
                per[t]["mal_gold_and_C"] += int(g and c)
                per[t]["mal_C"] += int(c)
                per[t]["mal_C_and_gold"] += int(c and g)
            else:
                per[t]["ben_C"] += int(c)
    def r(a_, b_):
        return round(a_ / b_, 4) if b_ else None
    res = {
        "kb_version": kb.version, "n": tot["mal"] + tot["ben"], "totals": dict(tot),
        "kb_scope": r(tot["mal_gold_in_kb"], tot["mal_with_gold"]),
        "kb_recall_in_scope": r(tot["mal_gold_in_kb_and_C"], tot["mal_gold_in_kb"]),
        "benign_nonempty_C": r(tot["ben_nonempty_C"], tot["ben"]),
        "witness_agrees_hint_gt": r(tot["witness_hits_hint_gt"], tot["hint_gt_available"]),
        "per_technique": {t: {"recall": r(c["mal_gold_and_C"], c["mal_gold"]),
                              "precision_mal": r(c["mal_C_and_gold"], c["mal_C"]),
                              "benign_fire": r(c["ben_C"], tot["ben"]),
                              "n_gold": c["mal_gold"]} for t, c in sorted(per.items())},
        "clause_hits": dict(clause_hits.most_common()),
    }
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True)
    r.add_argument("--model", default=None)
    r.add_argument("--mode", choices=MODES, default="eve")
    r.add_argument("--kb_rule", choices=("specific", "first", "alpha"), default="first")
    r.add_argument("--kb", default=str(Path(__file__).with_name("tech_preconditions.json")))
    r.add_argument("--out", required=True)
    r.add_argument("--device", default="cpu")
    r.add_argument("--dtype", default="float32")
    r.add_argument("--batch_size", type=int, default=8)
    r.add_argument("--max_len", type=int, default=4096)
    r.add_argument("--threads", type=int, default=None)
    r.add_argument("--no_prefix_cache", action="store_true")
    r.add_argument("--max_witnesses", type=int, default=24)
    r.add_argument("--max_spans", type=int, default=48)
    r.add_argument("--topk_evidence", type=int, default=2)
    r.add_argument("--max_new_tokens", type=int, default=384)
    r.add_argument("--attack_map", default=None)
    r.add_argument("--attested_only", action="store_true")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--only_malicious", action="store_true")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--progress", type=int, default=25)
    r.set_defaults(func=cmd_run)
    k = sub.add_parser("kbstats")
    k.add_argument("--data", required=True)
    k.add_argument("--kb", default=str(Path(__file__).with_name("tech_preconditions.json")))
    k.add_argument("--out", default=None)
    k.add_argument("--limit", type=int, default=0)
    k.set_defaults(func=cmd_kbstats)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
