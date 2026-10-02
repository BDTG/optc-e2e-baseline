import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from eve import read_jsonl
from kb import KB


def event_text(e):
    kv = " | ".join(f"{k}: {str(v)[:200]}" for k, v in e.get("fields", {}).items())
    return f"Event {e.get('eid')} {e.get('title', '')} | {kv}"


def load_train(data, max_records, seed):
    recs = [r for r in read_jsonl(data) if r.get("env") == "LAB"]
    rng = random.Random(seed)
    rng.shuffle(recs)
    return recs[:max_records] if max_records else recs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kb", default=str(Path(__file__).with_name("tech_preconditions.json")))
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--max_train", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    kb = KB(a.kb)
    t0 = time.time()
    tr = load_train(a.train, a.max_train, a.seed)

    ev_x, ev_y = [], []
    for r in tr:
        gt = set(r.get("evidence_gt_events") or [])
        if not r.get("label") or not gt:
            continue
        for e in r["events"]:
            ev_x.append(event_text(e))
            ev_y.append(int(e["idx"] in gt))
    ev_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=100_000, min_df=2, sublinear_tf=True)
    ev_clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=a.seed)
    ev_clf.fit(ev_vec.fit_transform(ev_x), ev_y)

    tt = [(r.get("text", ""), sorted(r["techniques_base"])[0]) for r in tr if r.get("label") and r.get("techniques_base")]
    tvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=100_000, min_df=2, sublinear_tf=True)
    tclf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=a.seed)
    tclf.fit(tvec.fit_transform([x for x, _ in tt]), [y for _, y in tt])

    pos = [r for r in tr if r.get("label")]
    neg = [r for r in tr if not r.get("label")]
    rng = random.Random(a.seed)
    k = min(len(pos), len(neg))
    vtr = rng.sample(pos, k) + rng.sample(neg, k)
    vvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=100_000, min_df=2, sublinear_tf=True)
    vclf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=a.seed)
    vclf.fit(vvec.fit_transform([r.get("text", "") for r in vtr]), [int(r["label"]) for r in vtr])
    train_s = time.time() - t0

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as fo:
        for r in read_jsonl(a.test):
            t1 = time.time()
            evs = r["events"]
            s = ev_clf.decision_function(ev_vec.transform([event_text(e) for e in evs])) if evs else np.array([])
            top = sorted(range(len(evs)), key=lambda i: -s[i])[:a.topk]
            evidence = [{"event": evs[i]["idx"], "eid": evs[i].get("eid"), "field": f, "value": str(v)}
                        for i in sorted(top) for f, v in evs[i].get("fields", {}).items()]
            text = r.get("text", "")
            probs = tclf.predict_proba(tvec.transform([text]))[0]
            tp = {c: float(p) for c, p in zip(tclf.classes_, probs)}
            tech = max(tp, key=tp.get)
            vm = float(vclf.decision_function(vvec.transform([text]))[0])
            ent = kb.covers(tech) and kb.entails(tech, evidence) if evidence else False
            rec = {
                "sample_id": r.get("sample_id"), "env": r.get("env"), "mode": "extractive", "model": "tfidf-lr",
                "label": r.get("label"), "gold_techniques": r.get("techniques_base") or [],
                "verdict_margin_raw": vm, "verdict_margin": vm, "verdict": "malicious" if vm > 0 else "benign",
                "technique": tech, "technique_sub": None, "technique_probs": tp, "candidates": sorted(tp),
                "evidence": evidence, "witness_clause": None, "status": "unconstrained",
                "entailment_ok": bool(ent), "minimal_ok": bool(ent) and kb.minimal(tech, evidence),
                "kb_candidates": kb.candidates(evs), "n_events": len(evs),
                "latency_ms": round(1000 * (time.time() - t1), 1),
            }
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    summary = {"mode": "extractive", "n": n, "train_records": len(tr), "event_train_rows": len(ev_y),
               "event_train_pos": int(sum(ev_y)), "technique_classes": len(tclf.classes_), "train_s": round(train_s, 1)}
    Path(str(out_path) + ".summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
