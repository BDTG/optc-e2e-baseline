import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_kb import POS

BENIGN = [
    {"eid": 1, "title": "Process Create", "fields": {"Image": r"C:\Windows\System32\dllhost.exe", "ParentImage": r"C:\Windows\System32\svchost.exe"}},
    {"eid": 7, "title": "Image loaded", "fields": {"Image": r"C:\Program Files\App\app.exe", "ImageLoaded": r"C:\Windows\System32\kernel32.dll", "Signed": "true"}},
    {"eid": 11, "title": "File created", "fields": {"Image": r"C:\Program Files\App\app.exe", "TargetFilename": r"C:\Users\<USERS_1>\Documents\report.docx"}},
    {"eid": 18, "title": "Pipe Connected", "fields": {"Image": r"C:\Program Files\Google\Chrome\chrome.exe", "PipeName": r"\LOCAL\crashpad_1"}},
]


def make(n, env, seed):
    rng = random.Random(seed)
    techs = sorted(POS)
    out = []
    for i in range(n):
        mal = i % 2 == 0
        evs = [dict(rng.choice(BENIGN)) for _ in range(rng.randint(1, 4))]
        gold, gt = [], []
        if mal:
            t = techs[(i // 2) % len(techs)] if rng.random() < 0.85 else "T1562"
            gold = [t]
            if t in POS:
                evs.insert(rng.randint(0, len(evs)), dict(POS[t][0]))
        events = []
        for k, e in enumerate(evs):
            events.append({"idx": k, "t": float(k), "eid": e["eid"], "user": "<USER_SYSTEM>", "title": e.get("title", ""),
                           "fields": dict(e["fields"]), "hints": gold if mal and e in [POS.get(gold[0], [None])[0]] else []})
        gt = [e["idx"] for e in events if e["hints"]]
        text = "\n".join(f"[E{e['idx']}] Event {e['eid']} {e['title']} | " +
                         " | ".join(f"{k}: {v}" for k, v in e["fields"].items()) for e in events)
        out.append({"sample_id": f"{env}_{i:04d}", "env": env, "label": int(mal), "verdict": "malicious" if mal else "benign",
                    "techniques": gold, "techniques_base": gold, "events": events, "text": text,
                    "spans": [], "evidence_gt_events": gt})
    return out


def run(args):
    print("$", " ".join(args))
    r = subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"FAILED: {args}")
    return r.stdout


def main():
    import tempfile
    tmp = Path(tempfile.gettempdir()) / "eve_smoke"
    tmp.mkdir(parents=True, exist_ok=True)
    model = tmp / "tiny"
    if not (model / "config.json").exists():
        run(["tests/make_tiny_model.py", str(model)])
    for env, n, seed in (("LAB", 40, 1), ("REAL", 40, 2)):
        with open(tmp / f"{env}.jsonl", "w", encoding="utf-8") as f:
            for r in make(n, env, seed):
                f.write(json.dumps(r) + "\n")
    run(["tests/test_kb.py"])
    run(["eve.py", "kbstats", "--data", str(tmp / "REAL.jsonl"), "--out", str(tmp / "kbstats.json")])
    files = []
    for mode in ("eve", "anchored", "json_enum", "json", "free"):
        for env in ("LAB", "REAL"):
            if env == "LAB" and mode != "eve":
                continue
            out = tmp / f"{env}_{mode}.jsonl"
            run(["eve.py", "run", "--data", str(tmp / f"{env}.jsonl"), "--model", str(model), "--mode", mode,
                 "--out", str(out), "--max_new_tokens", "24", "--progress", "0"])
            if env == "REAL":
                files.append(str(out))
    for rule in ("specific", "first", "alpha"):
        run(["eve.py", "run", "--data", str(tmp / "REAL.jsonl"), "--mode", "kb_only", "--kb_rule", rule,
             "--out", str(tmp / f"REAL_kb_{rule}.jsonl"), "--progress", "0"])
    kbo = {json.loads(l)["sample_id"]: json.loads(l) for l in open(tmp / "REAL_kb_specific.jsonl")}
    assert all(r["entailment_ok"] and r["minimal_ok"] for r in kbo.values() if r["technique"])
    assert all(r["model"] is None and r["latency_ms"] < 1000 for r in kbo.values())
    files.append(str(tmp / "REAL_kb_specific.jsonl"))
    rs = [json.loads(l) for l in open(tmp / "REAL_eve.jsonl")]
    bad = [r["sample_id"] for r in rs if r["technique"] and not (r["entailment_ok"] and r["minimal_ok"])]
    assert not bad, f"EVE produced unsupported or non-minimal output: {bad}"
    assert all(set(r["candidates"]) <= set(r["kb_candidates"]) for r in rs)
    assert all(set(r["candidates"]) == set(kbo[r["sample_id"]]["candidates"]) for r in rs)
    assert all(abs(sum(r["technique_probs"].values()) - 1) < 1e-6 for r in rs if r["technique_probs"])
    und = [r for r in rs if r["status"] == "undetermined"]
    assert all(not r["kb_candidates"] for r in und)
    run(["eval_eve.py", "--results", *files, "--data", str(tmp / "REAL.jsonl"), "--dev", str(tmp / "LAB_eve.jsonl"),
         "--bootstrap", "100", "--out", str(tmp / "eval.json")])
    run(["conformal.py", "--calib", str(tmp / "LAB_eve.jsonl"), "--test", str(tmp / "REAL_eve.jsonl"),
         "--weighted", "--calib_data", str(tmp / "LAB.jsonl"), "--test_data", str(tmp / "REAL.jsonl"),
         "--out", str(tmp / "conformal.json")])
    inj = {}
    for pl in ("instruction", "fake_fields", "keyword_stuffing"):
        run(["inject.py", "make", "--data", str(tmp / "REAL.jsonl"), "--out", str(tmp / f"REAL_inj_{pl}.jsonl"), "--payload", pl])
        run(["eve.py", "run", "--data", str(tmp / f"REAL_inj_{pl}.jsonl"), "--model", str(model), "--mode", "eve",
             "--out", str(tmp / f"REAL_eve_inj_{pl}.jsonl"), "--progress", "0"])
        run(["inject.py", "compare", "--clean", str(tmp / "REAL_eve.jsonl"), "--injected", str(tmp / f"REAL_eve_inj_{pl}.jsonl"),
             "--out", str(tmp / f"inj_{pl}.json")])
        inj[pl] = json.load(open(tmp / f"inj_{pl}.json"))
    assert inj["instruction"]["kb_candidates_change_rate"] == 0.0
    assert inj["fake_fields"]["kb_candidates_change_rate"] == 0.0
    assert inj["keyword_stuffing"]["technique_change_via_unattested_rate"] == inj["keyword_stuffing"]["technique_change_rate"]
    run(["eve.py", "run", "--data", str(tmp / "REAL.jsonl"), "--model", str(model), "--mode", "eve", "--attested_only",
         "--out", str(tmp / "REAL_eve_att.jsonl"), "--progress", "0"])
    run(["eve.py", "run", "--data", str(tmp / "REAL_inj_keyword_stuffing.jsonl"), "--model", str(model), "--mode", "eve",
         "--attested_only", "--out", str(tmp / "REAL_eve_att_inj.jsonl"), "--progress", "0"])
    run(["inject.py", "compare", "--clean", str(tmp / "REAL_eve_att.jsonl"), "--injected", str(tmp / "REAL_eve_att_inj.jsonl"),
         "--out", str(tmp / "inj_att.json")])
    att = json.load(open(tmp / "inj_att.json"))
    assert att["technique_change_rate"] == 0.0 and att["kb_candidates_change_rate"] == 0.0, att
    ev = json.load(open(tmp / "eval.json"))
    cf = json.load(open(tmp / "conformal.json"))
    assert all(x["bound_holds"] for x in cf["unweighted"])
    eve_row = [t for t in ev if t["mode"] == "eve"][0]
    assert eve_row["evidence"]["entailment_ok"] == 1.0
    print(f"SMOKE OK: {len(rs)} REAL records, eve entailment_ok=1.0, undetermined={len(und)}, "
          f"modes={[t['mode'] for t in ev]}, conformal alphas={[x['alpha'] for x in cf['unweighted']]}, "
          f"injection kb_change={ {k: v['kb_candidates_change_rate'] for k, v in inj.items()} }, "
          f"attested_only keyword_stuffing change={att['technique_change_rate']}")


if __name__ == "__main__":
    main()
