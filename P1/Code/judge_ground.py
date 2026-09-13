"""Phuong an (b) — judge cham GROUNDEDNESS noi dung rationale (1 tieu chi, thang 0-2).
Blind: judge khong biet rationale cua model nao (tron slm05/teacher, danh ma X/Y).
3 judges re: spark / deepseek-v4.1-flash / qwen3.8-flash.
Usage: OPTC_ZEN_KEY=... python judge_ground.py all
"""
import hashlib, json, os, re, sys, time, urllib.request, uuid

KEY = os.environ.get("OPTC_ZEN_KEY", "")
SID = "sess-" + uuid.uuid4().hex[:16]  # session routing bat buoc (Zen Go)
BASE = "https://opencode.ai/zen/go/v1"
JUDGES = ["muse-spark-1.3-contributor", "deepseek-v4.1-flash", "qwen3.8-flash"]

PROMPT = """You are grading whether an analyst-style EXPLANATION is grounded in the alert narrative.
Narrative:
{narr}
Explanation R:
{a}
Score ONE criterion 0-2:
groundedness: every factual claim in R appears in or follows directly from the narrative (2 = all grounded, 1 = mostly, one minor stretch, 0 = invented facts / contradicts narrative).
Reply exactly: G:<0-2> | NOTE:<under 20 words>"""

EP = {"muse-spark-1.3-contributor": ("/responses", "resp"),
      "deepseek-v4.1-flash": ("/chat/completions", "chat"),
      "qwen3.8-flash": ("/messages", "msg")}


def post(path, payload):
    q = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                               headers={"Authorization": "Bearer " + KEY,
                                        "x-api-key": KEY,
                                        "Content-Type": "application/json",
                                        "User-Agent": "curl/8.0",
                                        "x-opencode-session": SID})
    return json.load(urllib.request.urlopen(q, timeout=300))


def call(judge, prompt, tries=4):
    last = None
    for a in range(tries):
        try:
            kind = EP[judge][1]
            if kind == "resp":
                r = post("/responses", {"model": judge, "input": prompt,
                                        "max_output_tokens": 2500,
                                        "reasoning": {"effort": "low"}})
                t, u = _txt_resp(r), {"in": 0, "out": 0}
                if not t.strip() or t.strip().startswith("{"):
                    raise ValueError("empty/incomplete resp: " + t[:80])
                return t, u
            if kind == "chat":
                r = post("/chat/completions", {"model": judge,
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 1500})
                u = r.get("usage", {})
                t = r["choices"][0]["message"]["content"] or ""
                if not t.strip():
                    raise ValueError("empty chat content")
                return t, {"in": u.get("prompt_tokens", 0),
                       "out": u.get("completion_tokens", 0)}
            r = post("/messages", {"model": judge, "max_tokens": 1500,
                     "messages": [{"role": "user", "content": prompt}]})
            u = r.get("usage", {})
            t = "".join(b.get("text", "") for b in r.get("content", []))
            if not t.strip():
                raise ValueError("empty msg content")
            return t, {
                "in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0)}
        except Exception as e:
            last = e
            time.sleep(5 * (a + 1))
    raise last


def _txt_resp(r):
    o = r.get("output", [])
    if isinstance(o, list):
        t = []
        for b in o:
            for c in (b.get("content", []) if isinstance(b, dict) else []):
                if c.get("type") == "output_text":
                    t.append(c.get("text", ""))
        if t:
            return "\n".join(t)
    return json.dumps(r)[:500]


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    T = r"C:\Users\BDTG\AppData\Local\Temp"
    r05 = {x["nid"]: x["rationale"] for x in json.load(open(f"{T}/rationale-05b.json", encoding="utf-8")) if "VERDICT" in x["rationale"]}
    r7b = {x["nid"]: x["rationale"] for x in json.load(open(f"{T}/rationale-7b.json", encoding="utf-8")) if "VERDICT" in x["rationale"]}
    print("compliant:", len(r05), len(r7b), flush=True)
    narrs = {}
    for l in open(f"{T}/adgen-ttp-bench.jsonl", encoding="utf-8"):
        o = json.loads(l)
        ch = o.get("parent_chain", []) or []
        narrs[o["nid"]] = " | ".join([(c.get("msg") or "") for c in ch[-5:]])[:1500]
    nids = sorted(set(r05) & set(r7b))
    try:
        D = json.load(open(f"{T}/judge-ground-scores.json", encoding="utf-8"))
    except Exception:
        D = []
    done = {(r["nid"], r["judge"], r["src"]) for r in D}
    jobs = []
    for nid in nids:
        for src, rr in (("slm05", r05), ("t7b", r7b)):
            for j in JUDGES:
                if (nid, j, src) not in done:
                    jobs.append((nid, src, rr[nid], j))
    if only != "all":
        jobs = jobs[:int(only)]
    print("jobs:", len(jobs), flush=True)
    for i, (nid, src, rat, j) in enumerate(jobs):
        try:
            txt, us = call(j, PROMPT.replace("{narr}", narrs[nid]).replace("{a}", rat[:1200]))
            m = re.search(r"G:\s*([0-2])", txt)
            D.append({"nid": nid, "src": src, "judge": j,
                      "groundedness": int(m.group(1)) if m else None, "raw": txt[:200]})
        except Exception as e:
            D.append({"nid": nid, "src": src, "judge": j,
                      "groundedness": None, "raw": "JOB_FAIL: " + str(e)[:120]})
        if (i + 1) % 10 == 0:
            json.dump(D, open(f"{T}/judge-ground-scores.json", "w"), indent=1)
            print(f"{i+1}/{len(jobs)}", flush=True)
    json.dump(D, open(f"{T}/judge-ground-scores.json", "w"), indent=1)
    print("DONE", len(D), flush=True)


main()
