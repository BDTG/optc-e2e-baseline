"""Viec 2 Paper 2 — LLM-as-judge blind, 3 judges x 3 endpoint families:
spark (/responses) + deepseek-v4.1-flash (/chat/completions) + qwen3.8-flash (/messages).
Rubric thay 0-2 x4 + pairwise. Resume tu partial. Doc key tu env.
GPT/Grok TAT (user cam quota) — du chi phi thuc te chi ~cent.
Usage: OPTC_ZEN_KEY=sk-... python llm_judge3.py [1|all]
"""
import json, os, sys, time, urllib.request, hashlib

KEY = os.environ.get("OPTC_ZEN_KEY", "")
assert KEY, "thieu OPTC_ZEN_KEY"
T = r"C:\Users\BDTG\AppData\Local\Temp"
NARR = T + r"\narratives-40.json"
KEYF = T + r"\human-eval-key.json"
S05 = T + r"\raw-explain-05b.json"
ST7 = T + r"\raw-explain-teacher.json"
OUT = T + r"\llm-judge-scores.json"
SID = "optc-p2-judge-blind"
JUDGES = ["muse-spark-1.3-contributor", "deepseek-v4.1-flash", "qwen3.8-flash", "gpt-5.6-luna", "grok-4.6"]
CHEAP = {"gpt-5.6-luna", "grok-4.6"}  # budget chat: narrative 1500 chars, max 800 out

PROMPT = """You are grading two security-analysis outputs (A and B) for the same Windows Sysmon narrative.
Narrative: {narr}
Output A: {a}
Output B: {b}
Each output is JSON {{verdict, mitre_technique, evidence, recommended_action}}.
Score EACH output 0-2 on: (1) evidence_support: does it point to real evidence in the narrative;
(2) attck_alignment: do verdict+technique fit the behavior; (3) rationale_consistency: fields coherent, no contradictions;
(4) actionability: can an analyst act on it.
Then pick pairwise winner (A/B/tie) for overall explanation quality.
Reply ONE JSON only: {{"A":{{"evidence_support":0,"attck_alignment":0,"rationale_consistency":0,"actionability":0}},"B":{{...}},"winner":"A","reason":"one sentence"}}"""


def fmt(x):
    return json.dumps({"verdict": "MALICIOUS" if x["verdict"] == 1 else "BENIGN",
                       "mitre_technique": x["ttp_pred"],
                       "evidence": x["evidence"],
                       "recommended_action": x.get("action_pred", "?")})


def post(url, payload, anthropic=False):
    h = {"Content-Type": "application/json", "x-opencode-session": SID,
         "User-Agent": "optc-eval/1.0"}
    if anthropic:
        h["x-api-key"] = KEY
        h["anthropic-version"] = "2023-06-01"
    else:
        h["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=h)
    return json.load(urllib.request.urlopen(req, timeout=300))


def call(judge, prompt, budget=False, tries=4):
    last = None
    for a in range(tries):
        try:
            return _call_once(judge, prompt, budget)
        except Exception as e:
            last = e
            time.sleep(5 * (a + 1))
    raise last


def _call_once(judge, prompt, budget=False):
    if judge == "qwen3.8-flash":
        d = post("https://opencode.ai/zen/go/v1/messages",
                 {"model": judge, "max_tokens": 3000,
                  "messages": [{"role": "user", "content": prompt}]}, anthropic=True)
        txt = "".join(c.get("text", "") for c in d.get("content", []) if c.get("type") == "text")
        return txt, {"input_tokens": d.get("usage", {}).get("input_tokens", 0),
                     "output_tokens": d.get("usage", {}).get("output_tokens", 0)}
    if judge == "deepseek-v4.1-flash":
        d = post("https://opencode.ai/zen/go/v1/chat/completions",
                 {"model": judge, "max_tokens": 3000,
                  "messages": [{"role": "user", "content": prompt}]})
        m = d.get("choices", [{}])[0].get("message", {})
        return m.get("content", ""), {"input_tokens": d.get("usage", {}).get("prompt_tokens", 0),
                                      "output_tokens": d.get("usage", {}).get("completion_tokens", 0)}
    d = post("https://opencode.ai/zen/go/v1/responses",
             {"model": judge, "input": prompt,
              "max_output_tokens": 800 if budget else 1500, "reasoning": {"effort": "low"}})
    txt = ""
    for o in d.get("output", []):
        if o.get("type") == "message":
            for c in o.get("content", []):
                if c.get("type") == "output_text":
                    txt += c.get("text", "")
    return txt, {"input_tokens": d.get("usage", {}).get("input_tokens", 0),
                 "output_tokens": d.get("usage", {}).get("output_tokens", 0)}


def main():
    narrs = json.load(open(NARR, encoding="utf-8"))
    nids = list(json.load(open(KEYF, encoding="utf-8")).keys())
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode != "all":
        nids = nids[:int(mode)]
    r05 = {x["nid"]: x for x in json.load(open(S05, encoding="utf-8"))["records"]}
    rt7 = {x["nid"]: x for x in json.load(open(ST7, encoding="utf-8"))["records"]}
    done = {}
    if os.path.exists(OUT):
        try:
            for r in json.load(open(OUT, encoding="utf-8")):
                done[(r["nid"], r["judge"])] = r
        except Exception:
            pass
    tin = tout = 0
    tasks = [(n, j) for n in nids for j in JUDGES if (n, j) not in done]
    for k, (nid, judge) in enumerate(tasks):
        swap = (int(hashlib.md5(nid.encode()).hexdigest(), 16) % 2 == 0)
        A, B = (r05[nid], rt7[nid]) if swap else (rt7[nid], r05[nid])
        Ais = "slm05" if swap else "t7b"
        nn = narrs.get(nid, "")
        bgt = judge in CHEAP
        if bgt: nn = nn[:1500]
        txt, usage = call(judge, PROMPT.replace("{narr}", nn).replace(
            "{a}", fmt(A)).replace("{b}", fmt(B)), budget=bgt)
        tin += usage.get("input_tokens", 0)
        tout += usage.get("output_tokens", 0)
        try:
            s, e = txt.index("{"), txt.rindex("}") + 1
            sc = json.loads(txt[s:e])
        except Exception:
            sc = {"_raw": txt[:300], "_parse": "fail"}
        sc.update({"nid": nid, "judge": judge, "A_is": Ais})
        done[(nid, judge)] = sc
        json.dump(list(done.values()), open(OUT, "w"), indent=1)
        print(f"{k+1}/{len(tasks)} {nid} {judge[:12]} winner={sc.get('winner', '?')} "
              f"(tok {tin}+{tout})", flush=True)
        time.sleep(2)
    print(f"DONE {len(done)} ratings, tokens in={tin} out={tout}", flush=True)


if __name__ == "__main__":
    main()
