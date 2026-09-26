"""product_explain.py — SẢN PHẨM Paper 2 (Việc 1, spec thầy chốt).
Nhận 1 alert (narrative text) -> SLM 0.5B constrained scoring -> JSON 4 trường.
Ràng buộc: chạy thuần CPU (device=cpu mặc định), không phụ thuộc GPU.
Cơ chế y hệt bench đã đo (prefix/fewshot/enum/rule/playbook) để số Việc 2/3 khớp.

Usage:
  python P1/Code/product_explain.py --text "...narrative..." [--model Qwen/Qwen2.5-0.5B-Instruct] [--device cpu]
  python P1/Code/product_explain.py --file alert.txt
  echo ... | python P1/Code/product_explain.py
Output (stdout): {"verdict":..,"mitre_technique":..,"evidence":..,"recommended_action":..,"latency_s":..}
"""
import argparse, json, re, sys, time
import torch
import torch.nn.functional as F

TTP_ENUM = ["T1003", "T1059", "T1053", "T1071", "T1218", "T1027",
            "T1562", "T1574", "T1490", "T1087", "T1082", "T1083", "none"]
VERDICTS = ["MALICIOUS", "BENIGN"]
FEWSHOT = (
    'Example: Chain: powershell.exe | cmd: powershell -nop -w hidden -enc SQBFAFgAIA==\n'
    '{"verdict":"MALICIOUS","technique_id":"T1059","evidence_field":"cmdline","confidence":"high"}\n'
    'Example: Chain: svchost.exe | chrome.exe | cmd: none\n'
    '{"verdict":"BENIGN","technique_id":"none","evidence_field":"none","confidence":"high"}\n'
)
PREFIX = ("You are a security analyst. Classify this Windows process provenance chain.\n"
          + FEWSHOT + "Chain: {chain}\n"
          'Respond with one JSON object: {{"verdict":"{v}","technique_id":"{t}","evidence_field":"{e}"}}\n'
          "JSON:")
PLAYBOOK = {
    "T1059": "terminate_process", "T1055": "terminate_process",
    "T1218": "terminate_process", "T1053": "terminate_process",
    "T1003": "isolate_host", "T1490": "isolate_host", "T1562": "isolate_host",
    "T1071": "get_network_flow",
    "T1547": "query_registry", "T1543": "query_registry", "T1574": "query_registry",
    "T1546": "query_registry",
    "T1027": "get_file_metadata", "T1070": "get_file_metadata",
    "T1036": "get_file_metadata", "T1083": "get_file_metadata",
    "T1553": "get_file_metadata",
    "T1082": "check_threat_intel", "T1087": "check_threat_intel",
    "T1046": "check_threat_intel",
}


def rule_evidence(text):
    t = text.lower()
    if "cmd: " in t and "cmd: none" not in t:
        return "cmdline"
    if "|" in text:
        return "parent_chain"
    if "event" in t:
        return "event_seq"
    return "none"


def batch_scores(model, tok, prefix, candidates):
    pre = tok(prefix, add_special_tokens=False)["input_ids"]
    cands = [tok(c, add_special_tokens=False)["input_ids"] for c in candidates]
    seqs = [pre + c for c in cands]
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    inp = torch.tensor([[pad] * (L - len(s)) + s for s in seqs]).to(model.device)
    mask = (inp != pad).long()
    with torch.no_grad():
        logits = model(input_ids=inp, attention_mask=mask).logits.float()
    logp = F.log_softmax(logits, dim=-1)
    out = []
    for row, c in enumerate(cands):
        m = len(c)
        lp = sum(float(logp[row, L - m - 1 + j, c[j]]) for j in range(m)) / m
        out.append(lp)
    return out


def explain(text, model, tok):
    t0 = time.time()
    txt = text[:800]
    base = PREFIX.replace("{chain}", txt)
    vs = batch_scores(model, tok, base + '{"verdict":"', VERDICTS)
    verdict = "MALICIOUS" if vs[0] > vs[1] else "BENIGN"
    ts = batch_scores(model, tok, base + '{"verdict":"' + verdict + '","technique_id":"', TTP_ENUM)
    ttp = TTP_ENUM[int(__import__("numpy").argmax(ts))]
    evid = rule_evidence(text)
    if verdict == "BENIGN" or ttp == "none":
        action = "no_action"
    else:
        action = PLAYBOOK.get(re.match(r"(T\d{4})", ttp).group(1), "check_threat_intel")
    return {"verdict": verdict, "mitre_technique": ttp, "evidence": evid,
            "recommended_action": action, "latency_s": round(time.time() - t0, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None)
    ap.add_argument("--file", default=None)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.file:
        text = open(a.file, encoding="utf-8").read()
    elif a.text:
        text = a.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        ap.error("can --text / --file / stdin")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float32, device_map=None).to(a.device)
    model.eval()
    print(json.dumps(explain(text, model, tok), ensure_ascii=False))


if __name__ == "__main__":
    main()
