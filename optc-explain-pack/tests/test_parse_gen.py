import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eve  # noqa: E402

REC = {"events": [
    {"idx": 0, "eid": 1, "fields": {"Image": r"C:\Windows\System32\cmd.exe", "CommandLine": "cmd /c whoami"}},
    {"idx": 1, "eid": 10, "fields": {"TargetImage": r"C:\Windows\system32\lsass.exe", "GrantedAccess": "0x1fffff"}},
]}


def parse(gen, valid=None):
    ex = eve.Explainer.__new__(eve.Explainer)
    ex.valid_ids = valid
    return ex.parse_generation(REC, gen)


def main():
    fails = []

    def check(name, cond):
        if not cond:
            fails.append(name)

    r = parse('{"verdict": "malicious", "evidence": ["E1.TargetImage=C:\\\\Windows\\\\system32\\\\lsass.exe"], '
              '"technique": "T1003.001 OS Credential Dumping"}')
    check("json E#.Field=value", r["parse"] == "json" and r["evidence"][0]["field"] == "TargetImage")
    check("json technique field", r["technique"] == "T1003" and r["technique_sub"] == "T1003.001")
    check("verdict field", r["verdict_gen"] == "malicious")

    r = parse('```json\n{"verdict": "Malicious", "evidence": [{"EventIndex": "E0", "Evidence": '
              '"Process Create | Image: C:\\Windows\\System32\\cmd.exe | CommandLine: cmd /c whoami"}], '
              '"technique": {"ID": "T1059", "Name": "x"}}\n```')
    check("fence + bad escape + dict evidence", r["parse"] == "json" and len(r["evidence"]) == 2)
    check("dict technique", r["technique"] == "T1059")

    r = parse('{"verdict": "malicious", "technique": {"ID": "T1027_004", "Name": "x"}}')
    check("underscore sub-technique", r["technique"] == "T1027" and r["technique_sub"] == "T1027.004")
    r = parse('{"verdict": "malicious", "technique": {"ID": "T12345", "Name": "x"}}')
    check("5-digit id rejected", r["technique"] is None)

    r = parse('{"verdict": "benign", "evidence": [{"event_index": "[1]", "field": "GrantedAccess", "value": "0x1fff')
    check("truncated json repaired", r["parse"] == "json_repaired" and r["evidence"][0]["field"] == "GrantedAccess")

    r = parse('{"verdict": "malicious", "evidence": ["E0.CommandLine=C:\\\\AtomicRedTeam\\\\atomics\\\\T1087.002"], '
              '"technique": {"ID": "T100", "Name": "x"}}')
    check("no technique leak from evidence path", r["technique"] is None)
    check("fabricated evidence counted", r["n_evidence_fabricated"] == 1 and not r["evidence"])

    r = parse("Verdict: not malicious. Technique: T1059.", valid={"T1059"})
    check("free text verdict negation", r["verdict_gen"] == "benign")
    check("free text technique + valid id", r["technique"] == "T1059" and r["technique_valid_id"] is True)

    r = parse('{"verdict": "malicious", "technique": "T9999"}', valid={"T1059"})
    check("invalid ATT&CK id flagged", r["technique_valid_id"] is False)

    print("FAIL\n" + "\n".join(fails) if fails else "OK parse_generation: json/fence/escape/truncation/leak/negation/valid_id")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
