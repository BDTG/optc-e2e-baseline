import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kb import KB


def ev(eid, t, **d):
    data = "".join(f"<Data Name='{k}'>{v}</Data>" for k, v in d.items())
    return (f"<Event xmlns='http://schemas.microsoft.com/win/2004/08/events/event'><System>"
            f"<Provider Name='Microsoft-Windows-Sysmon' Guid='{{5770385f}}'/><EventID>{eid}</EventID>"
            f"<TimeCreated SystemTime='2022-01-12T10:00:{t:02d}.1234567Z'/><Computer>h</Computer>"
            f"<Security UserID='S-1-5-18'/></System><EventData><Data Name='RuleName'>technique_id=T1003.001</Data>"
            f"<Data Name='UtcTime'>2022-01-12 10:00:{t:02d}.123</Data>{data}</EventData></Event>")


def write(root, tech, name, lines, new=True):
    d = Path(root, "datasets", "attack_techniques", tech, name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "windows-sysmon.log").write_text("\n".join(lines) + "\n")
    if new:
        y = (f"mitre_technique:\n- {tech}\ndatasets:\n- name: windows-sysmon\n"
             f"  path: /datasets/attack_techniques/{tech}/{name}/windows-sysmon.log\n"
             f"  sourcetype: XmlWinEventLog\n  source: XmlWinEventLog:Microsoft-Windows-Sysmon/Operational\n")
    else:
        y = (f"technique:\n- {tech}\ndataset:\n- https://media.githubusercontent.com/media/splunk/attack_data/master/"
             f"datasets/attack_techniques/{tech}/{name}/windows-sysmon.log\n")
    (d / f"{name}.yml").write_text(y)


def build(root):
    svc = lambda t, g: ev(1, t, ProcessGuid=g, Image="C:\\Windows\\System32\\svchost.exe",
                          CommandLine="svchost -k netsvcs", ParentProcessGuid="{P0}")
    write(root, "T1003.001", "atomic_red_team", [
        ev(1, 1, ProcessGuid="{H}", Image="C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
           CommandLine="powershell -c Invoke-AtomicTest T1003.001", ParentProcessGuid="{P0}"),
        ev(1, 2, ProcessGuid="{C1}", Image="C:\\AtomicRedTeam\\procdump.exe",
           CommandLine="procdump.exe -ma lsass.exe C:\\Temp\\l.dmp", ParentProcessGuid="{H}"),
        ev(10, 3, SourceProcessGUID="{C1}", SourceImage="C:\\AtomicRedTeam\\procdump.exe",
           TargetImage="C:\\Windows\\system32\\lsass.exe", GrantedAccess="0x1fffff"),
        svc(4, "{B}"),
        ev(1, 5, ProcessGuid="{S}", Image="C:\\Program Files\\SplunkUniversalForwarder\\bin\\splunkd.exe",
           ParentProcessGuid="{H}")])
    for i, tech in enumerate(["T1055", "T1059.001", "T1543.003", "T1490", "T1218.011"]):
        img = "C:\\Windows\\System32\\rundll32.exe" if tech == "T1218.011" else f"C:\\Users\\Public\\e{i}.exe"
        lines = [svc(1, f"{{B{i}}}"), ev(1, 2, ProcessGuid=f"{{X{i}}}", Image=img, CommandLine=f"run {tech}",
                                         ParentProcessGuid="{P0}")]
        if tech == "T1055":
            lines.append(ev(8, 3, SourceProcessGuid=f"{{X{i}}}", SourceImage=img,
                            TargetImage="C:\\Windows\\explorer.exe"))
        write(root, tech, "sim", lines, new=(i % 2 == 0))
    d = Path(root, "datasets", "attack_techniques", "T1047", "lfs")
    d.mkdir(parents=True, exist_ok=True)
    (d / "windows-sysmon.log").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:x\n")
    (d / "lfs.yml").write_text("mitre_technique:\n- T1047\ndatasets:\n- name: s\n  path: /x/windows-sysmon.log\n"
                               "  source: XmlWinEventLog:Microsoft-Windows-Sysmon/Operational\n")


def main():
    tmp = Path(tempfile.mkdtemp())
    build(tmp)
    out = tmp / "out.jsonl"
    r = subprocess.run([sys.executable, "attackdata_to_eve.py", "convert", "--root", str(tmp), "--out", str(out),
                        "--bg_min_files", "3"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = json.loads(Path(str(out) + ".manifest.json").read_text())
    recs = [json.loads(l) for l in open(out)]
    kb = KB(ROOT / "tech_preconditions.json")
    fails = []
    if m["file_stats"].get("file_lfs_pointer") != 1:
        fails.append("lfs pointer not detected")
    t1003 = [x for x in recs if x["techniques_base"] == ["T1003"]]
    if len(t1003) != 1 or t1003[0]["label_source"] != "tree":
        fails.append(f"tree mode: {[(x['techniques'], x['label_source']) for x in t1003]}")
    elif "T1003" not in kb.candidates(t1003[0]["events"]):
        fails.append("KB did not fire T1003 on procdump->lsass")
    if any("splunkd" in x["text"].lower() or "svchost -k" in x["text"].lower() for x in recs):
        fails.append("agent or background process leaked into alerts")
    if any("technique_id" in x["text"] for x in recs):
        fails.append("RuleName hint leaked into model input")
    got = set(m["alerts_per_technique"])
    for t in ("T1055", "T1059", "T1543", "T1490", "T1218"):
        if t not in got:
            fails.append(f"missing alerts for {t}")
    t1055 = [x for x in recs if x["techniques_base"] == ["T1055"]]
    if not t1055 or "T1055" not in kb.candidates(t1055[0]["events"]):
        fails.append("KB did not fire T1055 on CreateRemoteThread")
    print("FAIL\n" + "\n".join(fails) if fails else f"OK attackdata adapter: {len(recs)} alerts, {m['alerts_per_technique']}")
    return 1 if fails else 0


if __name__ == "__main__":
    os.chdir(ROOT)
    sys.exit(main())
