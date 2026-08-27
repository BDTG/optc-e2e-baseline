"""
Buoc 3/3: TTP Holdout that su - tao provenance chains tu Atomic Red Team YAML.
Sinh alerts theo format CAR-like (subject/predicate/object) giong OpTC V2.
Labels: malicious cho atomic test, benign cho benign baseline (PowerShell
binh thuong, cmd help, system tasks).

Test BERT 150M (train tren OpTC V2) tren TTP unseen de xem co that su
generalize duoc khong.
"""
import json, os, random, yaml, re, sys
from pathlib import Path

ARTROOT = Path("D:/OpTC-thesis/data/atomic-red-team/art-repo/atomics")
OUT = Path("P1/Output")
OUT.mkdir(exist_ok=True)

random.seed(42)
TTP_IDS = ["T1218.001","T1218.002","T1059.001","T1027","T1027.001",
           "T1053.005","T1003.001","T1490","T1087.001","T1082"]

# ---- benign templates (giong OpTC benign V2 style: svchost/Idle/cmd=None) ----
BENIGN_PROCS = [
    ("svchost.exe", "-k netsvcs -s Schedule"),
    ("svchost.exe", "-k LocalServiceNoNetwork"),
    ("lsass.exe", None),
    ("services.exe", None),
    ("System", "Idle"),
    ("explorer.exe", None),
    ("powershell.exe", "Get-Process"),
    ("powershell.exe", "Get-Service"),
    ("cmd.exe", "dir /b"),
    ("cmd.exe", "echo hello"),
    ("taskhostw.exe", None),
]

def benign_chain():
    proc, cmd = random.choice(BENIGN_PROCS)
    msgs = []
    for _ in range(random.randint(2,4)):
        prev = proc
        msgs.append(f"subject {prev} | cmd: {cmd}")
    return msgs

# ---- parse 1 atomic test -> malicious provenance chain ----
def parse_atomic(yaml_path):
    """Extract command lines from atomic test. Returns list of (name, cmd)."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    out = []
    if not data: return out
    tests = data.get("atomic_tests", []) if isinstance(data, dict) else []
    for test in tests:
        name = test.get("name","?")
        executor = test.get("executor", {})
        cmd = executor.get("command", "").strip()
        if cmd and len(cmd) > 3:
            out.append((name, cmd))
    return out

def malicious_chain_from_cmd(cmd):
    """Build pseudo-provenance chain from a single atomic command line."""
    # Split cmd on pipes / semicolons / && / quotes
    tokens = re.split(r"[|;&]|\\\"|\"", cmd)
    tokens = [t.strip().strip('"') for t in tokens if t.strip() and len(t.strip()) > 2]
    if not tokens:
        tokens = [cmd.strip()[:80]]
    msgs = []
    for t in tokens[:5]:
        # make it CAR-like
        msgs.append(f"subject process.exe | cmd: {t[:120]}")
    return msgs

# ---- build dataset ----
alerts = []
# Benign: 300
for _ in range(300):
    chain = benign_chain()
    alerts.append({
        "nid": random.randint(100000, 999999),
        "label": 1,  # benign
        "self_label": chain[-1],
        "parent_chain": [{"node": f"node_{i}", "op":"CREATE",
                          "msg": m} for i,m in enumerate(chain)],
        "event_seq": [],
        "src":"ttp_template",
    })

# Malicious: 1 per atomic test
mal_count = 0
for tid in TTP_IDS:
    yp = ARTROOT / tid / f"{tid}.yaml"
    if not yp.exists(): continue
    cmds = parse_atomic(yp)
    for name, cmd in cmds[:2]:  # max 2 tests per technique
        chain = malicious_chain_from_cmd(cmd)
        alerts.append({
            "nid": random.randint(1000000, 9999999),
            "label": 0,
            "self_label": chain[-1],
            "parent_chain": [{"node": f"node_{tid}", "op":"EXECUTE",
                              "msg": m} for m in chain],
            "event_seq": [],
            "src": f"ttp:{tid}/{name[:30]}",
            "ttp_id": tid,
            "ttp_cmd": cmd[:200],
        })
        mal_count += 1

random.shuffle(alerts)
print(f"TTP holdout: {len(alerts)} alerts, {mal_count} malicious, {len(alerts)-mal_count} benign")

with open(OUT/"ttp_holdout.jsonl","w") as f:
    for a in alerts:
        f.write(json.dumps(a) + "\n")
print(f"Saved to {OUT}/ttp_holdout.jsonl")