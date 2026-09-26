"""
Extract provenance chains tu EVTX (Sysmon Event ID 1 = process create).
Build CAR-like chains: subject + cmd + parent_chain + event_seq.
Filter: T1050 AtomicRedTeam sample (known attack) + 1 benign file.
"""
import Evtx.Evtx as evtx
import defusedxml.ElementTree as ET
import json, re, os, random
from pathlib import Path
from collections import defaultdict, Counter

random.seed(42)
EVTX_DIR = Path("data/external/evtx")
NS = {'ns': 'http://schemas.microsoft.com/win/2004/08/events/event'}

# Heuristic detect malicious vs benign by keyword in path/cmdline
SUSPICIOUS_KEYWORDS = [
    "AtomicRedTeam", "atomic", "invoke-atomic", "atomaticredteam",
    "meterpreter", "msfvenom", "nc.exe", "powershell -enc",
    "rundll32", "regsvr32", "vssadmin", "comsvcs",
    "certutil", "bitsadmin", "wmic", "mshta",
]

# file -> (label, technique_id, description)
FILE_LABELS = {
    "PanacheSysmon_vs_AtomicRedTeam01.evtx": ("mixed", "T1050+T1059+T1064", "Atomic Red Team execution + discovery"),
    "WinDefender_Events_1117_1116_AtomicRedTeam.evtx": ("mixed", "T1562", "WinDefender events with atomic tests"),
    "DE_RDP_Tunneling_4624.evtx": ("mixed", "T1572", "Defense Evasion: RDP tunneling"),
    "bits_openvpn.evtx": ("mixed", "T1071.001", "Command and Control: BITS transfer"),
    "Malware/rundll32_cmd_schtask.evtx": ("mixed", "T1218.011+T1053", "Malware: rundll32 + scheduled task"),
    "Malware/sideloading_injection_persistence_run_key.evtx": ("mixed", "T1574.002+T1060", "Malware: DLL sideloading + run key"),
}

def parse_evtx(path):
    """Stream process events, return list of {pid, cmd, parent_pid, parent_cmd, image, parent_image}"""
    events = []
    try:
        with evtx.Evtx(str(path)) as log:
            for record in log.records():
                try:
                    root = ET.fromstring(record.xml())
                    eid = root.find('.//ns:EventID', NS)
                    if eid is None or eid.text != '1':
                        continue
                    data = {}
                    for d in root.findall('.//ns:Data', NS):
                        name = d.get('Name')
                        if name:
                            data[name] = d.text
                    if 'ProcessId' in data and 'Image' in data:
                        events.append({
                            'pid': data.get('ProcessId'),
                            'cmd': data.get('CommandLine'),
                            'image': data.get('Image'),
                            'parent_pid': data.get('ParentProcessId'),
                            'parent_cmd': data.get('ParentCommandLine'),
                            'parent_image': data.get('ParentImage'),
                        })
                except Exception:
                    continue
    except Exception as e:
        print(f"  parse fail {path}: {e}")
    return events

def is_suspicious(cmd, image):
    """Heuristic: malicious if any suspicious keyword in cmd/image"""
    text = (cmd or '') + ' ' + (image or '')
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in SUSPICIOUS_KEYWORDS)

def build_chains(events, max_alerts=300):
    """Group events into provenance chains (parent-child relationships)"""
    by_pid = {e['pid']: e for e in events if e['pid']}
    chains = []
    for e in events:
        # Build chain by walking up parent_pid
        chain = []
        cur = e
        visited = set()
        while cur and cur['pid'] not in visited:
            visited.add(cur['pid'])
            chain.append({
                'pid': cur['pid'],
                'cmd': cur['cmd'],
                'image': cur['image'],
                'op': 'EXECUTE',
            })
            parent_pid = cur.get('parent_pid')
            if parent_pid and parent_pid in by_pid:
                cur = by_pid[parent_pid]
            else:
                cur = None
            if len(chain) >= 5:
                break
        chain.reverse()
        if len(chain) >= 2:  # only keep chains with parent context
            chains.append({
                'nid': e['pid'] + '_' + str(random.randint(1000, 9999)),
                'is_suspicious': is_suspicious(e['cmd'], e['image']),
                'self_label': f"subject {e['image']} | cmd: {e['cmd'] or 'None'}",
                'parent_chain': chain,
                'event_seq': [],  # could add network/file events
                'src': 'evtx',
            })
    return chains[:max_alerts]

# Build dataset
out = []
for fpath in sorted(EVTX_DIR.glob("**/*.evtx")):
    fname = fpath.name
    if any(skip in str(fpath) for skip in ["Other"]):
        continue
    print(f"parsing {fpath.name}...", flush=True)
    events = parse_evtx(fpath)
    if not events:
        continue
    chains = build_chains(events, max_alerts=50)
    print(f"  -> {len(events)} events, {len(chains)} chains", flush=True)
    out.extend(chains)

print(f"\nTotal chains: {len(out)}")
n_susp = sum(1 for c in out if c['is_suspicious'])
print(f"Suspicious: {n_susp}, Benign: {len(out)-n_susp}")

# Check cmdline richness
cmd_stats = Counter()
for c in out:
    has_cmd = any("| cmd:" in pc['cmd'] or pc['cmd'] for pc in c['parent_chain'])
    for pc in c['parent_chain']:
        if pc['cmd'] and pc['cmd'].strip() and pc['cmd'].lower() != 'none':
            cmd_stats['cmd_present'] += 1
            break
    else:
        cmd_stats['cmd_none'] += 1

print(f"\nCmdline stats: {dict(cmd_stats)}")
print(f"% with non-empty cmd: {cmd_stats['cmd_present']/len(out)*100:.1f}%")

# Save
with open("P1/Output/data/evtx-chains.jsonl","w") as f:
    for c in out:
        f.write(json.dumps(c) + "\n")
print(f"\nSaved P1/Output/data/evtx-chains.jsonl")