"""Tao attack_id_map_v15.json (ID technique/sub-technique hop le, khong revoked/deprecated) cho eve.py --attack_map.

  python make_attack_map.py                      # tai enterprise-attack-15.1.json tu mitre/cti
  python make_attack_map.py --stix local.json    # dung file STIX co san
"""
import argparse
import json
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/mitre/cti/ATT%26CK-v15.1/enterprise-attack/enterprise-attack.json"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stix", default=None)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", default=str(Path(__file__).with_name("attack_id_map_v15.json")))
    a = ap.parse_args(argv)
    if a.stix:
        bundle = json.loads(Path(a.stix).read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(a.url, timeout=120) as r:
            bundle = json.loads(r.read().decode("utf-8"))
    valid, names, dropped = set(), {}, 0
    for o in bundle["objects"]:
        if o.get("type") != "attack-pattern":
            continue
        ext = next((x for x in o.get("external_references", []) if x.get("source_name") == "mitre-attack"), None)
        if not ext:
            continue
        if o.get("revoked") or o.get("x_mitre_deprecated"):
            dropped += 1
            continue
        valid.add(ext["external_id"])
        names[ext["external_id"]] = o.get("name")
    out = {"source": a.stix or a.url, "attack_version": "15.1", "n_valid": len(valid),
           "n_techniques": sum("." not in t for t in valid), "n_dropped_revoked_deprecated": dropped,
           "valid": sorted(valid), "names": dict(sorted(names.items()))}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("attack_version", "n_valid", "n_techniques", "n_dropped_revoked_deprecated")}))


if __name__ == "__main__":
    main()
