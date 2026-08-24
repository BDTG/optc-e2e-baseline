"""refine_labels.py — Tinh chinh nhan tu (host,thoi gian) sang process lineage.

Van de: export_optc_ecar.py gan label=1 cho MOI process CREATE tren host bi tan cong
trong ca cua so thoi gian tan cong -> lan qua trinh nen vo hai xay ra trung gio
(SearchProtocolHost.exe, smartscreen.exe, svchost -k wsappx...) vao class malicious.

Sua: chi gan label=1 cho seed process khop signature tan cong (theo command_line) VA
toan bo cay con cua no (duyet cay qua (host,pid)->(host,ppid) bang stack/DFS, gioi han
trong window nhu cu de khong lan sang hoat dong khac ngay — thu tu duyet khong anh
huong ket qua vi malicious_keys la mot set, chi quan tam co-toi-duoc-hay-khong).

Seed signature (tu ground truth):
  Day1 "Plain PowerShell Empire" tren Sysclient0201/0205:
    - "runme.bat"                      : batch stager tai ve
    - "-enc"/"-EncodedCommand"         : base64-encoded PowerShell (Empire launcher).
                                          Khop moi dang viet tat PowerShell chap nhan
                                          (-e, -en, -enc, ... -EncodedCommand) vi Day1
                                          dung "-enc" nhung Day2 dung "-EncodedCommand"
                                          day du — text da duoc export_optc_ecar.py giai
                                          ma nen phan sau flag la script doc duoc, khong
                                          con la base64.
    - "-nop -sta -w 1" / "-noP -sta"   : Empire launcher flags dac trung
    - "mimikatz"                       : thu thap credential
    - "invoke-wmi" / "invoke_wmi"      : pivot lateral movement
  Day2 "Custom PowerShell Empire" tren Sysclient0501:
    - "plink"                          : upload plink.exe, reverse-ssh port-forward RDP
    - "filetransfer1000"               : nc.exe doi ten dung de exfil (dac thu campaign
                                          nay, khong tong quat hoa duoc sang campaign khac)

Chay:
  python refine_labels.py --in "E:\\dataset\\processed\\train_data_201.jsonl" --out "E:\\dataset\\processed\\train_data_201_refined.jsonl"
  python refine_labels.py --in "E:\\dataset\\processed\\train_data_501.jsonl" --out "E:\\dataset\\processed\\train_data_501_refined.jsonl"
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
from collections import defaultdict

from utils import PS_ENC_FLAG_PREFIX, read_jsonl

SEED_PATTERNS = [
    re.compile(r"runme\.bat", re.IGNORECASE),
    # -e/-en/-enc/.../-EncodedCommand (moi do dai viet tat PowerShell chap nhan).
    # Chi can biet flag co mat, KHONG doi base64 phia sau nhu export_optc_ecar.py —
    # luc nay text da duoc giai ma roi (script doc duoc, khong con la base64).
    re.compile(PS_ENC_FLAG_PREFIX + r"\s", re.IGNORECASE),
    re.compile(r"-no[pP]\s+-sta\s+-w\s+1", re.IGNORECASE),
    re.compile(r"mimikatz", re.IGNORECASE),
    re.compile(r"invoke[-_]wmi", re.IGNORECASE),
    re.compile(r"plink", re.IGNORECASE),
    re.compile(r"filetransfer1000", re.IGNORECASE),
]

MAX_LINEAGE_SPAN_SEC = 5 * 3600  # khong lan qua 5h tu seed (window ground truth ~4h)


def is_seed(text: str) -> bool:
    return any(p.search(text) for p in SEED_PATTERNS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.inp, progress_every=200_000, tag="refine")

    def key_of(r: dict):
        """(host, pid, ts) — dinh danh 1 INSTANCE process CREATE cu the, de phan
        biet cac lan tai su dung cung so PID."""
        m = r["meta"]
        return (m["host"], m["pid"], m["ts"])

    n_was_mal = sum(r["label"] for r in rows)

    # chi xet lineage trong pham vi cac host TUNG duoc gan malicious (host khac giu nguyen label=0)
    hosts_with_mal = {r["meta"]["host"] for r in rows if r["label"] == 1}
    print(f"[refine] host co nhan malicious cu: {sorted(hosts_with_mal)}")

    # index theo (host,pid) -> list CREATE cua chinh pid do, sap theo ts (de tra ve dung
    # INSTANCE con tro toi, vi PID bi Windows tai su dung lien tuc trong 1 ngay)
    by_host_pid = defaultdict(list)
    # index theo (host,ppid) -> cac row co the la con (ung vien), tra cuu nhanh thay vi quet toan bo rows
    by_host_ppid = defaultdict(list)
    for r in rows:
        m = r["meta"]
        by_host_pid[(m["host"], m["pid"])].append(r)
        by_host_ppid[(m["host"], m["ppid"])].append(r)
    for v in by_host_pid.values():
        v.sort(key=lambda r: r["meta"]["ts"])
    # ts rieng, song song voi by_host_pid (da sap xep) -> tra cuu nearest_instance
    # bang bisect O(log m) thay vi quet+max O(m) tren moi canh BFS/DFS.
    by_host_pid_ts = {k: [x["meta"]["ts"] for x in v] for k, v in by_host_pid.items()}

    seeds = [r for r in rows if r["meta"]["host"] in hosts_with_mal and is_seed(r["text"])]
    print(f"[refine] tim thay {len(seeds)} seed process (khop signature tan cong)")
    for s in seeds:
        m = s["meta"]
        print(f"  seed: host={m['host']} pid={m['pid']} ts={m['ts']:.0f} text={s['text'][:90]}")

    malicious_keys = set()  # (host, pid, ts) de phan biet nhieu lan CREATE cung pid (PID reuse)
    stack = list(seeds)
    for s in seeds:
        malicious_keys.add(key_of(s))

    def nearest_instance(host: str, pid, at_ts: float):
        """Instance CREATE gan nhat cua (host,pid) truoc/bang at_ts. None neu tien trinh
        da ton tai tu truoc pham vi du lieu (khong bat duoc CREATE)."""
        key = (host, pid)
        insts = by_host_pid.get(key)
        if not insts:
            return None
        idx = bisect.bisect_right(by_host_pid_ts[key], at_ts) - 1
        return insts[idx] if idx >= 0 else None

    while stack:
        cur = stack.pop()
        host, pid, ts = key_of(cur)
        # tim con: row co ppid == pid CUNG host, ts trong pham vi cua so
        for r in by_host_ppid.get((host, pid), []):
            r_ts = r["meta"]["ts"]
            if r_ts < ts or r_ts - ts > MAX_LINEAGE_SPAN_SEC:
                continue
            # BAT BUOC: PID bi tai su dung -> phai xac nhan r thuc su la con cua DUNG
            # instance "cur" (instance CREATE cua pid gan r nhat), khong phai mot tien
            # trinh benign khac tai su dung cung so PID sau khi "cur" da thoat.
            parent_inst = nearest_instance(host, pid, r_ts)
            if parent_inst is None or parent_inst["meta"]["ts"] != ts:
                continue
            key = key_of(r)
            if key not in malicious_keys:
                malicious_keys.add(key)
                stack.append(r)

    n_new_mal = 0
    for r in rows:
        r["label"] = 1 if key_of(r) in malicious_keys else 0
        n_new_mal += r["label"]

    with open(args.out, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")

    print(f"[refine] malicious: cu={n_was_mal:,} -> moi={n_new_mal:,} "
          f"(giam {100*(1-n_new_mal/max(n_was_mal,1)):.1f}%)")
    print(f"[out] -> {args.out}")


if __name__ == "__main__":
    main()
