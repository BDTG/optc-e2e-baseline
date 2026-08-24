"""export_optc_ecar.py — Loc PROCESS/CREATE tu file eCAR NDJSON (OpTC) -> JSONL
{text, raw_text, label, meta:{host,ts,pid,ppid,size,ps_decoded,ps_truncated}}.
`text`/`raw_text`/`label` la feature+target dua thang vao model; moi thu con lai
nam trong `meta` (khong dua vao model, chi dung de split/truy vet/debug).

Khac export_dataset.py (Sysmon/Elastic field map + tech_id): eCAR khong co rule_technique_id,
nen nhan malicious duoc gan truc tiep tu cua so thoi gian (host, start, end) doc tu
OpTCRedTeamGroundTruth.pdf. Xem ATTACK_WINDOWS ben duoi — chi khai bao cho cac host CO trong
bundle dang xu ly; host khac trong cung dai (vd 202-225 tru 201/205) mac dinh la benign.

errata.md: "Duplicate Process Objects" — nhieu process source khong de-conflict, actorID co
the sai. Script nay khong co gang sua loi do (out of scope Phase 0), chi dem so record
thieu parent_image_path va bao cao trong dong [done] cuoi cung (khong in tung dong vi
truong hop nay khong hiem — xem thieu_parent_image_path trong output).

Chay (thu tu --data phai theo dung thu tu thoi gian: chunk som truoc, chunk cuoi sau):
  python export_optc_ecar.py ^
      --data "E:\\dataset\\raw\\AIA-201-225\\AIA-201-225.ecar-2019-12-08T11-05-10.046.json" ^
      --data "E:\\dataset\\raw\\AIA-201-225\\AIA-201-225.ecar.json" ^
      --out "E:\\dataset\\processed\\train_data_201.jsonl"

  python export_optc_ecar.py ^
      --data "E:\\dataset\\raw\\AIA-501-525\\AIA-501-525.ecar-2019-11-17T04-01-58.625.json.gz" ^
      --data "E:\\dataset\\raw\\AIA-501-525\\AIA-501-525.ecar.json" ^
      --out "E:\\dataset\\processed\\train_data_501.jsonl"
"""
from __future__ import annotations

import argparse
import base64
import gzip
import ipaddress
import json
import re
import time
from datetime import datetime
from typing import List, Optional, Tuple

from utils import PS_ENC_FLAG_PREFIX, basename, mask_ips, mask_usernames

# Dai IP noi bo cua testbed OpTC (142.20.0.0/16 — khong thuoc RFC1918 chuan nen
# classify_ip() khong tu nhan ra, phai khai bao rieng o day thay vi hardcode trong
# parser/utils.py, vi day la kien thuc campaign-cu-the, khong phai co che chung).
_OPTC_PRIVATE_NETS = [ipaddress.ip_network("142.20.0.0/16")]

# (hostname eCAR, start ISO, end ISO) — tu OpTCRedTeamGroundTruth.pdf, gio -04:00 (EDT) nhu trong data.
# Window la (host, thoi gian) tho: gan label=1 cho MOI process tren host do trong window,
# se qua-gan-nhan nhieu process nen benign chay trung gio — refine_labels.py thu hep lai
# theo lineage tu seed process that. Diem cuoi window = hanh dong cuoi ground truth ghi
# nhan TREN CHINH host do (khong tinh hoat dong sau khi attacker da RDP/pivot sang host khac).
ATTACK_WINDOWS: List[Tuple[str, str, str]] = [
    # Day1 "Plain PowerShell Empire" (23Sep19-red), bundle AIA-201-225
    ("SysClient0201.systemia.com", "2019-09-23T11:23:29-04:00", "2019-09-23T15:30:00-04:00"),
    ("SysClient0205.systemia.com", "2019-09-23T14:45:13-04:00", "2019-09-23T15:24:33-04:00"),
    # Day1 WMI-spread tu DC1 toi SYSCLIENT0503 (14:45:13 spread -> 15:24:33 kill), can
    # file eCAR moi (AIA-501-525.ecar-day1.json) vi ban goc bundle 501-525 khong co Day1.
    ("SysClient0503.systemia.com", "2019-09-23T14:45:13-04:00", "2019-09-23T15:24:33-04:00"),
    # Day2 "Custom PowerShell Empire" (24Sep19), bundle AIA-501-525
    ("SysClient0501.systemia.com", "2019-09-24T10:36:51-04:00", "2019-09-24T15:28:36-04:00"),
    # Day2 WMI-spread tu DC1 toi Sysclient0203 (15:42:36 spread -> qua dem den ~09:05
    # 25/09 khi agent overnight con hoat dong), can file eCAR moi
    # (AIA-201-225.ecar-day2.json) vi ban goc bundle 201-225 khong co Day2.
    ("SysClient0203.systemia.com", "2019-09-24T15:42:36-04:00", "2019-09-25T09:05:00-04:00"),
]


def _parse_ecar_ts(s: str) -> float:
    # vd "2019-09-23T11:23:29.127-04:00" hoac khong co phan .mmm
    return datetime.fromisoformat(s).timestamp()


_WINDOWS = [(h, _parse_ecar_ts(a), _parse_ecar_ts(b)) for h, a, b in ATTACK_WINDOWS]


def is_malicious(hostname: str, ts_epoch: float) -> int:
    for h, a, b in _WINDOWS:
        if hostname == h and a <= ts_epoch <= b:
            return 1
    return 0


# PowerShell -EncodedCommand/-enc/-e: base64 cua script UTF-16LE. Day la obfuscation
# CHUAN, khong phai noise ngau nhien -> giai ma o tien xu ly de model thay noi dung
# script that thay vi mot khoi base64 vo nghia (dung ky thuat cac cong cu phong thu
# that nhu PowerShell ScriptBlock Logging deu lam).
_ENC_FLAG_RE = re.compile(
    PS_ENC_FLAG_PREFIX + r"\s+([A-Za-z0-9+/]{20,}={0,2})",
    re.IGNORECASE,
)


def decode_ps_encoded(cmd: str) -> Tuple[str, bool, bool]:
    """Tim '-enc <base64>' trong command line, giai ma UTF-16LE.

    eCAR cat cung command_line o ~547 ky tu (gioi han buffer cua collector) -> payload
    Empire (thuong 800-2000+ ky tu) hau nhu luon bi cat giua chung, khong the giai ma
    tron ven. Thay vi bo cuoc, giai ma PHAN CON NGUYEN VEN cua base64 (cat ve boi so
    cua 4 ky tu, bo byte UTF-16 le do o cuoi) -> van lay duoc phan dau script (thuong
    la doan AMSI-bypass boilerplate) thay vi de nguyen mot khoi base64 vo nghia.

    Tra ve (text da thay base64 bang script/script-cut, co giai ma duoc mot phan nao
    khong, co bi cat (truncated) khong).
    """
    m = _ENC_FLAG_RE.search(cmd)
    if not m:
        return cmd, False, False
    token = m.group(1)
    truncated = m.end(1) == len(cmd)  # base64 chay den het chuoi -> nghi bi eCAR cat cung
    usable = token[: len(token) - len(token) % 4]  # bo duoi le khong du 1 nhom base64
    if len(usable) < 8:
        return cmd, False, truncated
    try:
        raw = base64.b64decode(usable, validate=True)
        raw = raw[: len(raw) - len(raw) % 2]  # bo byte le cuoi (utf-16le can so chan byte)
        script = raw.decode("utf-16-le", errors="ignore")
    except Exception:
        return cmd, False, truncated
    script = script.strip()
    if not script:
        return cmd, False, truncated
    script = " ".join(script.split())  # gon 1 dong, bo newline/tab thua
    if truncated and len(usable) < len(token):
        script += " [...CAT BOI eCAR...]"
    replaced = cmd[:m.start(1)] + script + cmd[m.end(1):]
    return replaced, True, truncated


def build_text(props: dict) -> Tuple[str, str, bool, bool]:
    """Tra ve (text hien thi cho model [da giai ma + mask IP/username], text goc
    chua giai ma/chua mask, co_giai_ma_duoc_khong, co_bi_eCAR_cat_command_line_khong).

    `text` mask dia chi IP (-> <IP_LOCAL/PRIVATE/PUBLIC>) va username trong duong dan
    (\\Users\\<ten>\\ -> \\Users\\<USER>\\) — day la dac thu 1 moi truong cu the (mang
    testbed, tai khoan may nay), khong phai tin hieu hanh vi, de model khong hoc tat
    theo dia chi/user cua rieng OpTC. `raw_text` GIU NGUYEN khong mask, de doi chieu/debug."""
    parent = basename(props.get("parent_image_path"))
    image = basename(props.get("image_path"))
    cmd_raw = str(props.get("command_line") or "").strip()
    cmd_decoded, was_encoded, was_truncated = decode_ps_encoded(cmd_raw)
    cmd_masked = mask_usernames(mask_ips(cmd_decoded, _OPTC_PRIVATE_NETS))
    text = f"{parent} -> {image} | {cmd_masked}".strip()
    text_raw = f"{parent} -> {image} | {cmd_raw}".strip()
    return text, text_raw, was_encoded, was_truncated


def iter_ndjson(path: str, stats: Optional[dict] = None):
    """stats (neu co): dict de cong don n_json_err khi mot dong khong parse duoc,
    de main() bao cao trong tong ket thay vi lang le bo qua."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if stats is not None:
                    stats["json_err"] = stats.get("json_err", 0) + 1
                continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True,
                     help="duong dan file eCAR NDJSON, lap lai --data cho nhieu file (theo dung thu tu thoi gian)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="gioi han so dong doc moi file (debug)")
    args = ap.parse_args()

    n_read = n_kept = n_mal = 0
    no_parent = n_decoded = 0
    stats = {"json_err": 0, "bad_ts": 0}
    t_start = time.time()

    with open(args.out, "w", encoding="utf-8") as fo:
        for path in args.data:
            print(f"[read] {path}", flush=True)
            for i, rec in enumerate(iter_ndjson(path, stats)):
                n_read += 1
                if args.limit and i >= args.limit:
                    break
                if n_read % 5_000_000 == 0:
                    elapsed = time.time() - t_start
                    print(f"[progress] doc={n_read:,} giu={n_kept:,} malicious={n_mal:,} "
                          f"({n_read/max(elapsed,1e-9):,.0f} dong/s, {elapsed:.0f}s)", flush=True)

                if rec.get("action") != "CREATE" or rec.get("object") != "PROCESS":
                    continue

                props = rec.get("properties") or {}
                if not (props.get("parent_image_path") or props.get("image_path")
                        or props.get("command_line")):
                    continue  # khong co gi de tao text -> bo qua, khoi can strip() do sau
                if not props.get("parent_image_path"):
                    no_parent += 1

                host = rec.get("hostname", "")
                ts_raw = rec.get("timestamp")
                try:
                    ts_epoch = _parse_ecar_ts(ts_raw)
                except Exception:
                    stats["bad_ts"] += 1
                    continue

                label = is_malicious(host, ts_epoch)
                text, text_raw, was_encoded, was_truncated = build_text(props)

                row = {
                    "text": text,
                    "raw_text": text_raw,
                    "label": label,
                    "meta": {
                        "host": host,
                        "ts": ts_epoch,
                        "pid": rec.get("pid"),
                        "ppid": rec.get("ppid"),
                        "size": 1,
                        "ps_decoded": was_encoded,
                        "ps_truncated": was_truncated,
                    },
                }
                fo.write(json.dumps(row) + "\n")
                n_kept += 1
                n_mal += label
                n_decoded += was_encoded

    print(f"[done] doc={n_read:,} giu(PROCESS/CREATE)={n_kept:,} malicious={n_mal:,} "
          f"({100*n_mal/max(n_kept,1):.3f}%) thieu_parent_image_path={no_parent:,} "
          f"ps_decoded(mot phan hoac tron ven)={n_decoded:,} "
          f"json_err={stats['json_err']:,} bad_ts={stats['bad_ts']:,} "
          f"({time.time()-t_start:.0f}s)")
    print(f"[out] -> {args.out}")


if __name__ == "__main__":
    main()
