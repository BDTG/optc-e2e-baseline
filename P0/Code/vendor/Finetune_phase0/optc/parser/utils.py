"""utils.py — Helper dung chung cho cac script trong parser/ (export_optc_ecar.py,
export_dataset.py, refine_labels.py). KHONG import config/data/model/metrics/utils
o optc/ goc — parser/ tu than, doc lap voi phan train/baseline (xem optc/README.md).
"""
from __future__ import annotations

import ipaddress
import json
import re
from typing import Iterable, List, Optional


def basename(p: Optional[str]) -> str:
    """Rut gon duong dan Windows (hoac chuoi co dau /) ve ten file, chu thuong."""
    if not p:
        return ""
    return str(p).strip().strip('"').replace("/", "\\").split("\\")[-1].lower()


# Tien to regex cho moi do dai viet tat PowerShell chap nhan cua -EncodedCommand:
# -e, -en, -enc, -enco, ... -EncodedCommand. Dung chung boi export_optc_ecar.py (tim
# + giai ma base64 phia sau) va refine_labels.py (chi can biet co mat flag nay hay
# khong, TREN TEXT DA GIAI MA nen khong the dung chung 1 regex hoan chinh — xem
# ghi chu trong tung file goi).
PS_ENC_FLAG_PREFIX = (
    r"-e(?:n(?:c(?:o(?:d(?:e(?:d(?:c(?:o(?:m(?:m(?:a(?:n(?:d)?)?)?)?)?)?)?)?)?)?)?)?)?"
)


def classify_ip(ip_str: str, extra_private_nets: Iterable = ()) -> str:
    """LOCAL (loopback) | PRIVATE (RFC1918 hoac trong extra_private_nets) | PUBLIC.

    extra_private_nets: cac ipaddress.ip_network BO SUNG rieng cho 1 campaign/testbed cu
    the (vd dai IP noi bo cua 1 mang khong thuoc RFC1918 chuan) — CO Y de trong mac dinh,
    vi day la co che chung, kien thuc campaign-cu-the phai truyen vao tu noi goi, khong
    hardcode o day (xem _OPTC_PRIVATE_NETS trong export_optc_ecar.py)."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return "PUBLIC"  # khong parse duoc (hiem) -> coi nhu cong khai, an toan hon la giau
    if ip.is_loopback:
        return "LOCAL"
    if ip.is_private or any(ip in net for net in extra_private_nets):
        return "PRIVATE"
    return "PUBLIC"


_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def mask_ips(text: str, extra_private_nets: Iterable = ()) -> str:
    """Thay dia chi IP cu the bang the loai <IP_LOCAL>/<IP_PRIVATE>/<IP_PUBLIC>.

    Ly do: IP cu the (vd dai noi bo cua 1 testbed) la dac thu 1 moi truong — model hoc
    'nhac toi IP nay -> chac chan benign' la overfitting vao dung mang nay, khong tong
    quat hoa duoc sang moi truong trien khai khac (dung RQ5a/RQ5b lo ngai). Van giu LOAI
    dia chi (noi bo/cong khai/localhost) vi day la tin hieu hanh vi that (lateral movement
    thuong nham dia chi noi bo, C2 thuong goi ra dia chi cong khai)."""
    return _IP_RE.sub(lambda m: f"<IP_{classify_ip(m.group(0), extra_private_nets)}>", text)


_USER_PATH_RE = re.compile(r"(Users\\)[A-Za-z0-9_.\-]+(\\)", re.IGNORECASE)


def mask_usernames(text: str) -> str:
    """Thay ten user trong duong dan \\Users\\<ten>\\ bang <USER> — cung ly do voi
    mask_ips: ten user cu the la dac thu 1 moi truong/tai khoan, khong phai tin hieu
    hanh vi (vd 'sysadmin' xuat hien o ca benign LAN malicious trong dataset nay —
    context cua lenh moi la thu phan biet duoc, khong phai ten user)."""
    return _USER_PATH_RE.sub(r"\1<USER>\2", text)


def read_jsonl(path: str, progress_every: Optional[int] = None, tag: str = "read") -> List[dict]:
    """Doc toan bo JSONL vao list. progress_every: in tien do moi N dong (None = im lang)."""
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if progress_every and i % progress_every == 0:
                print(f"[{tag}] doc {i:,} dong...", flush=True)
    return rows
