"""dedup_cross_host.py — Buoc 3 trong pipeline lam sach dataset: loai trung lap GIUA
CAC HOST (khong dung vao trung lap TRONG CUNG 1 host — do la tan suat that, giu nguyen).

Van de: nhieu process he thong/agent giam sat (vd lwabeat.exe) sinh ra command line
GIONG HET NHAU tren MOI host trong testbed (47 host dung chung baseline phan mem). Dieu
nay khien mot dong text co the xuat hien tren ca host TRAIN lan host TEST -> host-holdout
split khong con do dung KHA NANG TONG QUAT HOA sang host la, ma do KHA NANG GHI NHO
chuoi da thay luc train (da kiem chung: 90.7% dong benign trong test trung y het voi
train, 35% dong malicious cung vay).

Quy tac: voi moi text xuat hien o >=2 host, CHI GIU LAI cac dong cua host xuat hien SOM
NHAT (theo ts nho nhat trong nhom) — vi day la "nguon goc" that su cua hanh vi do; cac
host khac lap lai y het duoc coi la du thua. Text chi xuat hien o 1 host duy nhat khong
bi dong den (giu nguyen toan bo, ke ca lap lai nhieu lan trong host do).

Chay:
  python dedup_cross_host.py --data E:\\dataset\\processed\\train_data_combined.jsonl \
      --out E:\\dataset\\processed\\train_data_deduped.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json

from run_utils import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.data)
    groups: dict = collections.defaultdict(list)
    for r in rows:
        groups[r["text"]].append(r)

    kept: list = []
    removed_by_label = {0: 0, 1: 0}
    cross_host_groups = 0
    hosts_affected = set()

    for text, grp in groups.items():
        hosts_in_group = set(r["meta"]["host"] for r in grp)
        if len(hosts_in_group) <= 1:
            kept.extend(grp)  # trung lap TRONG 1 host -> giu nguyen, khong dong den
            continue
        cross_host_groups += 1
        host_first_ts: dict = {}
        for r in grp:
            h, ts = r["meta"]["host"], r["meta"]["ts"]
            if h not in host_first_ts or ts < host_first_ts[h]:
                host_first_ts[h] = ts
        keep_host = min(host_first_ts, key=host_first_ts.get)
        for r in grp:
            if r["meta"]["host"] == keep_host:
                kept.append(r)
            else:
                removed_by_label[r["label"]] += 1
                hosts_affected.add(r["meta"]["host"])

    kept.sort(key=lambda r: r["meta"]["ts"])
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_removed = len(rows) - len(kept)
    print(f"[dedup] vao: {len(rows):,} dong ({len(groups):,} text duy nhat, "
          f"{cross_host_groups:,} nhom xuat hien o >=2 host)")
    print(f"[dedup] loai {n_removed:,} dong trung lap GIUA CAC HOST "
          f"(benign={removed_by_label[0]:,} malicious={removed_by_label[1]:,}), "
          f"{len(hosts_affected)} host bi rut bot")
    print(f"[dedup] con lai: {len(kept):,} dong -> {args.out}")
    n_pos = sum(r["label"] for r in kept)
    print(f"[dedup] ty le malicious sau dedup: {n_pos:,}/{len(kept):,} "
          f"({100*n_pos/len(kept):.2f}%)")


if __name__ == "__main__":
    main()
