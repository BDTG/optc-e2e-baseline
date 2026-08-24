"""make_final_test.py — Trich mot lat cat NHO, CO DINH (seed co dinh, stratify theo
label) tu host holdout hien tai, dong bang lam final test — bao ve tinh toan ven danh gia.

Chay SAU dedup_cross_host.py — dataset dau vao da duoc dam bao KHONG con text nao trung
lap giua 2 host (xem dedup_cross_host.py), nen khong can loc rieng o day: bat ky dong nao
cua host holdout cung an toan, khong con la "da thay luc train" nua.

Van de: moi lan chay truoc gio (TF-IDF, ModernBERT, SLM...) deu danh gia tren TOAN BO
host holdout (SysClient0501). Cac quyet dinh cau hinh (neg-ratio, batch-size, mask-token)
it nhieu bi anh huong boi chi so quan sat duoc tren chinh host do -> host nay thuc chat
da tro thanh mot "dev set" ngam thay vi test set thuan tuy.

Tu sau khi chay script nay:
  - final_test.jsonl (lat cat NHO, CO DINH): CHI danh gia 1 LAN DUY NHAT, luc bao cao
    ket qua cuoi cung. KHONG dung de tune bat ky tham so nao (neg-ratio, lora-r,
    batch-size, threshold, ...).
  - Phan CON LAI cua host holdout (da bot final_test) tiep tuc dung binh thuong lam
    dev-test cho qua trinh tuning nhu truoc gio.

data.py::load_split() tu dong loai bo cac dong trung khoa (host,pid,ppid,ts) voi
final_test.jsonl khoi BAT KY tap train/dev nao duoc load sau nay, bat ke dataset_path
la file nao -> khong the vo tinh lam ro ri final_test vao train/dev du quen flag.

Luu y: vi cac lan chay TRUOC DAY da danh gia tren toan bo host nay (bao gom ca cac dong
se bi trich ra lam final_test), aggregate metric cua cac lan do da "nhin thay" mot phan
thong tin cua final_test. Tu day tro di final_test duoc bao ve hoan toan; day la buoc
thuc te tot nhat co the lam voi du lieu hien co (chi 3/47 host co nhan malicious).

Chay (chi can 1 LAN — ket qua co dinh nho seed):
  python make_final_test.py --data E:\\dataset\\processed\\train_data_combined.jsonl \
      --host SysClient0501.systemia.com --frac 0.2
"""
from __future__ import annotations

import argparse
import json
import random

from run_utils import read_jsonl


def row_key(r: dict):
    m = r["meta"]
    return (m.get("host"), m.get("pid"), m.get("ppid"), m.get("ts"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--host", required=True,
                     help="host holdout hien tai (nguon de trich final test)")
    ap.add_argument("--frac", type=float, default=0.2,
                     help="ty le (theo tung lop) trich lam final test, con lai giu dev-test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"E:\dataset\processed\final_test.jsonl")
    args = ap.parse_args()

    rows = read_jsonl(args.data)
    host_rows = [r for r in rows if r["meta"].get("host") == args.host]
    if not host_rows:
        raise SystemExit(f"[loi] khong tim thay dong nao cho host '{args.host}'")

    pos = [r for r in host_rows if r["label"] == 1]
    neg = [r for r in host_rows if r["label"] == 0]
    rnd = random.Random(args.seed)
    n_pos_final = max(1, round(len(pos) * args.frac))
    n_neg_final = max(1, round(len(neg) * args.frac))
    final_rows = rnd.sample(pos, n_pos_final) + rnd.sample(neg, n_neg_final)
    final_rows.sort(key=lambda r: r["meta"]["ts"])

    with open(args.out, "w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[final-test] host={args.host}: total pos={len(pos)} neg={len(neg)}")
    print(f"[final-test] dong bang {n_pos_final} malicious + {n_neg_final} benign "
          f"(seed={args.seed}, frac={args.frac}) -> {args.out}")
    print(f"[final-test] phan con lai cua {args.host} "
          f"({len(pos) - n_pos_final} malicious + {len(neg) - n_neg_final} benign) "
          f"van dung binh thuong lam dev-test qua data.py::load_split() (tu dong loai "
          f"final_test.jsonl neu file ton tai).")
    print("[final-test] CHI chay danh gia tren file nay 1 LAN DUY NHAT luc bao cao ket "
          "qua cuoi cung. KHONG dung de tune tham so.")


if __name__ == "__main__":
    main()
