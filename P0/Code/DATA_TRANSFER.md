# DATA_TRANSFER — file lớn KHÔNG trong GitHub (giới hạn 100MB)

File ≥100MB phải chuyển bằng đường khác. Đề xuất theo thứ tự ưu tiên:

## Cách 1 (khuyến nghị nhất) — Ổ cứng ngoài/USB giữa 2 máy
Nếu 2 máy gặp nhau vật lý hoặc có chung mạng LAN:
- `D:\OpTC-raw\25Sept\` trên máy BDTG → gồm `AIA-51-75` (1.34GB) + `AIA-351-375` (1.5GB) + `HUONG_DAN_MO_RONG_DAY3.md`
- `C:\Users\BDTG\OpTC-data\phase1\handoff_remote.zip` (1.63GB) → gồm 2 dump + patches + config + scripts + runbook

## Cách 2 — Google Drive
Tải từ `https://drive.google.com/drive/folders/1n3kkS3KR31KUegn42yk3-e6JkZvf0Caa` (ecar/evaluation/25Sept)
với rclone (máy BDTG có cấu hình sẵn):
```
rclone --config D:/Tools/rclone-v1.75.0-windows-amd64/rclone.conf copy mydrivename1:ecar/evaluation/25Sept/AIA-351-375/ <dest> -P
# AIA-51-75 tương tự (máy BDTG cũng có sẵn tại phase1/data/raw/host051_25Sept/)
```

## Tóm tắt các file lớn

| File | Size | Vị trí nguồn (máy BDTG) | Cách chuyển |
|---|---|---|---|
| `optc_h051_full.dump` | 1.13 GB | `phase1/handoff_remote/dumps/` | USB / zip hiện có / LAN |
| `optc_051_sub.dump` | 504 MB | `phase1/handoff_remote/dumps/` | USB / zip |
| `AIA-51-75.ecar-last.json.gz` | 1.34 GB | `phase1/data/raw/host051_25Sept/` | USB / Drive |
| `AIA-351-375.ecar-last.json.gz` | 1.50 GB | `D:\OpTC-raw\25Sept\AIA-351-375/` | USB / Drive |

## Checksums (verify sau khi chuyển)
```
1a76c0ff605497df11b63b2d38b8dea3ac1962a4244c4984b06939b56ac3a302  optc_051_sub.dump
93f9a65cb07f4417f66b4c8d21b4267b7b54155b1c4902c86a8692b2aef944cd  optc_h051_full.dump
6afd2056353f825efe9c083a887781068fe09dd134715dc34700353f21efd0b0  AIA-51-75.ecar-last.json.gz
d52c3fc3439de53123fe199374f08a7d7b2af8e9358727bf88b2a5325a32cc75  AIA-351-375.ecar-last.json.gz
```
