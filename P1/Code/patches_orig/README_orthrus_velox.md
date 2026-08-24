# Patches: ORTHRUS + Velox trên OpTC H051

Áp dụng trên máy L6 (thầy) — WSL Ubuntu user vung2.

## orthrus_opc_h051.patch — áp vào repo `ubc-provenance/orthrus` (clone ~/optec-l6/orthrus)

```bash
cd ~/optec-l6/orthrus
git apply patches/orthrus_opc_h051.patch
```

Nội dung các fix (theo thứ tự phát hiện):
1. `src/config.py` — thêm dataset `OPTC_H051` (DB optc_h051_full, Sept 2019, split train 19-21 / val 22 / test 25, GT node_h051_0925.csv 114 nodes, attack window 25/09 10:29-14:25); rel2id đổi sang OpTC ops (OPEN/READ/CREATE/MODIFY/MESSAGE/START/RENAME/DELETE/TERMINATE/WRITE); ROOT_GROUND_TRUTH_DIR bỏ `/darpa`; DB host localhost
2. `src/graph_construction/build_orthrus_graphs.py` — event_table query 8 cột explicit (OpTC có cột edge_label thừa làm unpack fail); nodeid2msg thêm key **node_uuid** (OpTC events tham chiếu UUID, không phải hash_id — root cause graphs rỗng)
3. `src/provnet_utils.py` — get_all_files_from_folders fallback lấy tất cả subdirs khi split list rỗng
4. `src/edge_featurization/embed_edges_feature_word2vec.py` — skip graph rỗng (tránh torch.vstack crash)
5. `src/data_utils.py` — RAM fix: val/test không load vào RAM; msg/t/edge_type của graphs re-point thành views của full_data
6. `src/detection/orthrus_gnn_testing.py` — stream val/test one-by-one từ disk; pad full_data zero-slots cho vùng test; max_node_num tính trên toàn bộ splits; resize neighbor_loader sau load_model (checkpoint build với train-only max_node)

## velox_worktree.patch — áp vào PIDSMaker branch `velox` @54f687c (worktree ~/optec-l6/velox)

```bash
cd ~/optec-l6/PIDSMaker
git worktree add ../velox 54f687c54aa
cd ../velox
git apply <path>/patches/velox_worktree.patch
```

Nội dung: DB name `optc_051` → `optc_h051_full`; host `postgres` → `localhost`; event query 8 cột explicit.
Chạy: `PYTHONHASHSEED=0 python src/benchmark.py velox optc_h051 --cpu --tuned --from_weights`

## Trạng thái đã verify (23/08)
- ORTHRUS pipeline chạy qua: graphs 469 TW (có edges), word2vec, embed_edges (train 230/val 94/test 56), GNN training epoch 1 loss 0.9233, testing 94+56 streams — toàn bộ trên 28GB RAM
- Kẹt cuối: neighbor_loader size mismatch sau load_model → patch resize đã viết trong patch file, chưa chạy lại (dừng theo yêu cầu)
- Full run (test 3 ngày) cần ≥40GB RAM → máy Nam 63.6GB
