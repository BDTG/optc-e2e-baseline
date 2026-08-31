# YÊU CẦU CẢI THIỆN — Re-run Phase 0 (chuẩn hóa protocol)

> Gửi kèm gói handoff. 5 điểm bắt buộc để số liệu Phase 0 dùng được làm bằng chứng.
> Lý do từng điểm nêu ngắn — đã phân tích từ metrics.json hiện tại (15 runs).

---

## 1. GHI RÕ split_frac KHÔNG ảnh hưởng trong host_holdout (đã xác nhận bằng code gốc)

**Đã nhận `load_split()` gốc** — kết luận:

```python
if cfg.split_mode == "host_holdout":
    train = [r for r in rows if r["meta"].get("host") not in hold]  # TOÀN BỘ non-holdout
    test  = [r for r in rows if r["meta"].get("host") in hold]
    # split_frac KHÔNG được dùng ở đây
else:
    cut = int(len(rows) * cfg.split_frac)  # chỉ causal mới dùng split_frac
```

→ `split_frac` 0.6/0.7 trong metrics.json là **field còn sót, không tác động cỡ train**.
→ KHÔNG cần đồng bộ split_frac. Cả hai model đã train trên toàn bộ 10,302 dòng non-holdout.

**Yêu cầu thực sự:**
- Ghi `epochs` chính xác của Qwen (hiện None) — đồng bộ số epoch (khuyến nghị 3 như ModernBERT) để công bằng.
- Nếu muốn test sensitivity theo cỡ train, hãy CHỦ ĐỘNG đưa split_frac vào nhánh host_holdout (hiện không có) — nếu không thì bỏ field này khỏi metrics.json để tránh hiểu nhầm.

## 2. GHI RÕ SỐ EPOCH QWEN ĐÃ CHẠY

- Trường `epochs` của Qwen trong metrics.json hiện **None** — thiếu.
- Bắt buộc ghi: epochs, batch_size, gradient accumulation, total steps, patience/early-stop (nếu có).
- Chèn vào metrics.json mọi run mới.

## 3. NHIỀU SEED + NHIỀU HOST HOLDOUT → BÁO CÁO mean ± std

**Hiện trạng:** mọi run chỉ 1 seed (42) + 1 host holdout (SysClient0501).
**QUAN TRỌNG (từ code gốc):** undersample dùng `random.Random(42)` CỐ ĐỊNH và host_holdout
không shuffle train → **đổi seed cấu hình KHÔNG thay đổi train/test hiện tại**.

**Yêu cầu (bắt buộc sửa code trước khi chạy):**
1. `load_split`: đổi `neg = random.Random(42).sample(...)` → `random.Random(cfg.seed).sample(...)`
   (và nếu muốn) thêm shuffle train pool theo `cfg.seed`.
2. Chạy **≥3 seed** (42, 123, 2026) mỗi cấu hình.
3. **≥2 host holdout** — thêm host benign thuần (VD `SysClient0205` — 25 dòng, 1 mal trong
   train) để đo FP + unseen-host.
4. Báo cáo **mean ± std** của AUC-PR, recall@FPR, MCC, F1.
5. Lý do: test chỉ 18 mẫu mal → ±5.6%/mẫu; hiện seed vô hiệu nên số liệu chưa đủ vững.

## 4. METRIC CHÍNH = AUC-PR (hoặc F1/MCC TẠI THRESHOLD CALIBRATED), KHÔNG PHẢI AUC

**Hiện trạng:** bảng chính thức đặt AUC cạnh AUC-PR, và cột quyết định hay bị đọc theo AUC (Qwen thắng AUC 0.964 → dễ hiểu nhầm).

**Yêu cầu:**
- **AUC-PR là metric quyết định** (mất cân bằng 2.64% — AUC-ROC bơm số ở FPR thấp, AUC-PR nhạy đúng với lớp hiếm).
- Bổ sung **F1 và MCC tại threshold đã calibrate** (không dùng ngưỡng 0.5 vốn tùy ý; calibrate trên validation/hoặc theo recall mục tiêu của SOC).
- Cột AUC chuyển xuống là metric tham khảo, đánh dấu "không dùng để quyết định".
- Điều này KHỚP với protocol của đề tài (AUC-PR chính, không AUC-ROC) — đã ghi trong `DE_CUONG.md` / `proposal/03_rq_metrics.md`.

## 5. GHI PEAK VRAM CHO MODERNBERT

**Hiện trạng:** metrics.json ModernBERT **không có `peak_mem_gb`** (chỉ Qwen có: 8.667GB).

**Yêu cầu:**
- Ghi `peak_mem_gb` cho ModernBERT (dùng `torch.cuda.max_memory_allocated()` sau train).
- Bổ sung cả `latency_ms` ghi rõ device (GPU/CPU) cho từng model.
- Lý do: bảng chi phí cần đủ 3 trục — **thời gian train / VRAM / latency** — để so sánh "chi phí hệ thống" (RQ4a) và justify kiến trúc tầng hai.

---

## Tóm tắt thay đổi bắt buộc (checklist cho re-run)

- [ ] Sửa `load_split`: `random.Random(42)` → `random.Random(cfg.seed)` (+ shuffle train theo seed nếu cần)
- [ ] Ghi `epochs` (đồng bộ 3 như ModernBERT), `batch_size`, `peak_mem_gb` (cả 2 model), `latency_device`
- [ ] ≥3 seed × ≥2 host holdout → báo cáo mean ± std
- [ ] Bảng kết quả: AUC-PR cột chính; thêm F1/MCC@calibrated threshold; AUC chuyển tham khảo
- [ ] (Không cần đụng split_frac — đã chứng minh vô hiệu trong host_holdout; hoặc bỏ field khỏi metrics.json)
- [ ] Gửi về: metrics.json bản mới + bảng tổng hợp mean±std + script đo VRAM

> Kết quả này thay thế số Phase 0 hiện tại trong `baseline_comparison.md` §3.4 —
> máy BDTG sẽ cập nhật khi nhận được.
