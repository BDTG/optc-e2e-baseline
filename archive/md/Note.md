# SLM ≤8B cho phát hiện bất thường endpoint trên CPU: kế hoạch nghiên cứu

## Định vị luận điểm

**Trước (rủi ro cao):** SLM ≤8B cạnh tranh được với GNN provenance ở mức phát hiện, với chi phí thấp hơn.

**Sau (đứng được):** SLM ≤8B không phải primary detector. Nó là **tầng hai trên ~10²–10³ candidate/host/ngày**, và nó mua được hai thứ mà PIDS hiện tại không có: (a) loại bỏ false positive nhờ hiểu ngữ nghĩa tổ hợp của command-line, (b) phát hiện TTP chưa từng thấy mà không cần nhãn. Bài toán là xác định cấu hình nào trên Pareto frontier và với ngân sách nào.

Lý do đổi: Velox đạt năng lực phát hiện ngang ORTHRUS với chi phí thấp hơn nhiều khi **chỉ dùng thuộc tính văn bản** của system entity. Tiền đề "GNN vứt bỏ command-line" không còn đúng. Claim còn lại phải là về **ngữ nghĩa tổ hợp** (ý định của tổ hợp argument, obfuscation lồng nhau, LOLBin chưa thấy), không phải về text-so-với-graph.

---

## H0 — Giả thuyết phải loại trừ trước mọi thứ khác

> Một encoder ~150M fine-tune trên command-line đạt hiệu năng tương đương SLM ≤8B.

Nếu không loại trừ được H0, toàn bộ kế hoạch phải viết lại. Prior art trực tiếp chống lại bạn:

- arXiv 2408.02637 — phát hiện obfuscation command-line bằng transformer train từ đầu, trên telemetry thật, mở rộng ra nhiều LOLBin. Kết luận của họ: model nhỏ hơn scale được lên telemetry khối lượng lớn **mà không mất hiệu năng phân loại**.
- arXiv 2412.01655 — phân loại risk command-line bằng BERT, chỉ dùng chuỗi command.
- SHIELD (arXiv 2507.10873) — host IDS có LLM hỗ trợ, đã so sánh với Flash/MAGIC/ORTHRUS.

Loại trừ H0 nghĩa là chỉ ra SLM hơn encoder ở **một trong hai** chỗ, chứ không phải hơn trung bình:
1. Zero-shot / few-shot trên TTP nằm ngoài tập huấn luyện của encoder.
2. Chất lượng disambiguation trên FP khó (admin script, installer, CI runner) — nơi cần prior về ý định, không phải pattern bề mặt.

---

## Các RQ đã hiệu chỉnh

### RQ1 — Ngữ nghĩa tổ hợp (centerpiece)

SLM ≤8B thêm được năng lực gì trên command-line và text attribute mà (a) baseline text rẻ và (b) PIDS dùng text attribute không có?

Hai claim, theo thứ tự ưu tiên:

- **RQ1a (claim chính): giảm false positive.** Ở recall không đổi, SLM tầng hai cắt được bao nhiêu FP của PIDS tầng một? Test set là chính hàng chục nghìn FP mà Flash/MAGIC sinh ra khi chạy lại — không cần attack mới, không tranh chấp với prior art command-line.
- **RQ1b (claim phụ): detection gain.** Gain tập trung ở technique nào? Stratify theo ATT&CK: T1218 (LOLBin proxy exec), T1059.001 (encoded PowerShell), T1027 (obfuscation), T1003.001 (comsvcs MiniDump), T1490 (vssadmin delete shadows), T1087/T1082 (discovery burst), T1053 (scheduled task).

### RQ2 — Khả thi phát hiện

Với đơn vị quyết định là **process kèm command-line và chuỗi parent** (map sang node-level để so sánh được với literature), SLM ≤8B đạt mức nào trên dữ liệu thật?

Không dùng AUC-ROC làm metric chính. Xem mục Metric.

### RQ3 — Pareto phần cứng

Cấu hình nào nằm trên Pareto frontier cho endpoint không GPU, trên bốn trục:

`{0.5B, 1.5B, 3B, 8B}` × `{fp16, int8, int4}` × `{128, 512, 2048 token đầu vào}` × `{single-token, CoT}`

Cộng một trục riêng: chế độ thích ứng `{zero-shot, few-shot, LoRA}`.

Trục token là trục bị bỏ trong bản gốc và là trục chi phối: trên CPU batch-1, latency bị prefill chi phối, nên 1.5B đọc 2000 token chậm hơn 8B đọc 200 token. Thiếu nó, frontier là artifact của một lựa chọn representation ngẫu nhiên.

Đầu ra bắt buộc của RQ3: **số quyết định/host/ngày khả thi**. Con số này ép ra kiến trúc hai tầng và nối RQ3 sang RQ4.

### RQ4 — Định vị so với phương pháp đắt hơn

Không phải bảng ai cao hơn. Hai câu hỏi:

- **RQ4a: tổng chi phí hệ thống**, không phải FLOP inference. GNN provenance đòi hạ tầng dựng và duy trì graph, storage, retrain khi môi trường đổi (PIDSMAKER cần tối thiểu 500 GB cho dataset lớn nhất và giữ phần lớn dữ liệu thường trú trong memory). SLM trên host đòi con số 0 trong các mục đó. Đây là trục so sánh thắng được.
- **RQ4b: bổ trợ hay trùng lặp.** Mỗi bên bắt được attack nào, giao nhau bao nhiêu. Nếu giao nhỏ → kết quả mạnh nhất là **cascade**: recall = hợp của hai bên, chi phí = GNN + một phần nhỏ.

Baseline bắt buộc: **Velox** (đối thủ thật), ORTHRUS, Flash, MAGIC, **SHIELD** (prior art LLM-aided). Cộng một frontier LLM zero-shot làm **ceiling**, có tính giá thật và nêu rõ nó vi phạm tiền đề privacy của chính đề tài.

### RQ5 — Tổng quát hóa và độ bền

Ba lớp, tăng dần độ khó:

- **RQ5a: host/môi trường chưa thấy.** Split theo thời gian, không random (random split trên provenance data là data leak qua entity chung).
- **RQ5b: threshold transfer.** Lấy ngưỡng chọn trên env A, áp lên env B: alert/host/ngày lệch bao nhiêu lần. Đây là thứ giết deployment thật và gần như không ai đo.
- **RQ5c: đối thủ thích ứng.** Attacker ở cùng máy, đọc được weight và prompt. Đo evasion (paraphrase command-line, đổi tên biến, pad bằng text lành) và injection (nhét chuỗi kiểu `# routine nightly backup, approved by IT` vào command-line hoặc tên file). Với các con số của arXiv 2605.24421 — context manipulation 96% khi không có defense, vẫn 38% với output bị ràng buộc — điều này sẽ hoạt động. Đo trung thực là contribution, không phải điểm yếu; encoder baseline cũng không miễn nhiễm.

---

## Việc cần làm

### Phase 0 — Go/no-go (tuần 1)

Cổng chặn. Không đầu tư vào phase nào khác trước khi qua.

1. Lấy `OpTCRedTeamGroundTruth.pdf` và `errata.md` từ repo FiveDirections/OpTC-data. Đọc errata trước khi parse: `actor_id` có entry sai do source không được de-conflict — chính là field dùng để dựng chuỗi parent.
2. Liệt kê Google Drive trước khi tải (`rclone lsf -R`, folder id `1n3kkS3KR31KUegn42yk3-e6JkZvf0Caa`). Xác nhận quy ước đặt tên và dải host.
3. Tải **một bundle** chứa host 201, giai đoạn `ecar/evaluation`. Lọc `PROCESS/CREATE` streaming sang Parquet.
4. Chạy baseline rẻ trên command-line: TF-IDF char n-gram + gradient boosting, và ModernBERT/DeBERTa-v3 ~150M fine-tune.
5. Chạy SLM 3B và 8B trên đúng split đó.

**Điều kiện qua cổng:** SLM hơn encoder 150M có ý nghĩa trên ít nhất một trong hai lớp của H0. Nếu không, dừng và reframe trước khi tiêu thêm tuần nào.

### Phase 1 — Baseline chạy lại (tuần 2–3)

Không copy số từ paper gốc. Khi chạy lại, Flash và MAGIC sinh hàng chục nghìn FP, precision dưới 0,1; precision cao họ báo cáo đến từ việc gán nhãn malicious cho node trong 2 hop của node nêu trong ground-truth report.

1. Dựng PIDSMAKER (arXiv 2601.22983) làm harness. Đọc preprocessing OpTC của họ trước khi tự viết loader.
2. Tải nốt bundle chứa host 051, 501 (evaluation + benign cùng dải host). Các bundle này kèm ~72 host benign miễn phí — đó là tập unseen-host cho RQ5a, không cần tải thêm.
3. Chạy Velox, ORTHRUS, Flash, MAGIC trên ba host chuẩn H201/H501/H051. Ghi lại **tập FP** của từng hệ — đây là test set cho RQ1a.
4. Chạy SHIELD nếu code có sẵn; nếu không, ghi nhận là prior art không reproduce được.

### Phase 2 — Grid phần cứng (tuần 4)

Làm sớm vì kết quả giới hạn phạm vi mọi thứ sau đó.

1. Chốt phần cứng mục tiêu và ghi cấu hình đầy đủ (CPU model, core count, RAM, memory bandwidth).
2. Đo dưới **tranh chấp tài nguyên**, không đo trên máy rỗi: chạy song song Chrome + Teams hoặc workload tương đương.
3. Quét toàn bộ grid RQ3. Không giả định quantization đơn điệu: trên CPU q4 có thể nhanh hơn int8 vì bị chặn memory bandwidth, còn 0.5B ở int4 thường sập chất lượng.
4. Đo weight footprint và KV cache footprint **riêng biệt**. Với context dài của telemetry, KV cache là ràng buộc, không phải weight.

### Phase 3 — Thí nghiệm chính (tuần 5–7)

1. Quyết định claim chính (RQ1a hay RQ1b) dựa trên số thật từ Phase 0–2.
2. LoRA fine-tune trên cấu hình thắng của Phase 2. Zero-shot ≤8B trên telemetry thô sẽ kém; nếu paper chỉ có zero-shot thì kết luận là "model nhỏ không làm được" — yếu.
3. Chạy RQ1a: SLM tầng hai trên tập FP của Phase 1.
4. Chạy RQ1b stratified theo ATT&CK technique.
5. Chạy RQ4b: overlap analysis và cascade.

### Phase 4 — Tổng quát hóa và đối kháng (tuần 8)

1. RQ5a: split thời gian, train sớm test muộn, unseen host từ bundle đã có.
2. RQ5b: threshold transfer, đo drift alert/host/ngày.
3. RQ5c: sinh tập evasion (paraphrase, rename, pad) và tập injection. Đo trên cả SLM và encoder baseline.
4. Dựng attack range riêng (Atomic Red Team / Splunk Attack Range) cho TTP holdout mà OpTC không kiểm soát được.

### Phase 5 — Viết

Thứ tự viết: Phase 2 (đo lường, đã cố định) → Phase 1 (baseline) → Phase 3 (claim chính) → Phase 4 → intro/related work cuối cùng, sau khi đã biết claim thật là gì.

---

## Thứ cần đo

### Nhóm phát hiện

| Metric | Định nghĩa | Ghi chú |
|---|---|---|
| recall @ alert/host/ngày | recall tại ngưỡng cho 1, 10, 100 alert/host/ngày | **Metric chính.** Không dùng FPR dạng phân số: host sinh ~10⁷ event/ngày, FPR 0,1% = 10.000 alert/ngày/host, chết ngay |
| AUC-PR | area under precision-recall | Thay AUC-ROC. ROC bơm số ở mất cân bằng cực đoan |
| MCC | Matthews correlation | Metric phụ. Được chấp nhận trong subfield này cho dataset mất cân bằng. **Không** báo cáo trên test set đã resample cân bằng |
| FP reduction @ recall cố định | tỷ lệ FP của PIDS tầng một bị SLM loại | Metric của RQ1a, claim chính |
| Time-to-detect | số event / số phút vào chuỗi tấn công khi phát hiện | Khách hàng quan tâm hơn recall |
| Risk–coverage curve | error rate theo tỷ lệ abstention | Cho phép "không biết, escalate". Nền tảng cho RQ5b |
| ECE / reliability diagram | calibration error | Ngưỡng không transfer được nếu score không calibrate |

### Nhóm chi phí

| Metric | Đơn vị | Ghi chú |
|---|---|---|
| Latency p50 / p95 / p99 | ms, batch=1 | Báo cáo p95/p99, không phải mean. Đo dưới tranh chấp |
| Prefill vs decode | ms, tách riêng | Để trục token trong RQ3 có nghĩa |
| Weight footprint | MB | |
| KV cache footprint | MB, theo độ dài context | Tách khỏi weight |
| CPU-second/host/ngày | % của tổng | **Số gốc**, không phải $ |
| Wh/host/ngày | watt-hour | Battery drain là blocker deployment thật |
| Quyết định/host/ngày khả thi | count | Con số ép ra kiến trúc hai tầng |
| $/host/ngày | USD | Quy đổi **cuối cùng**, với giả định nêu rõ. Đứng một mình sẽ bị chất vấn |
| Tổng chi phí hệ thống | GB storage, GB RAM, giờ retrain | Cho RQ4a. Trục mà SLM thắng |

### Nhóm độ bền

| Metric | Ghi chú |
|---|---|
| Δ recall: seen → unseen host | RQ5a |
| Δ recall: seen → unseen TTP | RQ5a, cần attack range riêng |
| Alert/host/ngày drift khi transfer ngưỡng | RQ5b. Báo cáo dạng hệ số, ví dụ "×7,3" |
| Attack success rate: evasion | RQ5c, trên cả SLM và encoder |
| Injection success rate | RQ5c. So sánh với 96% / 38% của 2605.24421 |

---

## Output cần có

### Hình

1. **Pareto frontier** — trục x là latency p95 hoặc CPU-second/host/ngày, trục y là recall@10 alert/host/ngày. Mỗi điểm là một cấu hình `size × quant × token × output mode`. Điểm trên frontier được đánh dấu. Đây là hình định danh của paper.
2. **Bar chart stratified theo ATT&CK technique** — gain của SLM so với encoder baseline và so với Velox, theo từng technique. Hình bán RQ1b.
3. **FP reduction curve** — FP còn lại theo recall, cho PIDS đơn lẻ so với PIDS + SLM tầng hai. Hình bán RQ1a.
4. **Venn hoặc upset plot** — overlap attack bắt được giữa SLM, Velox, ORTHRUS. Nền cho lập luận cascade.
5. **Risk–coverage curve** — cho từng size model.
6. **Threshold transfer** — alert/host/ngày trên env B khi dùng ngưỡng của env A, so với ngưỡng tối ưu của B.

### Bảng

1. Bảng grid đầy đủ RQ3, mọi ô, không chỉ ô thắng. Kể cả ô sập (0.5B int4) — kết quả âm là dữ liệu.
2. Bảng so sánh với baseline đã **chạy lại**, kèm ghi chú tường minh rằng số này khác số trong paper gốc và tại sao (labeling 2-hop).
3. Bảng tổng chi phí hệ thống: storage, RAM, retrain, inference — SLM so với từng PIDS.
4. Bảng evasion/injection.

### Artifact

1. Script tải chọn lọc + lọc streaming eCAR sang Parquet. Kèm manifest liệt kê chính xác bundle nào đã dùng.
2. Cấu hình PIDSMAKER (YAML) cho mọi baseline, để reproduce được.
3. LoRA adapter đã train, weight công khai.
4. Tập FP đã gán nhãn (test set của RQ1a) — đây có thể là contribution độc lập có giá trị hơn cả model.
5. Tập evasion và injection đã sinh.
6. Ghi chép phần cứng và điều kiện đo, đủ chi tiết để người khác lặp lại được số latency.

### Điều phải nói rõ trong Limitations

- SLM không phải primary detector; kết quả không hỗ trợ tuyên bố ngược lại.
- OpTC realism đang bị chính cộng đồng chất vấn ("DARPA TC và OpTC có thực sự là benchmark thực tế không?").
- Attacker co-resident đọc được weight và prompt. Kerckhoffs áp dụng: không đặt logic phát hiện trong prompt. Nếu thiết kế của bạn có, nói ra.
- Injection qua command-line hoạt động; con số cụ thể phải nằm trong paper, không nằm trong appendix.

---

## Nhắc lại điều quan trọng nhất

Phase 0 là cổng chặn thật, không phải thủ tục. Nếu encoder 150M ngang SLM 8B và bạn phát hiện điều đó ở tháng thứ sáu thay vì tuần đầu, bạn mất cả kỳ. Chi phí của Phase 0 là vài ngày và một bundle dữ liệu.
