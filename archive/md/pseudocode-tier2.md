# PSEUDOCODE — TẦNG 2 (SLM TIER-2) VÀ CÁC THUẬT TOÁN ĐO LƯỜNG

Tài liệu này chuyển logic tính toán của Phase 1/tầng 2 thành mã giả
để đưa vào paper. Mỗi thuật toán: 1) mục đích, 2) mã giả, 3) tham số
thực tế đã dùng, 4) vị trí code trong repo.

---

## A0. Ký hiệu chung

```
G        = đồ thị provenance theo thời gian (TemporalData)
N        = tập node; |N| ≈ 914,000–927,000 (tùy model)
s(v)     = anomaly score tầng 1 của node v (ORTHRUS hoặc Velox)
Y        = tập node malicious theo ground truth, |Y| = 114
A(k)     = top-k node theo s(v) giảm dần  (alert budget k)
Y_act    = các node GT thực sự xuất hiện trong test period
           ("active"), |Y_act| ≈ 34–37
msg(v)   = chuỗi mô tả node v: đường dẫn file / lệnh chạy / netflow
           (path + cmdline + ip:port)
```

---

## A1. THRESHOLD SWEEP — Recall@k của tầng 1

**Mục đích:** đo chất lượng *ranking* của tầng 1 độc lập với ngưỡng,
trả lời "với ngân sách xem xét k cảnh báo, bắt được bao nhiêu %".

```
THRESHOLD-SWEEP(scores s(v) ∀v∈N, ground truth Y, budgets K):
    order  ← sort N by s(v) descending
    for each k in K:                      # K = {10K, 20K, 50K}
        A_k     ← first k nodes in order
        TP_k    ← |A_k ∩ Y|
        R@k     ← TP_k / |Y|              # recall trên full GT
        R̃@k    ← TP_k / |Y ∩ N_scored|   # recall trên active-GT ceiling
        P@k     ← TP_k / k
    return {(k, TP_k, R@k, P@k)}
```
Tham số thực tế: K = {10000, 20000, 50000}.
Code: `P1/Code/patches/orthrus_threshold_sweep.py`.

---

## A2. PRECISION CEILING — trần precision của path+cmd

**Mục đích:** trả lời câu hỏi gate cho tầng 2 — "nếu chỉ dùng văn bản
mô tả node (path+cmd), có thể phân biệt FP khỏi TP tới đâu?" FP mà
có `msg()` trùng hệt một TP thì **không thể** tách bằng nội dung.

```
PRECISION-CEILING(A(K), Y, msg):
    TP   ← A_K ∩ Y ;  FP ← A_K \ Y
    collide_GT  ← ∅ ; collide_TP ← ∅
    for each f in FP:
        m_f = msg(f)
        if m_f is None:          # thiếu label (hiếm)
            continue
        if m_f ∈ msg(y), ∀y ∈ Y:      # trùng với BẤT KỲ node GT
            collide_GT.add(f)
        if m_f ∈ msg(t), ∀t ∈ TP:     # trùng với TP đã phát hiện
            collide_TP.add(f)
    # Trần precision nếu bộ lọc hoàn hảo loại mọi FP có thể tách:
    ceiling(k) = |TP| / (|TP| + |FP \ collide_GT|)
    return collide_GT, collide_TP, ceiling(k)
```

Kết quả thực tế (ORTHRUS): k=2K → ceiling 33.3%; k=10K → 8.46%.
Code: `P1/Code/patches/orthrus_precision_ceiling.py`,
`orthrus_ceiling_corrected.py` (bản sửa bug label lookup).

---

## A3. ENRICH ALERT — dựng ngữ cảnh cho tầng 2

**Mục đích:** mỗi alert chỉ là 1 node ID + score — SLM cần *ngữ cảnh*.
Enrich gắn thêm chuỗi tổ tiên (parent chain) và chuỗi sự kiện gần nhất.

```
ENRICH-ALERT(v, G):
    parent_chain ← []
    u ← v
    while u has parent p and depth < 3:      # tối đa 3 thế hệ
        append {node: label(p), op: op(p→u)} to parent_chain
        u ← p
    event_seq ← 20 events mới nhất có src hoặc dst = v
                (lấy từ event_table, sort by timestamp)
    self_label ← label(v)                    # path + cmd hoặc netflow
    return JSONL record {nid, rank, score, self_label,
                         parent_chain, event_seq}
```
Thực tế: 2250/10000 alerts đã enrich (bị chặn bởi WSL; script
resume-capable). Code: nằm ở máy thay (`enrich_alerts.py`), bản partial
`P1/Output/data/alerts-enriched-partial.jsonl`.

---

## A4. BASELINE H0 — TF-IDF char n-gram + Logistic Regression

**Mục đích:** baseline encoder RẺ để loại trừ (H0). Nếu TF-IDF cắt được
FP tốt như SLM thì SLM không có giá trị gia tăng (RQ1a).

```
TFIDF-BASELINE(alerts E, Y, k):
    for each e in E:
        text(e) ← self_label + Σ op+node(parent_chain)
                       + Σ src+op+dst(event_seq[:10])
        y(e)   ← 1 if e.nid ∈ Y else 0
    X ← TFIDF(text, analyzer=char_wb, ngram=(2,5),
              max_features=50000, sublinear_tf=True)
    clf ← LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y)
    scores ← clf.predict_proba(X)[:,1]
    order  ← argsort(-scores); A_k ← order[:k]
    keep   ← {i ∈ A_k : scores[i] > 0.5}       # ngưỡng filter 0.5
    TP'    ← |{i ∈ keep : y(i)=1}| ;  FP' ← |{i ∈ keep : y(i)=0}|
    FP-reduction ← 1 − FP'/FP_before
    return metrics
```
Kết quả thực tế: giữ 12/12 TP (100%), FP 1988→13 (−99.35%).
Code: `run_baseline_tfidf()` trong `P1/Code/slm_tier2.py`.

---

## A5. SLM TIER-2 ZERO-SHOT CLASSIFIER (thuật toán chính)

**Mục đích:** dùng SLM (Qwen2.5-3B int4) phân loại lại từng alert
malicious/benign dựa trên ngữ cảnh enriched — KHÔNG fine-tune,
KHÔNG đặt logic phát hiện vào prompt (tuân thủ Kerckhoffs theo Note.md:
prompt chỉ hướng dẫn *cách trình bày*, tiêu chí nhận định là kiến thức
chung của analyst, không phải signature riêng của dataset).

```
SLM-TIER2(E_enriched, Y, k):
    model ← LoadQuantized("Qwen2.5-3B-Instruct", nf4-int4,
                          double_quant=True, compute_dtype=fp16)
    results ← []
    for each alert e in E_enriched sorted by score desc (top-k):
        prompt ← TEMPLATE(
            system : vai trò analyst + yêu cầu format đầu ra
                     [CLASSIFICATION / CONFIDENCE / REASON]
            user   : Process Identity (self_label)
                   + Process Tree ≤3 tổ tiên (parent_chain)
                   + 20 sự kiện gần nhất (event_seq)
                   + câu hỏi MALICIOUS hay BENIGN?)
        resp   ← greedy_generate(model, prompt, max_new_tokens=128)
        (pred, conf, reason) ← PARSE(resp)
        append {nid, score, slm_pred=pred, confidence=conf}
    # Bộ lọc tier-2: chỉ giữ alert được SLM xác nhận "malicious"
    kept  ← {r : r.slm_pred = "malicious"}
    TP₂   ← |{r ∈ kept : r.nid ∈ Y}| ;  FP₂ ← |kept| − TP₂
    FP-red ← 1 − FP₂/FP₁
    return results, FP-red, precision trước/sau
```

PARSE (phân tích đầu ra có cấu trúc, fail-safe về "uncertain"):
```
PARSE(resp):
    pred ← "uncertain"; conf ← 0.5
    for line in splitlines(resp):
        if line starts with "CLASSIFICATION:" :
            pred ← "malicious" if contains MALICIOUS
                  else "benign" if contains BENIGN
        elif line starts with "CONFIDENCE:" :
            conf ← float(...)  (giữ 0.5 nếu parse lỗi)
        elif line starts with "REASON:" :
            reason ← line sau dấu ":"
    return (pred, conf, reason)
```

Điểm thiết kế đáng nêu trong paper:
- Greedy decoding (do_sample=False) để tái lập được kết quả.
- Parse fail-safe: không parse được → "uncertain" (conf 0.5), KHÔNG
  suy đoán; uncertain bị filter (an toàn phía giảm FP).
- Quantization nf4/int4 double-quant để chạy trên GPU 16GB.
- Confidence dùng vẽ risk–coverage curve (RQ phụ).

Code: `classify_with_slm()`, `parse_slm_response()` trong
`P1/Code/slm_tier2.py`.

---

## A6. METRICS SAU FILTER — công thức đánh giá tầng 2

```
EVAL-TIER2(results, Y, k):
    A_k   ← top-k alerts (theo score tầng 1)
    TP₁   ← |A_k ∩ Y| ;  FP₁ ← k − TP₁
    kept  ← {a ∈ A_k : a.slm_pred = "malicious"}
    TP₂   ← |kept ∩ Y| ;  FP₂ ← |kept| − TP₂
    Precision₁ = TP₁/k
    Precision₂ = TP₂/(TP₂+FP₂)
    FPReduction = 1 − FP₂/max(FP₁,1)
    RecallBefore = TP₁/|Y| ;  RecallAfter = TP₂/|Y|
    # Điều kiện thành công RQ1a:
    #   FPReduction cao  VÀ  RecallAfter ≈ RecallBefore (fixed recall)
    return tất cả
```
Code: `compute_metrics()` trong `P1/Code/slm_tier2.py`.

---

## GHI CHÚ CHO PAPER

1. **Thứ tự trình bày khuyến nghị**: A1 (đo tầng 1) → A2 (gate) →
   A3 (enrich) → A4 (baseline H0) → A5 (SLM) → A6 (đánh giá).
2. Ceiling (A2) nên đứng TRƯỚC kết quả SLM: nó định nghĩa không gian
   khả dĩ — FP trùng msg với TP là giới hạn lý thuyết của mọi bộ lọc
   dựa trên path+cmd.
3. Số liệu hiện có: A1, A2, A4 đã chạy đủ. A3 đang 2250/10000.
   A5+A6 chờ A3 xong (hoặc chạy trên 2250 alerts làm thí điểm).
