"""baseline_tfidf.py — Baseline re nhat: TF-IDF char n-gram + GradientBoosting.

H0 phai loai truoc SLM: neu encoder ~150M (hoac ngay ca baseline re nay) da ngang SLM,
ke hoach phai viet lai. Chay script nay TRUOC baseline_encoder.py va train.py (SLM).

Chay (causal split — tach theo thoi gian, cung host o train/test):
  python baseline_tfidf.py --data "E:\\dataset\\processed\\train_data_201_refined.jsonl"

Chay (host-holdout — train host nay, test host khac chua thay; dung cho RQ5a/kiem tra
tong quat hoa qua host thay vi ghi nho trong 1 host):
  python baseline_tfidf.py --data "E:\\dataset\\processed\\train_data_frozen_v2.jsonl" ^
      --split-mode host_holdout --holdout-hosts SysClient0501.systemia.com
"""
from __future__ import annotations

import argparse
import json
import time

from config import DataConfig
from data import load_split
from metrics import all_metrics, print_metrics
from run_utils import file_sha256, make_run_dir, save_run_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-frac", type=float, default=0.6)
    ap.add_argument("--split-mode", choices=["causal", "host_holdout"], default="causal")
    ap.add_argument("--holdout-hosts", nargs="*", default=None,
                     help="danh sach host chi dung o test (split-mode=host_holdout)")
    ap.add_argument("--neg-ratio", type=float, default=None,
                     help="undersample benign trong TRAIN ve <ratio>:1 so voi malicious "
                          "(vd 20 = 20 benign/1 malicious). Test khong doi. Dung de so "
                          "sanh cong bang voi train.py/baseline_encoder.py cung neg-ratio.")
    ap.add_argument("--seed", type=int, default=42,
                     help="seed cho undersample + HistGBM. Doi de chay nhieu seed cung 1 "
                          "cau hinh -> bao cao mean+-std thay vi 1 con so.")
    args = ap.parse_args()

    holdout_tag = "-".join(h.split(".")[0] for h in (args.holdout_hosts or []))
    run_dir = make_run_dir("baseline_tfidf", split=args.split_mode,
                            holdout=holdout_tag, seed=args.seed)
    print(f"[run] artifact -> {run_dir}")
    dataset_hash = file_sha256(args.data)
    print(f"[data] sha256({args.data}) = {dataset_hash}")

    cfg = DataConfig(dataset_path=args.data, split_frac=args.split_frac,
                      split_mode=args.split_mode, holdout_hosts=args.holdout_hosts,
                      neg_pos_ratio=args.neg_ratio, seed=args.seed)
    train_rows, test_rows = load_split(cfg)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import HistGradientBoostingClassifier

    X_train_text = [r["text"] for r in train_rows]
    y_train = [r["label"] for r in train_rows]
    X_test_text = [r["text"] for r in test_rows]
    y_test = [r["label"] for r in test_rows]

    print("[tfidf] fit vectorizer (char 3-5 n-gram)...")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=4_096,
                           sublinear_tf=True)
    t0 = time.time()
    X_train = vec.fit_transform(X_train_text).toarray().astype("float32")
    X_test = vec.transform(X_test_text).toarray().astype("float32")
    print(f"[tfidf] vocab={len(vec.vocabulary_):,} fit+transform={time.time()-t0:.1f}s "
          f"shape train={X_train.shape} test={X_test.shape}")

    print("[gbm] fit HistGradientBoostingClassifier...")
    t0 = time.time()
    n_pos = sum(y_train); n_neg = len(y_train) - n_pos
    # early_stopping=False: mac dinh sklearn tu tach 10% X_train lam validation noi bo
    # THEO VI TRI trong mang (khong phai theo noi dung) de quyet dinh dung som. Voi tap
    # duong cuc nho (vai tram mau), doi thu tu dong (du cung tap hop, cung noi dung) se
    # doi han nhung dong nao roi vao validation do -> AUC khong tai lap duoc giua cac lan
    # chay (da kiem chung: doi thu tu -> AUC lech >0.05). Tat han de moi lan chay du dung
    # max_iter=300 vong, khong phu thuoc thu tu mang.
    clf = HistGradientBoostingClassifier(
        max_iter=300, class_weight="balanced", random_state=args.seed, early_stopping=False)
    clf.fit(X_train, y_train)
    train_sec = time.time() - t0
    print(f"[gbm] train={train_sec:.1f}s (pos={n_pos} neg={n_neg})")

    t0 = time.time()
    scores = clf.predict_proba(X_test)[:, 1]
    infer_ms = 1000 * (time.time() - t0) / max(len(test_rows), 1)

    m = all_metrics(y_test, list(scores))
    print("\n===== BASELINE: TF-IDF(char 3-5) + HistGBM =====")
    print_metrics(m)
    print(f"  latency/mau   : {infer_ms:.3f} ms (CPU, sklearn)")
    print(f"  train time    : {train_sec:.1f}s")
    print(f"  test pos/neg  : {sum(y_test)}/{len(y_test)-sum(y_test)}")

    with open(run_dir / "scores.jsonl", "w", encoding="utf-8") as f:
        for r, sc in zip(test_rows, scores):
            f.write(json.dumps({"score": float(sc), "label": int(r["label"]),
                                "host": r["meta"].get("host")}) + "\n")
    save_run_metrics(run_dir, data=args.data, dataset_sha256=dataset_hash,
                      split_mode=args.split_mode,
                      holdout_hosts=args.holdout_hosts, split_frac=args.split_frac,
                      neg_pos_ratio=args.neg_ratio, seed=args.seed,
                      train_sec=round(train_sec, 2), latency_ms=round(infer_ms, 4),
                      test_pos=int(sum(y_test)), test_neg=int(len(y_test) - sum(y_test)),
                      **m)
    print(f"[run] scores + metrics.json -> {run_dir}")


if __name__ == "__main__":
    main()
