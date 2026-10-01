import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_ttp(data, min_pos, max_train, seed):
    lab, real = [], []
    for r in read_jsonl(data):
        if not r.get("label") or not r.get("techniques_base"):
            continue
        row = {"id": r["sample_id"], "text": r.get("text", ""), "tech": set(r["techniques_base"])}
        (lab if r.get("env") == "LAB" else real if r.get("env") == "REAL" else []).append(row)
    count = lambda rows, t: sum(t in x["tech"] for x in rows)
    alltech = sorted({t for x in lab + real for t in x["tech"]})
    common = [t for t in alltech if count(lab, t) >= min_pos and count(real, t) >= min_pos]
    rng = random.Random(seed)
    if max_train and len(lab) > max_train:
        lab = rng.sample(lab, max_train)
    Y = lambda rows: np.array([[int(t in x["tech"]) for t in common] for x in rows])
    return lab, real, common, Y(lab), Y(real)


def load_verdict(data, test_file, max_train, seed):
    lab = [{"id": r["sample_id"], "text": r.get("text", ""), "y": int(r["label"])}
           for r in read_jsonl(data) if r.get("env") == "LAB"]
    rng = random.Random(seed)
    pos = [x for x in lab if x["y"]]
    neg = [x for x in lab if not x["y"]]
    k = min(len(pos), len(neg), max_train // 2 if max_train else len(pos))
    lab = rng.sample(pos, k) + rng.sample(neg, k)
    rng.shuffle(lab)
    test = [{"id": r["sample_id"], "text": r.get("text", ""), "y": int(r["label"])} for r in read_jsonl(test_file)]
    return lab, test


def tfidf_fit_predict(tr_text, Ytr, te_text, C, seed):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=100_000, min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(tr_text)
    Xte = vec.transform(te_text)
    Ytr = Ytr if Ytr.ndim == 2 else Ytr[:, None]
    P = np.zeros((Xte.shape[0], Ytr.shape[1]))
    for j in range(Ytr.shape[1]):
        if Ytr[:, j].sum() == 0:
            continue
        clf = LogisticRegression(max_iter=3000, C=C, class_weight="balanced", random_state=seed)
        clf.fit(Xtr, Ytr[:, j])
        P[:, j] = clf.predict_proba(Xte)[:, 1]
    return P


def tfidf_tune(tr_text, Ytr, grid, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(tr_text))
    cut = int(0.8 * len(idx))
    a, b = idx[:cut], idx[cut:]
    best, best_c = -1, grid[0]
    for C in grid:
        P = tfidf_fit_predict([tr_text[i] for i in a], Ytr[a], [tr_text[i] for i in b], C, seed)
        Yb = Ytr[b] if Ytr.ndim == 2 else Ytr[b][:, None]
        aps = [average_precision_score(Yb[:, j], P[:, j]) for j in range(Yb.shape[1]) if Yb[:, j].sum() > 0]
        m = float(np.mean(aps)) if aps else -1
        if m > best:
            best, best_c = m, C
    return best_c, best


def hf_fit_predict(model_name, tr_text, Ytr, te_text, a, seed):
    import torch
    import torch.nn as nn
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    dev = a.device
    dtype = torch.bfloat16 if a.dtype in ("bf16", "bfloat16") else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    Ytr = Ytr if Ytr.ndim == 2 else Ytr[:, None]
    n_lab = Ytr.shape[1]
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_lab,
                                                               problem_type="multi_label_classification")
    model.config.pad_token_id = tok.pad_token_id
    model.to(dev)
    if dtype == torch.bfloat16:
        model.to(dtype)
    if a.grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    pos = Ytr.sum(0)
    pw = torch.tensor(np.clip((len(Ytr) - pos) / np.maximum(pos, 1), 1, 50), dtype=torch.float32, device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = max(1, (len(tr_text) // a.batch_size + 1) * a.epochs // a.grad_accum)
    sched = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)

    def batches(texts, ys=None, shuffle=False):
        order = list(range(len(texts)))
        if shuffle:
            random.shuffle(order)
        for s in range(0, len(order), a.batch_size):
            idx = order[s:s + a.batch_size]
            enc = tok([texts[i] for i in idx], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            yield enc, (torch.tensor(ys[idx], dtype=torch.float32, device=dev) if ys is not None else None)

    t0 = time.time()
    model.train()
    step = 0
    for ep in range(a.epochs):
        tot, nb = 0.0, 0
        for enc, y in batches(tr_text, Ytr, shuffle=True):
            loss = loss_fn(model(**enc).logits.float(), y) / a.grad_accum
            loss.backward()
            nb += 1
            if nb % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
            tot += loss.item() * a.grad_accum
            if a.progress and nb % a.progress == 0:
                print(f"seed {seed} ep {ep} batch {nb} loss {tot / nb:.4f} {time.time() - t0:.0f}s", flush=True)
    model.eval()
    out = []
    with torch.no_grad():
        for enc, _ in batches(te_text):
            out.append(torch.sigmoid(model(**enc).logits.float()).cpu().numpy())
    return np.vstack(out), time.time() - t0


def run_once(a, seed, cache):
    if a.task == "ttp":
        lab, real, common, Ytr, Yte = load_ttp(a.data, a.min_pos, a.max_train, seed)
        tr_text, te_text = [x["text"] for x in lab], [x["text"] for x in real]
    else:
        lab, real = load_verdict(a.data, a.test_file, a.max_train, seed)
        common = ["malicious"]
        tr_text, te_text = [x["text"] for x in lab], [x["text"] for x in real]
        Ytr = np.array([[x["y"]] for x in lab])
        Yte = np.array([[x["y"]] for x in real])
    info = {"seed": seed, "n_train": len(tr_text), "n_test": len(te_text), "labels": common}
    if a.backend == "tfidf":
        C = a.C
        if a.tune:
            C, dev_ap = tfidf_tune(tr_text, Ytr, a.C_grid, seed)
            info["tuned_C"], info["tune_dev_mean_ap"] = C, round(dev_ap, 4)
        t0 = time.time()
        P = tfidf_fit_predict(tr_text, Ytr, te_text, C, seed)
        info["train_s"] = round(time.time() - t0, 1)
    else:
        P, sec = hf_fit_predict(a.model, tr_text, Ytr, te_text, a, seed)
        info["train_s"] = round(sec, 1)
    if a.task == "ttp":
        per = []
        for j, t in enumerate(common):
            y = Yte[:, j]
            per.append({"ttp": t, "AP": round(float(average_precision_score(y, P[:, j])), 4),
                        "AUC": round(float(roc_auc_score(y, P[:, j])), 4) if 0 < y.sum() < len(y) else None,
                        "n_train_pos": int(Ytr[:, j].sum()), "n_test_pos": int(y.sum())})
        info["per_ttp"] = per
        info["mean_AP"] = round(float(np.mean([p["AP"] for p in per])), 4) if per else None
        info["mean_AP_ci"] = boot_mean_ap(Yte, P, seed) if per else None
    else:
        y = Yte[:, 0]
        info["auc"] = round(float(roc_auc_score(y, P[:, 0])), 4)
        info["ap"] = round(float(average_precision_score(y, P[:, 0])), 4)
        if a.save_scores:
            ids = [x["id"] for x in real]
            Path(a.save_scores).write_text(json.dumps(dict(zip(ids, P[:, 0].tolist()))), encoding="utf-8")
    return info


def boot_mean_ap(Y, P, seed, n=300):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        idx = rng.randint(0, len(Y), len(Y))
        aps = [average_precision_score(Y[idx, j], P[idx, j]) for j in range(Y.shape[1]) if Y[idx, j].sum() > 0]
        if aps:
            vals.append(float(np.mean(aps)))
    vals.sort()
    return [round(vals[int(0.025 * (len(vals) - 1))], 4), round(vals[int(0.975 * (len(vals) - 1))], 4)] if vals else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("ttp", "verdict"), default="ttp")
    ap.add_argument("--backend", choices=("tfidf", "hf"), default="tfidf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--test_file", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--min_pos", type=int, default=20)
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--C_grid", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--progress", type=int, default=100)
    ap.add_argument("--save_scores", default=None)
    a = ap.parse_args(argv)
    if a.task == "verdict" and not a.test_file:
        raise SystemExit("--task verdict needs --test_file (e.g. REAL_test_matched.jsonl)")
    runs = [run_once(a, s, None) for s in a.seeds]
    key = "mean_AP" if a.task == "ttp" else "auc"
    vals = [r[key] for r in runs if r.get(key) is not None]
    summary = {"task": a.task, "backend": a.backend, "model": a.model if a.backend == "hf" else "tfidf-char2-5",
               "split": "LAB->REAL", "seeds": a.seeds, "metric": key,
               "mean": round(statistics.mean(vals), 4) if vals else None,
               "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0, "runs": runs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("task", "backend", "model", "metric", "mean", "std")}, indent=1))
    print("labels:", runs[0]["labels"])


if __name__ == "__main__":
    main()
