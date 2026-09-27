import argparse
import bisect
import json
import math
from pathlib import Path

from eve import read_jsonl


def rows(path):
    out = []
    for r in read_jsonl(path):
        if not r.get("label") or not r.get("gold_techniques"):
            continue
        p = r.get("technique_probs") or {}
        gold = {t[:5] for t in r["gold_techniques"]}
        inscope = bool(gold & set(p))
        score = 1.0 - max(p[g] for g in gold & set(p)) if inscope else None
        out.append({"id": r["sample_id"], "probs": p, "gold": gold, "inscope": inscope, "score": score})
    return out


def quantile(scores, alpha):
    s = sorted(scores)
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return math.inf if k > n else s[k - 1]


class WeightedQuantile:
    def __init__(self, scores, weights):
        pairs = sorted(zip(scores, weights))
        self.s = [a for a, _ in pairs]
        self.cum = []
        c = 0.0
        for _, w in pairs:
            c += w
            self.cum.append(c)
        self.total = c

    def __call__(self, alpha, w_test):
        target = (1 - alpha) * (self.total + w_test)
        i = bisect.bisect_left(self.cum, target)
        return math.inf if i >= len(self.s) else self.s[i]


def domain_weights(calib_ids, test_ids, calib_text, test_text, clip=20.0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    Xc = [calib_text[i] for i in calib_ids]
    Xt = [test_text[i] for i in test_ids]
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(Xc + Xt)
    y = [0] * len(Xc) + [1] * len(Xt)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    ratio = len(Xc) / max(len(Xt), 1)
    w = [min(clip, max(1.0 / clip, (q / max(1 - q, 1e-6)) * ratio)) for q in p]
    return dict(zip(calib_ids, w[:len(Xc)])), dict(zip(test_ids, w[len(Xc):]))


def evaluate(calib, test, alpha, wq=None, w_test=None):
    cal = [r["score"] for r in calib if r["inscope"]]
    q = quantile(cal, alpha) if wq is None else None
    cov, cov_in, sizes, empty = [], [], [], 0
    for r in test:
        qq = q if wq is None else wq(alpha, w_test[r["id"]])
        S = {t for t, p in r["probs"].items() if 1.0 - p <= qq}
        hit = int(bool(S & r["gold"]))
        cov.append(hit)
        if r["inscope"]:
            cov_in.append(hit)
            sizes.append(len(S))
            empty += int(not S)
    kmiss = 1 - sum(r["inscope"] for r in test) / len(test)
    miss = 1 - sum(cov) / len(cov)
    return {
        "alpha": alpha, "qhat": q, "n_calib_inscope": len(cal), "n_test": len(test),
        "coverage": 1 - miss, "coverage_inscope": sum(cov_in) / len(cov_in) if cov_in else None,
        "target_inscope": 1 - alpha, "avg_set_size_inscope": sum(sizes) / len(sizes) if sizes else None,
        "empty_set_rate_inscope": empty / len(sizes) if sizes else None,
        "knowledge_miss": kmiss, "model_miss": miss - kmiss,
        "bound_knowledge_plus_alpha": kmiss + alpha, "bound_holds": miss <= kmiss + alpha + 1e-9,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--calib_data", default=None)
    ap.add_argument("--test_data", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    calib, test = rows(a.calib), rows(a.test)
    if not calib or not test:
        raise SystemExit("need malicious samples with gold techniques in both calib and test")
    res = {"calib": a.calib, "test": a.test, "unweighted": [evaluate(calib, test, al) for al in a.alphas]}
    if a.weighted:
        if not (a.calib_data and a.test_data):
            raise SystemExit("--weighted needs --calib_data and --test_data for the domain classifier")
        ct = {d["sample_id"]: d.get("text", "") for d in read_jsonl(a.calib_data)}
        tt = {d["sample_id"]: d.get("text", "") for d in read_jsonl(a.test_data)}
        cids = [r["id"] for r in calib if r["inscope"] and r["id"] in ct]
        tids = [r["id"] for r in test if r["id"] in tt]
        wc, wt = domain_weights(cids, tids, ct, tt)
        cal = [r for r in calib if r["inscope"] and r["id"] in wc]
        wq = WeightedQuantile([r["score"] for r in cal], [wc[r["id"]] for r in cal])
        tst = [r for r in test if r["id"] in wt]
        res["weighted"] = [evaluate(cal, tst, al, wq, wt) for al in a.alphas]
        res["weight_stats"] = {"calib_mean": sum(wc.values()) / len(wc), "test_mean": sum(wt.values()) / len(wt)}
    txt = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
