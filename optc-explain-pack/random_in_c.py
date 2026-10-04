"""EVE so voi chon ngau nhien deu trong C(E) (cung tap ung vien, khong dung SLM).
Ky vong top-1 cua chon ngau nhien = trung binh |C ∩ gold| / |C| (0 khi C rong); Monte Carlo cho phan phoi
va p mot phia P(random >= he thong). Bao cao tren toan bo alert co nhan TTP va tren nhom |C| >= 2 (noi SLM phai chon).

  python random_in_c.py --results q05_REAL_eve.jsonl kb_only_first.jsonl --out random_in_c.json
"""
import argparse
import json
import random
from pathlib import Path

from eve import read_jsonl


def analyse(rs, n_sim, seed):
    m = [r for r in rs if r.get("label") and r.get("gold_techniques")]
    rows = []
    for r in m:
        gold = {t[:5] for t in r["gold_techniques"]}
        cand = sorted({c[:5] for c in r.get("candidates") or []})
        rows.append((cand, gold, int((r.get("technique") or "")[:5] in gold)))

    def block(sel):
        if not sel:
            return {"n": 0}
        n = len(sel)
        sys_acc = sum(h for _, _, h in sel) / n
        exp = sum(len(set(c) & g) / len(c) for c, g, _ in sel if c) / n
        rng = random.Random(seed)
        sims = []
        for _ in range(n_sim):
            sims.append(sum(int(rng.choice(c) in g) for c, g, _ in sel if c) / n)
        sims.sort()
        upper = sum(int(bool(set(c) & g)) for c, g, _ in sel) / n
        return {"n": n, "acc_system": round(sys_acc, 4), "acc_random_expected": round(exp, 4),
                "acc_random_ci95": [round(sims[int(0.025 * n_sim)], 4), round(sims[int(0.975 * n_sim) - 1], 4)],
                "p_random_ge_system": round(sum(s >= sys_acc - 1e-12 for s in sims) / n_sim, 4),
                "gain_over_random": round(sys_acc - exp, 4),
                "oracle_in_C": round(upper, 4)}

    amb = [x for x in rows if len(x[0]) >= 2]
    return {"all_ttp": block(rows), "C_ge_2": block(amb),
            "C_size_dist": {k: sum(1 for c, _, _ in rows if len(c) == k) for k in sorted({len(c) for c, _, _ in rows})}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--n_sim", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = []
    for p in a.results:
        rs = list(read_jsonl(p))
        res.append({"file": p, "mode": rs[0].get("mode") if rs else None, "model": rs[0].get("model") if rs else None,
                    **analyse(rs, a.n_sim, a.seed)})
        x, y = res[-1]["all_ttp"], res[-1]["C_ge_2"]
        print(f"{Path(p).name:40s} all n={x['n']} sys={x.get('acc_system')} rand={x.get('acc_random_expected')} | "
              f"|C|>=2 n={y['n']} sys={y.get('acc_system')} rand={y.get('acc_random_expected')} "
              f"p={y.get('p_random_ge_system')} oracle={y.get('oracle_in_C')}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
