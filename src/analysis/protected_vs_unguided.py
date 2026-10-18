"""
Guided (prefix-protected) variants versus the prefix-protected unguided baselines.

Per seed: the unguided baseline with the highest ASR on the same 500 URLs is the
reference (selecting it on the evaluation URLs favours the baseline, i.e. the test
is conservative for the guided methods). Each ranker's protected result is compared
with it by exact McNemar test (BH-FDR within seed and mode).

Run (repo root):
    python src/analysis/protected_vs_unguided.py --seed_dirs results/tables/seed0 ... seed4 \
        --out results/tables/cross_seed/protected_vs_unguided.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


def bh(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, prev = [0.0] * m, 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, ps[i] * m / rank)
        adj[i] = prev
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_dirs", nargs="+", required=True)
    ap.add_argument("--out", default="protected_vs_unguided.csv")
    args = ap.parse_args()
    rows, base_rows = [], []
    for s, d in enumerate(args.seed_dirs):
        d = Path(d)
        ung = {}
        for f in sorted((d / "unguided_protected").glob("*.csv")):
            x = pd.read_csv(f)
            ung[x["method"].iloc[0]] = x.set_index("url")["targeted_success"].astype(int)
        asr = {k: v.mean() for k, v in ung.items()}
        ref = max(asr, key=asr.get)
        for k, v in asr.items():
            base_rows.append(dict(seed=s, method=k, asr=v))
        for f in sorted((d / "variants_protected").glob("*.csv")):
            x = pd.read_csv(f)
            h = x.set_index("url")["targeted_success"].astype(int)
            c = h.index.intersection(ung[ref].index)
            a = int(((h[c] == 1) & (ung[ref][c] == 0)).sum())
            b = int(((h[c] == 0) & (ung[ref][c] == 1)).sum())
            p = 1.0 if a + b == 0 else binomtest(a, a + b, 0.5).pvalue
            rows.append(dict(seed=s, mode=x["mode"].iloc[0], ranker=x["ranker"].iloc[0], n=len(c),
                             asr=float(h[c].mean()), ref=ref, asr_ref=float(ung[ref][c].mean()),
                             guided_only=a, ref_only=b, p=p))
    r = pd.DataFrame(rows)
    r["p_fdr"] = float("nan")
    for (s, m), g in r.groupby(["seed", "mode"]):
        r.loc[g.index, "p_fdr"] = bh(g.p.tolist())
    r.to_csv(args.out, index=False)
    print("Best protected unguided baseline per seed (ASR@10 on the 500 URLs):")
    bd = pd.DataFrame(base_rows)
    print(bd.pivot(index="method", columns="seed", values="asr").round(3).to_string())
    print()
    summ = r.groupby(["mode", "ranker"]).apply(lambda g: pd.Series(dict(
        asr_mean=g.asr.mean(), ref_mean=g.asr_ref.mean(),
        seeds_above_ref=int((g.asr > g.asr_ref).sum()),
        seeds_sig_above=int(((g.asr > g.asr_ref) & (g.p_fdr < 0.05)).sum()),
        seeds_sig_below=int(((g.asr < g.asr_ref) & (g.p_fdr < 0.05)).sum())))).round(4)
    print(summ.sort_values(["mode", "asr_mean"], ascending=[True, False]).to_string())
    summ.to_csv(str(Path(args.out).with_suffix("")) + "_summary.csv")


if __name__ == "__main__":
    main()
