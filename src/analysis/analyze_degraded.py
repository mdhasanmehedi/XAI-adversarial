"""
Analyse the graded-degradation experiment (attacks/degraded_attribution.py).

Run from the repository root (NOT src/):
    python src/analysis/analyze_degraded.py --seed_dirs results/tables/seed0 results/tables/seed1 \
        results/tables/seed2 results/tables/seed3 results/tables/seed4 \
        --out results/tables/cross_seed/degraded.csv
    python src/analysis/analyze_degraded.py --protected --seed_dirs ... --out results/tables/cross_seed/degraded_protected.csv

For each guidance mode: mean Deletion AUC and mean +- SD targeted ASR at every degradation
level w (w = 0: Integrated Gradients ranking; w = 1: random ranking); within-seed Spearman
correlation between the five (Deletion AUC, ASR) points (mean over seeds, number of seeds
with a negative value); the pooled Spearman correlation over all (seed, w) points; and exact
McNemar tests of every degraded level against w = 0 (BH-corrected), as seeds significantly
lower / higher. Checks that w = 0 reproduces the IG re-run of guided_variants.py if present.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = 1.0 if n10 + n01 == 0 else float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)
    return n10, n01, p


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    r = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    out = np.empty_like(p)
    out[o] = np.minimum(r, 1.0)
    return out


def hits(path, budget):
    d = pd.read_csv(path)
    d["hit"] = d["targeted_edits"].apply(lambda e: pd.notna(e) and e <= budget)
    return d.set_index("url")["hit"].groupby(level=0).first()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protected", action="store_true")
    ap.add_argument("--budget", type=int, default=10)
    args = ap.parse_args()
    sub = "degraded_protected" if args.protected else "degraded"
    vsub = "variants_protected" if args.protected else "variants"

    pts, tests = [], []
    for sd in args.seed_dirs:
        sd = Path(sd)
        d = sd / sub
        levels = sorted({float(f.stem.split("_w")[1]) for f in d.glob("faith_w*.csv")})
        for mode in ["one_shot", "adaptive"]:
            cache = {}
            for w in levels:
                a = d / f"attack_w{w:g}_{mode}.csv"
                f = d / f"faith_w{w:g}.csv"
                if not (a.exists() and f.exists()):
                    continue
                cache[w] = hits(a, args.budget)
                pts.append({"seed": sd.name, "mode": mode, "w": w, "asr": float(cache[w].mean()),
                            "deletion_auc": float(pd.read_csv(f)["deletion_auc"].mean())})
            ig = sd / vsub / f"ig_{mode}.csv"
            if ig.exists() and 0.0 in cache:
                ref = hits(ig, args.budget)
                same = (ref.sort_index().values == cache[0.0].sort_index().values).all()
                print(f"[check] {sd.name}/{mode}: w=0 reproduces the IG re-run: {same}")
            for w, h in cache.items():
                if w == 0.0 or 0.0 not in cache:
                    continue
                i = h.index.intersection(cache[0.0].index)
                n10, n01, p = mcnemar(h.loc[i].values, cache[0.0].loc[i].values)
                tests.append({"seed": sd.name, "mode": mode, "w": w, "degraded_only": n10,
                              "ig_only": n01, "p": p})
    P, T = pd.DataFrame(pts), pd.DataFrame(tests)
    if P.empty:
        raise SystemExit("no degraded-attribution results found")
    T["p_BH"] = np.nan
    for m in T["mode"].unique():
        idx = T.index[T["mode"] == m]
        T.loc[idx, "p_BH"] = bh(T.loc[idx, "p"].values)
    T["sig_lower"] = (T.degraded_only < T.ig_only) & (T.p_BH < .05)
    T["sig_higher"] = (T.degraded_only > T.ig_only) & (T.p_BH < .05)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    P.to_csv(args.out, index=False)
    T.round(5).to_csv(str(args.out).replace(".csv", "_tests.csv"), index=False)

    for mode in ["one_shot", "adaptive"]:
        p = P[P["mode"] == mode]
        if p.empty:
            continue
        print(f"\n===== {mode}: {p.seed.nunique()} seeds =====")
        g = p.groupby("w").agg(deletion_auc=("deletion_auc", "mean"), asr_mean=("asr", "mean"),
                               asr_sd=("asr", "std")).round(4)
        print(g.to_string())
        rhos = []
        for s, x in p.groupby("seed"):
            if x.w.nunique() >= 3:
                rhos.append(spearmanr(x.deletion_auc, x.asr)[0])
        rp, pp = spearmanr(p.deletion_auc, p.asr)
        print(f"within-seed Spearman(Deletion AUC, ASR): mean {np.mean(rhos):.2f}; "
              f"{int(np.sum(np.array(rhos) < 0))}/{len(rhos)} seeds negative (expected sign)")
        print(f"pooled Spearman over {len(p)} (seed, w) points: rho = {rp:.2f}, p = {pp:.4f}")
        t = T[T["mode"] == mode].groupby("w").agg(seeds_sig_lower=("sig_lower", "sum"),
                                                   seeds_sig_higher=("sig_higher", "sum"))
        print("seeds in which the degraded ranking was significantly lower / higher than w = 0:")
        print(t.to_string())
    print(f"\nWritten to {args.out} and {str(args.out).replace('.csv', '_tests.csv')}")


if __name__ == "__main__":
    main()
