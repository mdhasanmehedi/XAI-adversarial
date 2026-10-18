"""
Robustness analysis for the prefix-protected attacks (guided_variants.py --protect_prefix).

For each seed directory it reads variants_protected/<ranker>_<mode>.csv and reports
  * ASR@budget per ranker and mode,
  * a check that the frozen prefix (scheme + 'www.') was never changed,
  * exact McNemar tests of every ranker against the IG re-run and against random
    (paired on the same URLs; Benjamini-Hochberg FDR within seed),
  * a cross-seed summary (mean ASR, seeds where IG > random, seeds significant, etc.).

Run (repo root):
    python src/analysis/protected_prefix_comparison.py \
        --seed_dirs results/tables/seed0 ... results/tables/seed4 \
        --out results/tables/cross_seed/protected_prefix.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

PREFIX_RE = re.compile(r"^(?:[^:/?#]*://)?(?:www\.)?", re.I)


def mcnemar_exact(a, b):
    a_only = int(((a == 1) & (b == 0)).sum())
    b_only = int(((a == 0) & (b == 1)).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else binomtest(a_only, n, 0.5).pvalue
    return a_only, b_only, p


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
    ap.add_argument("--out", default="protected_prefix.csv")
    args = ap.parse_args()

    rows = []
    for s, d in enumerate(args.seed_dirs):
        parts = {}
        for f in sorted((Path(d) / "variants_protected").glob("*.csv")):
            df = pd.read_csv(f)
            ranker, mode = df["ranker"].iloc[0], df["mode"].iloc[0]
            df["hit"] = df["targeted_success"].astype(int)
            bad = (df.apply(lambda r: not r.final_url.startswith(PREFIX_RE.match(r.url).group(0)),
                            axis=1)).sum()
            if bad:
                print(f"WARNING seed{s} {ranker}/{mode}: {bad} URLs changed the protected prefix")
            parts[(mode, ranker)] = df.set_index("url")["hit"]
        for mode in sorted({m for m, _ in parts}):
            mrows = []
            for ranker in [r for m, r in parts if m == mode]:
                h = parts[(mode, ranker)]
                row = dict(seed=s, mode=mode, ranker=ranker, n=len(h), asr=float(h.mean()))
                for ref in ("ig", "random"):
                    if (mode, ref) in parts and ranker != ref:
                        c = h.index.intersection(parts[(mode, ref)].index)
                        a_only, b_only, p = mcnemar_exact(h[c].values, parts[(mode, ref)][c].values)
                        row.update({f"vs_{ref}_variant_only": a_only, f"vs_{ref}_ref_only": b_only,
                                    f"p_vs_{ref}": p})
                mrows.append(row)
            for ref in ("ig", "random"):
                idx = [i for i, r in enumerate(mrows) if f"p_vs_{ref}" in r]
                adj = bh([mrows[i][f"p_vs_{ref}"] for i in idx]) if idx else []
                for i, a in zip(idx, adj):
                    mrows[i][f"p_vs_{ref}_fdr"] = a
            rows += mrows
    per = pd.DataFrame(rows)
    per.to_csv(args.out, index=False)

    summ = []
    for (mode, ranker), g in per.groupby(["mode", "ranker"]):
        r = dict(mode=mode, ranker=ranker, n_seeds=len(g), asr_mean=g.asr.mean(), asr_sd=g.asr.std())
        if "vs_ig_variant_only" in g and ranker != "ig":
            r["seeds_ig_higher"] = int((g.vs_ig_variant_only < g.vs_ig_ref_only).sum())
            r["seeds_sig_vs_ig_fdr"] = int((g.p_vs_ig_fdr < 0.05).sum())
        if "vs_random_variant_only" in g and ranker != "random":
            r["seeds_above_random"] = int((g.vs_random_variant_only > g.vs_random_ref_only).sum())
            r["seeds_sig_vs_random_fdr"] = int((g.p_vs_random_fdr < 0.05).sum())
        summ.append(r)
    sm = pd.DataFrame(summ).sort_values(["mode", "asr_mean"], ascending=[True, False])
    sm.to_csv(str(Path(args.out).with_suffix("")) + "_summary.csv", index=False)
    print(sm.round(4).to_string(index=False))
    print("\nper-seed ASR:")
    print(per.pivot_table(index=["mode", "ranker"], columns="seed", values="asr").round(3).to_string())


if __name__ == "__main__":
    main()
