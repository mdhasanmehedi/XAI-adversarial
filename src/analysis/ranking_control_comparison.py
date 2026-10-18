"""
Compares attribution-ranked guided attacks with the ranking-control / alternative-
attribution variants produced by attacks/guided_variants.py.

For every seed and every guidance mode (one_shot, adaptive) it reports, at the
chosen budget (default 10 edits), the targeted ASR of
    ig_original   Integrated Gradients, from the paper's guided_attack_results.csv
    <variants>    random, gradxinput, saliency, occlusion (and ig, as a reproduction check)
and two paired exact McNemar tests on the SAME URLs:
    (a) variant vs the base IG run (--base)  -> does IG beat the control / the alternative?
    (b) variant vs the seed's reference unguided baseline (highest ASR at the
        budget on the full unguided sample, as in the paper's Table 6)
Benjamini-Hochberg FDR is applied within each seed over all tests of each family
(a) and (b) separately. Results are then summarised across seeds (mean +/- SD,
number of seeds in which IG is higher, number significant after FDR).

Usage (from the repository root):
    python src/analysis/ranking_control_comparison.py \
        --seed_dirs results/tables/seed0 results/tables/seed1 results/tables/seed2 \
                    results/tables/seed3 results/tables/seed4 \
        --budget 10 --out results/tables/cross_seed/ranking_control.csv
Each seed dir needs: guided_attack_results.csv, unguided_attack_results.csv,
guided_variant_results.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ap = argparse.ArgumentParser()
ap.add_argument("--seed_dirs", nargs="+", required=True)
ap.add_argument("--budget", type=int, default=10)
ap.add_argument("--base", default="ig", choices=["ig", "ig_original"],
                help="Reference attribution run. 'ig' = the re-run of IG made by guided_variants.py "
                     "with the same random-number scheme as the variants (recommended: the attack also "
                     "draws random edit types and characters, so a re-run is the matched comparison). "
                     "'ig_original' = the paper's original guided_attack_results.csv.")
ap.add_argument("--out", default="ranking_control.csv")
args = ap.parse_args()


def hit(df, budget):
    return df["targeted_edits"].apply(lambda e: pd.notna(e) and e <= budget).astype(int)


def mcnemar_exact(a, b):
    """a, b: 0/1 arrays on the same URLs. Returns (n_a_only, n_b_only, two-sided exact p)."""
    a_only = int(((a == 1) & (b == 0)).sum())
    b_only = int(((a == 0) & (b == 1)).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else binomtest(min(a_only, b_only), n, 0.5).pvalue
    return a_only, b_only, p


def bh(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in zip(range(n, 0, -1), order[::-1]):
        prev = min(prev, p[i] * n / rank)
        adj[i] = prev
    return adj


rows = []
for s, d in enumerate(args.seed_dirs):
    d = Path(d)
    ig = pd.read_csv(d / "guided_attack_results.csv")
    ig["ranker"] = "ig_original"
    var = pd.read_csv(d / "guided_variant_results.csv")
    # Keep only the original-IG rows for the URLs that the variant runs also attacked
    # (guards against an older, larger guided_attack_results.csv).
    ig = ig[ig["url"].isin(set(var["url"]))]
    n_overlap = ig.groupby("mode")["url"].nunique().min() if len(ig) else 0
    if n_overlap < 400:
        print(f"WARNING seed {s}: only {n_overlap} URLs shared with the paper's guided run -- "
              f"the variants were probably run on the wrong evaluation sample for this seed.")
    ung = pd.read_csv(d / "unguided_attack_results.csv").rename(columns={"method": "attack"})
    ung["hit"] = hit(ung, args.budget)
    ref = ung.groupby("attack")["hit"].mean().idxmax()
    ref_hit = ung[ung.attack == ref].set_index("url")["hit"]
    allg = pd.concat([ig, var], ignore_index=True)
    seed_rows = []
    for mode in ["one_shot", "adaptive"]:
        g = allg[allg["mode"] == mode].copy()
        g["hit"] = hit(g, args.budget)
        base = g[g.ranker == args.base].set_index("url")["hit"]
        for rk, sub in g.groupby("ranker"):
            sub = sub.set_index("url")["hit"]
            common = base.index.intersection(sub.index)
            row = dict(seed=s, mode=mode, ranker=rk, n=len(sub), asr=float(sub.mean()),
                       reference_unguided=ref)
            if rk != args.base and len(common):
                a_only, b_only, p = mcnemar_exact(base[common].values, sub[common].values)
                row.update(n_common_with_base=len(common), base_only=a_only, variant_only=b_only, p_vs_base=p)
            common_r = sub.index.intersection(ref_hit.index)
            if len(common_r):
                a_only, b_only, p = mcnemar_exact(sub[common_r].values, ref_hit[common_r].values)
                row.update(n_common_with_ref=len(common_r), variant_only_vs_ref=a_only,
                           ref_only=b_only, p_vs_ref=p, asr_ref_same_urls=float(ref_hit[common_r].mean()))
            seed_rows.append(row)
    sr = pd.DataFrame(seed_rows)
    for col in ["p_vs_base", "p_vs_ref"]:
        m = sr[col].notna()
        if m.any():
            sr.loc[m, col + "_fdr"] = bh(sr.loc[m, col].values)
    rows.append(sr)

R = pd.concat(rows, ignore_index=True)
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
R.to_csv(out.with_name(out.stem + "_per_seed.csv"), index=False)

summ = []
ig0 = R[R.ranker == args.base].set_index(["seed", "mode"])["asr"]
for (mode, rk), g in R.groupby(["mode", "ranker"]):
    r = dict(mode=mode, ranker=rk, asr_mean=g.asr.mean(), asr_sd=g.asr.std(ddof=1), n_seeds=len(g))
    if rk != args.base:
        diff = [ig0[(sd, mode)] - a for sd, a in zip(g.seed, g.asr)]
        r.update(base_minus_variant_mean=float(np.mean(diff)),
                 seeds_base_higher=int(sum(x > 0 for x in diff)),
                 seeds_sig_vs_base_fdr=int((g.p_vs_base_fdr < 0.05).sum()))
    r.update(seeds_variant_above_ref=int((g.asr > g.asr_ref_same_urls).sum()),
             seeds_sig_vs_ref_fdr=int((g.p_vs_ref_fdr < 0.05).sum()))
    summ.append(r)
S = pd.DataFrame(summ).sort_values(["mode", "asr_mean"], ascending=[True, False])
S.to_csv(out, index=False)
pd.set_option("display.width", 220)
print(S.round(4).to_string(index=False))
