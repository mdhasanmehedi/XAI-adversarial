"""
Compare the HotFlip-adapted baseline (attacks/hotflip.py) with the rankers and the
reference baseline on the SAME 500 URLs, per seed, with exact McNemar tests.

Run from the repository root (NOT src/), after attacks.hotflip has been run for
every seed (add --protected for the V1 files):
    python src/analysis/analyze_hotflip.py --seed_dirs results/tables/seed0 \
        results/tables/seed1 results/tables/seed2 results/tables/seed3 \
        results/tables/seed4 --out results/tables/cross_seed/hotflip_vs_rankers.csv
    python src/analysis/analyze_hotflip.py --protected --seed_dirs ... \
        --out results/tables/cross_seed/hotflip_vs_rankers_protected.csv

Use the SAME seed directories you pass to your other analysis scripts. For each seed
directory the script looks for the unguided per-method files in <dir>/_partial/ (or in
the parent directory's _partial/ if the seed directory has none), the ranker files in
<dir>/variants/ and the HotFlip files in <dir>/hotflip/.

Reference baseline: the unguided method with the highest targeted ASR in that seed's
full unguided results (the same rule as the paper); with --protected, the best of the
unguided_protected/ methods. Its outcome on the 500 URLs is used for the comparison.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def mcnemar_exact(a, b):
    """a, b: aligned boolean arrays (True = evasion). Returns (a_only, b_only, p)."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    if n10 + n01 == 0:
        return n10, n01, 1.0
    return n10, n01, float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(ranked, 1.0)
    return out


def load_unguided(seed_dir, protected):
    d = Path(seed_dir)
    sub = "unguided_protected" if protected else "_partial"
    folder = d / sub
    if not folder.exists() and not protected and (d.parent / "_partial").exists():
        folder = d.parent / "_partial"
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise SystemExit(f"no unguided results found in {folder}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True), folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protected", action="store_true")
    ap.add_argument("--budget", type=int, default=10)
    args = ap.parse_args()

    vsub = "variants_protected" if args.protected else "variants"
    hsub = "hotflip_protected" if args.protected else "hotflip"
    rows = []
    for sd in args.seed_dirs:
        sd = Path(sd)
        seed = sd.name
        ung, folder = load_unguided(sd, args.protected)
        ung["hit"] = ung["targeted_edits"].apply(lambda e: pd.notna(e) and e <= args.budget)
        ref_name = ung.groupby("method")["hit"].mean().idxmax()
        ref = ung[ung["method"] == ref_name].set_index("url")["hit"].groupby(level=0).first()
        for mode in ["one_shot", "adaptive"]:
            hf_path = sd / hsub / f"hotflip_{mode}.csv"
            if not hf_path.exists():
                print(f"[{seed}/{mode}] missing {hf_path}, skipped")
                continue
            hf = pd.read_csv(hf_path)
            hf["hit"] = hf["targeted_edits"].apply(lambda e: pd.notna(e) and e <= args.budget)
            hf = hf.set_index("url")["hit"].groupby(level=0).first()
            comp = {"reference_baseline(" + ref_name + ")": ref}
            for r in ["random", "saliency", "gradxinput", "ig", "occlusion"]:
                p = sd / vsub / f"{r}_{mode}.csv"
                if p.exists():
                    v = pd.read_csv(p)
                    v["hit"] = v["targeted_edits"].apply(lambda e: pd.notna(e) and e <= args.budget)
                    comp[r] = v.set_index("url")["hit"].groupby(level=0).first()
            for name, other in comp.items():
                common = hf.index.intersection(other.index)
                if len(common) == 0:
                    continue
                a, b = hf.loc[common].values, other.loc[common].values
                n10, n01, p = mcnemar_exact(a, b)
                rows.append({"seed": seed, "mode": mode, "n_urls": len(common),
                             "hotflip_asr": round(float(a.mean()), 4),
                             "compared_with": name, "other_asr": round(float(b.mean()), 4),
                             "hotflip_only": n10, "other_only": n01, "mcnemar_p": p})
    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("nothing to compare")
    res["p_BH"] = np.nan
    for mode in res["mode"].unique():
        idx = res.index[res["mode"] == mode]
        res.loc[idx, "p_BH"] = bh(res.loc[idx, "mcnemar_p"].values)
    res["p_BH"] = res["p_BH"].round(4)
    res["mcnemar_p"] = res["mcnemar_p"].round(4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(res.to_string(index=False))

    print("\nMean ASR over seeds (hotflip vs each comparator):")
    g = res.groupby(["mode", "compared_with"])[["hotflip_asr", "other_asr"]].mean().round(4)
    sig = res.assign(win=(res.hotflip_only > res.other_only) & (res.p_BH < 0.05),
                     loss=(res.hotflip_only < res.other_only) & (res.p_BH < 0.05)
                     ).groupby(["mode", "compared_with"])[["win", "loss"]].sum()
    print(g.join(sig).rename(columns={"win": "seeds_hotflip_sig_higher", "loss": "seeds_hotflip_sig_lower"}).to_string())
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
