"""
Relates each ranking method's faithfulness to its attack efficiency (new).

Inputs, per seed directory (results/tables/seedN):
  faithfulness_by_ranker.csv         from faithfulness/faithfulness_by_ranker.py
  variants/<ranker>_<mode>.csv       from attacks/guided_variants.py

Outputs:
  <out>               method-level table: per (seed, mode, ranker) mean Deletion AUC,
                      comprehensiveness, sufficiency, ASR@budget
  <out>_summary.csv   per mode: pooled Spearman across all (seed, ranker) points,
                      mean within-seed Spearman, and the ranker-level means over seeds
Also prints the URL-level (within-ranker) Spearman of deletion AUC vs. success.

Run (from the repo root):
    python src/analysis/faithfulness_vs_efficiency.py \
        --seed_dirs results/tables/seed0 results/tables/seed1 results/tables/seed2 \
                    results/tables/seed3 results/tables/seed4 \
        --budget 10 --out results/tables/cross_seed/faithfulness_vs_efficiency.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


PROTECTED = False


def load_seed(d):
    d = Path(d)
    sfx = "_protected" if PROTECTED else ""
    f = pd.read_csv(d / f"faithfulness_by_ranker{sfx}.csv")
    parts = [pd.read_csv(p) for p in sorted((d / f"variants{sfx}").glob("*.csv"))]
    a = pd.concat(parts, ignore_index=True)
    return f, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_dirs", nargs="+", required=True)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protected", action="store_true",
                    help="use faithfulness_by_ranker_protected.csv and variants_protected/")
    args = ap.parse_args()
    global PROTECTED
    PROTECTED = args.protected

    method_rows, url_rows = [], []
    for s, d in enumerate(args.seed_dirs):
        f, a = load_seed(d)
        a = a[a["ranker"].isin(f["ranker"].unique())]
        if "edits_used" in a.columns and "targeted_success" in a.columns:
            pass
        for (mode, ranker), g in a.groupby(["mode", "ranker"]):
            fr = f[f["ranker"] == ranker]
            m = fr.merge(g[["url", "targeted_success"]], on="url", how="inner")
            if len(m) == 0:
                continue
            method_rows.append({
                "seed": s, "mode": mode, "ranker": ranker, "n": len(m),
                "deletion_auc": m.deletion_auc.mean(),
                "comprehensiveness": m.comprehensiveness.mean(),
                "sufficiency": m.sufficiency.mean(),
                "asr": m.targeted_success.astype(float).mean(),
            })
            m = m.assign(seed=s, mode=mode, ranker=ranker)
            url_rows.append(m[["seed", "mode", "ranker", "deletion_auc", "targeted_success"]])
    mt = pd.DataFrame(method_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mt.to_csv(args.out, index=False)

    summ = []
    for mode, g in mt.groupby("mode"):
        for metric, expect in [("deletion_auc", "negative"), ("comprehensiveness", "positive"),
                               ("sufficiency", "negative")]:
            pr, pp = spearmanr(g[metric], g["asr"])
            within = [spearmanr(h[metric], h["asr"])[0] for _, h in g.groupby("seed") if len(h) >= 3]
            summ.append({"mode": mode, "metric": metric, "expected_sign": expect,
                         "pooled_rho": pr, "pooled_p": pp, "n_points": len(g),
                         "mean_within_seed_rho": sum(within) / len(within) if within else float("nan"),
                         "seeds_with_expected_sign": sum((w < 0) if expect == "negative" else (w > 0)
                                                         for w in within),
                         "n_seeds": len(within)})
    sm = pd.DataFrame(summ)
    sm.to_csv(str(Path(args.out).with_suffix("")) + "_summary.csv", index=False)

    print("\n=== ranker means over seeds ===")
    print(mt.groupby(["mode", "ranker"])[["deletion_auc", "comprehensiveness", "sufficiency", "asr"]]
          .mean().round(4).sort_values(["mode", "asr"], ascending=[True, False]))
    print("\n=== method-level correlation with ASR ===")
    print(sm.round(4).to_string(index=False))

    u = pd.concat(url_rows, ignore_index=True)
    print("\n=== URL-level Spearman(deletion_auc, success), pooled over seeds, per ranker/mode ===")
    for (mode, ranker), g in u.groupby(["mode", "ranker"]):
        if g.targeted_success.nunique() > 1:
            r, p = spearmanr(g.deletion_auc, g.targeted_success.astype(float))
            print(f"{mode:9s} {ranker:11s} rho={r:+.3f} p={p:.3g} n={len(g)}")


if __name__ == "__main__":
    main()
