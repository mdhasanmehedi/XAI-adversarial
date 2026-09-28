"""
Builds the REAL Table 3 (guided vs. unguided paired comparison) per Draft
§11's frozen statistical plan -- not the eyeballed comparison you get from
reading table2_results.csv and table3_guided_only.csv side by side.

Why a separate script: unguided.py and guided.py each produce their own
per-URL results independently. guided.py typically runs on a SUBSAMPLE of
the frozen eval sample (attribution-guided attacks are far more expensive
per URL), so a valid paired comparison requires inner-joining both result
sets by URL first -- comparing on URLs only one method attacked would break
the pairing that McNemar's/Wilcoxon require.

Statistical plan implemented, exactly as locked in Draft §11:
  - Paired binary ASR outcomes (any method vs. the reference) -> McNemar's
    exact test (binomial test on the discordant pairs), at max_budget.
  - Paired continuous cost (edits) among URLs where BOTH methods succeeded
    -> Wilcoxon signed-rank test, with matched-pairs rank-biserial effect
    size (not just a p-value).
  - Budget-curve AUC (secondary efficiency metric, §9) via trapezoidal
    integration of ASR-targeted over the requested budget list, computed on
    the SAME joined URL subset as everything else for a fair comparison.
  - Benjamini-Hochberg FDR correction, applied separately within the McNemar
    p-value family and the Wilcoxon p-value family (not pooled together --
    they're testing different things).

Primary comparison / reference method: defaults to whichever unguided
baseline has the highest targeted ASR at max_budget in table2_results.csv
(the empirically hardest bar to beat), overridable via --reference_method.
Per §11 ("predefine the primary comparison before inspecting results"), note
in your Methods section which reference you used and why -- this script
picks a sensible default but the choice should be stated explicitly, not
left implicit.

Run (from src/), after both unguided.py and guided.py have produced results
in the same --outdir:
    python -m attacks.compare_guided_vs_unguided --outdir ../results/tables \
        --max_budget 10 --budgets 1 2 3 5 8 10
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest


def benjamini_hochberg(pvals):
    """Returns BH-adjusted (FDR) p-values, same order as input. NaNs pass through."""
    pvals = np.array(pvals, dtype=float)
    n = np.sum(~np.isnan(pvals))
    if n == 0:
        return pvals
    order = np.argsort(np.where(np.isnan(pvals), np.inf, pvals))
    ranked = pvals[order]
    adjusted = np.full_like(pvals, np.nan)
    valid_idx = [i for i in range(len(ranked)) if not np.isnan(ranked[i])]
    running_min = 1.0
    for rank_pos in reversed(range(len(valid_idx))):
        i = valid_idx[rank_pos]
        k = rank_pos + 1  # 1-indexed rank among valid p-values
        val = ranked[i] * n / k
        running_min = min(running_min, val)
        adjusted[order[i]] = running_min
    return adjusted


def rank_biserial_wilcoxon(x, y):
    """Matched-pairs rank-biserial correlation for paired samples x, y."""
    diffs = np.array(x, dtype=float) - np.array(y, dtype=float)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return None
    ranks = pd.Series(np.abs(diffs)).rank().to_numpy()
    pos = ranks[diffs > 0].sum()
    neg = ranks[diffs < 0].sum()
    return float((pos - neg) / (pos + neg))


def mcnemar_exact(only_a, only_b):
    """Exact McNemar's test via a binomial test on the discordant pairs."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    return binomtest(min(only_a, only_b), n, 0.5).pvalue


def trapezoidal_asr_auc(budgets, asr_by_budget):
    """Budget-curve AUC (§9 secondary efficiency metric): area under the
    ASR-vs-budget curve, trapezoidal rule over the (possibly unevenly
    spaced) requested budgets. Normalized by the budget range so it's
    comparable across different --budgets choices (max possible value = 1.0)."""
    b = np.array(budgets, dtype=float)
    a = np.array(asr_by_budget, dtype=float)
    order = np.argsort(b)
    b, a = b[order], a[order]
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz  # numpy >=2.0 renamed trapz -> trapezoid
    auc = trapz_fn(a, b)
    span = b[-1] - b[0]
    return float(auc / span) if span > 0 else float(a[0])


def load_results(outdir: Path):
    unguided_path = outdir / "unguided_attack_results.csv"
    guided_path = outdir / "guided_attack_results.csv"
    if not unguided_path.exists():
        raise FileNotFoundError(f"{unguided_path} not found -- run attacks.unguided first.")
    if not guided_path.exists():
        raise FileNotFoundError(f"{guided_path} not found -- run attacks.guided first.")
    unguided = pd.read_csv(unguided_path).rename(columns={"method": "name"})
    guided = pd.read_csv(guided_path).rename(columns={"mode": "name"})
    unguided["family"] = "unguided"
    guided["family"] = "guided"
    return pd.concat([unguided, guided], ignore_index=True)


def per_url_success_at_budget(df: pd.DataFrame, budget: int):
    """Boolean Series: did targeted success occur within this budget."""
    return df["targeted_edits"].apply(lambda e: pd.notna(e) and e <= budget)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True,
                     help="Directory containing BOTH unguided_attack_results.csv "
                          "and guided_attack_results.csv (i.e. your results/tables).")
    ap.add_argument("--max_budget", type=int, default=10,
                     help="Reference budget for the headline ASR-targeted / edits / "
                          "queries / McNemar / Wilcoxon comparison (one row per method).")
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10],
                     help="Budget list for the budget-curve AUC secondary metric.")
    ap.add_argument("--reference_method", default=None,
                     help="Unguided method to compare everything against. Default: "
                          "whichever unguided method has the highest targeted ASR at "
                          "--max_budget (the empirically hardest bar to beat).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    all_results = load_results(outdir)

    all_names = sorted(all_results["name"].unique())
    print(f"Methods found: {all_names}")

    # Pick the reference method if not given: best unguided ASR at max_budget.
    if args.reference_method:
        reference = args.reference_method
    else:
        unguided_names = all_results.loc[all_results.family == "unguided", "name"].unique()
        best_asr, reference = -1, None
        for name in unguided_names:
            sub = all_results[all_results.name == name]
            asr = per_url_success_at_budget(sub, args.max_budget).mean()
            if asr > best_asr:
                best_asr, reference = asr, name
        print(f"Auto-selected reference method: {reference!r} "
              f"(highest unguided ASR-targeted at budget {args.max_budget}: {best_asr:.4f})")
    if reference not in all_names:
        raise ValueError(f"Reference method {reference!r} not found among {all_names}")

    ref_df = all_results[all_results.name == reference].set_index("url")

    rows = []
    mcnemar_pvals, wilcoxon_pvals = [], []
    for name in all_names:
        m_df = all_results[all_results.name == name].set_index("url")
        # Inner join on URL -- this is what makes the comparison valid when
        # guided ran on a subsample of what unguided ran on.
        common_urls = m_df.index.intersection(ref_df.index)
        n_common = len(common_urls)
        m_sub = m_df.loc[common_urls]
        ref_sub = ref_df.loc[common_urls]

        m_succ_at_budget = per_url_success_at_budget(m_sub, args.max_budget)
        ref_succ_at_budget = per_url_success_at_budget(ref_sub, args.max_budget)
        asr_targeted = float(m_succ_at_budget.mean()) if n_common else None
        asr_ref = float(ref_succ_at_budget.mean()) if n_common else None
        delta_asr = (asr_targeted - asr_ref) if (asr_targeted is not None and name != reference) else 0.0

        # McNemar: paired binary success/fail, this method vs. reference.
        if name == reference:
            mcnemar_p = np.nan
        else:
            only_m = int((m_succ_at_budget & ~ref_succ_at_budget).sum())
            only_ref = int((~m_succ_at_budget & ref_succ_at_budget).sum())
            mcnemar_p = mcnemar_exact(only_m, only_ref)
        mcnemar_pvals.append(mcnemar_p)

        # Wilcoxon on edits, restricted to URLs where BOTH succeeded (cost is
        # undefined for failures, so comparing cost only makes sense there).
        both_succeeded = m_succ_at_budget & ref_succ_at_budget
        if name == reference or both_succeeded.sum() < 2:
            wilcoxon_p, effect_size = np.nan, None
        else:
            m_edits = m_sub.loc[both_succeeded, "targeted_edits"].to_numpy()
            ref_edits = ref_sub.loc[both_succeeded, "targeted_edits"].to_numpy()
            try:
                _, wilcoxon_p = wilcoxon(m_edits, ref_edits, zero_method="wilcox")
            except ValueError:
                wilcoxon_p = np.nan  # all differences zero -- no evidence either way
            effect_size = rank_biserial_wilcoxon(m_edits, ref_edits)
        wilcoxon_pvals.append(wilcoxon_p)

        # Budget-curve AUC on the same joined subset, for fairness.
        asr_by_budget = []
        for b in args.budgets:
            asr_by_budget.append(float(per_url_success_at_budget(m_sub, b).mean()) if n_common else 0.0)
        budget_auc = trapezoidal_asr_auc(args.budgets, asr_by_budget)

        median_edits = float(m_sub.loc[m_succ_at_budget, "targeted_edits"].median()) if m_succ_at_budget.any() else None
        median_queries = float(m_sub.loc[m_succ_at_budget, "queries"].median()) if m_succ_at_budget.any() else None

        rows.append({
            "method": name, "family": m_df["family"].iloc[0] if len(m_df) else None,
            "is_reference": name == reference, "n_common_urls": n_common,
            "asr_targeted": round(asr_targeted, 4) if asr_targeted is not None else None,
            "delta_asr_vs_reference": round(delta_asr, 4),
            "median_edits": median_edits, "median_queries": median_queries,
            "budget_auc": round(budget_auc, 4),
            "mcnemar_p": mcnemar_p, "wilcoxon_p": wilcoxon_p,
            "wilcoxon_rank_biserial_effect_size": (round(effect_size, 4) if effect_size is not None else None),
        })

    table3 = pd.DataFrame(rows)
    table3["mcnemar_p_fdr"] = benjamini_hochberg(table3["mcnemar_p"].to_numpy())
    table3["wilcoxon_p_fdr"] = benjamini_hochberg(table3["wilcoxon_p"].to_numpy())

    out_path = outdir / "table3_results.csv"
    table3.to_csv(out_path, index=False)

    print(f"\n=== Table 3: guided vs. unguided paired comparison (reference = {reference!r}, "
          f"budget={args.max_budget}) ===")
    display_cols = ["method", "family", "n_common_urls", "asr_targeted", "delta_asr_vs_reference",
                     "median_edits", "budget_auc", "mcnemar_p_fdr", "wilcoxon_p_fdr",
                     "wilcoxon_rank_biserial_effect_size"]
    print(table3[display_cols].to_string(index=False))
    print(f"\nWritten to {out_path}")
    print("\nNOTE: FDR correction (Benjamini-Hochberg) applied separately within the "
          "McNemar family and the Wilcoxon family, per §11. State the reference-method "
          "choice explicitly in your Methods section -- it was auto-selected here as the "
          "empirically strongest unguided baseline, which is a defensible default but "
          "should be confirmed as your predefined primary comparison, not left implicit.")


if __name__ == "__main__":
    main()
