"""
Cross-seed aggregation for Tables 2, 3, and 4, per Draft §11's cross-seed
statistical plan. Combines per-seed results (each produced by
run_seed_pipeline.sh into results/tables/seed<N>/) into cross-seed summaries.
This does NOT re-run any attacks/analyses -- it only aggregates results that
already exist on disk, one folder per seed.

Table 3 methodological note: attacks.compare_guided_vs_unguided.py
auto-selects the strongest unguided baseline as its reference METHOD
per-seed, and that choice can differ across seeds (it has, in practice --
seed 0 picked random_insertion, seed 1 picked domain_path_tld_transform,
seed 2 picked random_substitution). Pooling delta_asr_vs_reference directly
across seeds would therefore silently compare against a MOVING baseline in
each seed -- a real methodological error, not a formatting detail. This
script instead aggregates raw asr_targeted per method (always comparable
across seeds regardless of that seed's reference choice) and separately
reports which reference each seed picked, as a transparency item.

Table 4: with only 3-5 seeds, a formal cross-seed correlation test has
essentially no statistical power -- reporting a p-value from N=3 points
would be misleading. The primary, actually useful cross-seed output here is
a REPLICATION SUMMARY: for each method, how many seeds show a significant
pooled correlation (and in which direction), and how many show a significant
CLASS-ADJUSTED correlation (and in which direction). A secondary, heavily
caveated small-N seed-level correlation is also computed per §11's literal
instruction, but should not be over-interpreted given the sample size.

Run (from src/), once at least 2 seeds' worth of results exist:
    python aggregate_cross_seed.py --results_base ../results/tables \
        --seeds 0 1 2 3 4 --outdir ../results/tables/cross_seed
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _load_per_seed(results_base: Path, seeds, filename):
    frames = []
    missing = []
    for seed in seeds:
        path = results_base / f"seed{seed}" / filename
        if not path.exists():
            missing.append(seed)
            continue
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    return frames, missing


def load_seed_table2(results_base: Path, seeds):
    frames, missing = _load_per_seed(results_base, seeds, "table2_results.csv")
    if missing:
        print(f"[warn] Table 2: no results for seed(s) {missing} -- aggregating over "
              f"the {len(frames)} seed(s) available.")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def aggregate_table2(df: pd.DataFrame):
    rows = []
    for (method, budget), g in df.groupby(["attack", "budget"]):
        n_seeds = g["seed"].nunique()
        rows.append({
            "attack": method, "budget": budget, "n_seeds": n_seeds,
            "asr_targeted_mean": round(g["asr_targeted"].mean(), 4),
            "asr_targeted_std": round(g["asr_targeted"].std(), 4) if n_seeds > 1 else None,
            "asr_untargeted_mean": round(g["asr_untargeted"].mean(), 4),
            "asr_untargeted_std": round(g["asr_untargeted"].std(), 4) if n_seeds > 1 else None,
            "median_edits_mean": round(g["median_edits"].mean(), 2),
            "seeds_included": sorted(g["seed"].unique().tolist()),
        })
    return pd.DataFrame(rows).sort_values(["attack", "budget"])


def load_seed_table3(results_base: Path, seeds):
    frames, missing = _load_per_seed(results_base, seeds, "table3_results.csv")
    if missing:
        print(f"[warn] Table 3: no results for seed(s) {missing} -- aggregating over "
              f"the {len(frames)} seed(s) available.")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def aggregate_table3(df: pd.DataFrame):
    # Which method each seed picked as its own reference -- transparency item,
    # NOT pooled into the per-method rows below.
    ref_per_seed = (df[df["is_reference"] == True]  # noqa: E712
                    .groupby("seed")["method"].first().to_dict())

    rows = []
    for method, g in df.groupby("method"):
        n_seeds = g["seed"].nunique()
        sig_mask = g["mcnemar_p_fdr"] < 0.05
        n_sig = int(sig_mask.sum())
        sig_seeds_positive = g.loc[sig_mask & (g["delta_asr_vs_reference"] > 0), "seed"].tolist()
        sig_seeds_negative = g.loc[sig_mask & (g["delta_asr_vs_reference"] < 0), "seed"].tolist()
        rows.append({
            "method": method, "n_seeds": n_seeds,
            "asr_targeted_mean": round(g["asr_targeted"].mean(), 4),
            "asr_targeted_std": round(g["asr_targeted"].std(), 4) if n_seeds > 1 else None,
            "budget_auc_mean": round(g["budget_auc"].mean(), 4),
            "budget_auc_std": round(g["budget_auc"].std(), 4) if n_seeds > 1 else None,
            "effect_size_mean": (round(g["wilcoxon_rank_biserial_effect_size"].mean(), 4)
                                  if g["wilcoxon_rank_biserial_effect_size"].notna().any() else None),
            "n_seeds_mcnemar_sig": n_sig,
            "sig_seeds_beat_reference": sig_seeds_positive,
            "sig_seeds_underperform_reference": sig_seeds_negative,
        })
    out = pd.DataFrame(rows).sort_values("asr_targeted_mean", ascending=False)
    print(f"\n[info] Reference method auto-selected per seed (NOT pooled -- see module "
          f"docstring): {ref_per_seed}")
    return out


def load_seed_table4(results_base: Path, seeds):
    frames, missing = _load_per_seed(results_base, seeds, "table4_results.csv")
    if missing:
        print(f"[warn] Table 4: no results for seed(s) {missing} -- aggregating over "
              f"the {len(frames)} seed(s) available.")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _sig_direction(row, rho_col, lo_col, hi_col):
    """Kept for reference/backward compatibility -- no longer used in the hot
    path (see aggregate_table4). Returns '+'/'-'/None."""
    if pd.isna(row[lo_col]) or pd.isna(row[hi_col]):
        return None
    if row[lo_col] > 0:
        return "+"
    if row[hi_col] < 0:
        return "-"
    return None


def _count_significant(g: pd.DataFrame, lo_col: str, hi_col: str):
    """Vectorized, boolean-only significance counting -- deliberately avoids
    the (fragile) pattern of returning a string sentinel from a per-row
    function and then comparing strings, since that path showed inconsistent
    behavior in practice (one method's count silently came out wrong while a
    structurally identical method's count was correct, in the same run --
    consistent with a pandas dtype-inference edge case in the object-dtype
    apply() result, though the exact trigger wasn't pinned down). Booleans
    computed directly with .gt()/.lt()/.notna() have no such ambiguity."""
    lo = g[lo_col]
    hi = g[hi_col]
    valid = lo.notna() & hi.notna()
    positive = valid & (lo > 0)
    negative = valid & (hi < 0)
    return int(positive.sum()), int(negative.sum())  # CI includes 0, not significant


def aggregate_table4(df: pd.DataFrame, debug=False):
    rows = []
    for method, g in df.groupby("group"):
        n_seeds = g["seed"].nunique()
        pooled_pos, pooled_neg = _count_significant(g, "ci95_lo_deletion_auc", "ci95_hi_deletion_auc")
        adj_pos, adj_neg = _count_significant(g, "class_adjusted_ci95_lo", "class_adjusted_ci95_hi")
        if debug:
            print(f"\n[debug] method={method!r}")
            print(g[["seed", "class_adjusted_rho_deletion_auc", "class_adjusted_ci95_lo",
                      "class_adjusted_ci95_hi"]].to_string(index=False))
            print(f"[debug] adjusted: pos={adj_pos} neg={adj_neg} "
                  f"(computed via vectorized boolean masks, not per-row string comparison)")
        rows.append({
            "method": method, "n_seeds": n_seeds,
            "pooled_rho_mean": round(g["spearman_rho_deletion_auc_vs_efficiency"].mean(), 4),
            "n_seeds_pooled_sig_positive": pooled_pos,
            "n_seeds_pooled_sig_negative": pooled_neg,
            "adjusted_rho_mean": (round(g["class_adjusted_rho_deletion_auc"].mean(), 4)
                                   if g["class_adjusted_rho_deletion_auc"].notna().any() else None),
            "n_seeds_adjusted_sig_positive": adj_pos,
            "n_seeds_adjusted_sig_negative": adj_neg,
        })
    replication = pd.DataFrame(rows).sort_values("method")

    # Secondary: low-power seed-level correlation (§11's literal instruction).
    # One point per seed: that seed's overall mean deletion_auc (from any
    # unguided method's row -- they share the same faithfulness sample within
    # a seed, so this value is ~constant across unguided methods) vs. that
    # seed's mean ASR across unguided methods.
    seed_level = []
    for seed, g in df.groupby("seed"):
        unguided = g[~g["group"].isin(["one_shot", "adaptive"])]
        if len(unguided) == 0:
            continue
        seed_level.append({
            "seed": seed,
            "mean_deletion_auc": unguided["deletion_auc_mean"].mean(),
            "mean_asr": unguided["asr"].mean(),
        })
    seed_level_df = pd.DataFrame(seed_level)
    seed_corr, seed_corr_p, n_used = (None, None, 0)
    if len(seed_level_df) >= 3:
        clean = seed_level_df.dropna(subset=["mean_deletion_auc", "mean_asr"])
        n_dropped = len(seed_level_df) - len(clean)
        if n_dropped > 0:
            dropped_seeds = seed_level_df.loc[
                ~seed_level_df["seed"].isin(clean["seed"]), "seed"].tolist()
            print(f"[warn] {n_dropped} seed(s) {dropped_seeds} missing deletion_auc_mean or asr "
                  f"for this correlation -- excluded, not silently propagated as NaN.")
        n_used = len(clean)
        if n_used >= 3:
            seed_corr, seed_corr_p = spearmanr(clean["mean_deletion_auc"], clean["mean_asr"])
        else:
            print(f"[warn] Only {n_used} seed(s) have complete data -- below the minimum of 3 "
                  f"needed for even a nominal correlation. Not computed.")

    return replication, seed_level_df, seed_corr, seed_corr_p, n_used


def load_seed_table6(results_base: Path, seeds):
    frames, missing = _load_per_seed(results_base, seeds, "table6_results.csv")
    if missing:
        print(f"[warn] Table 6: no results for seed(s) {missing} -- aggregating over "
              f"the {len(frames)} seed(s) available.")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def aggregate_table6(df: pd.DataFrame):
    rows = []
    for defense, g in df.groupby("defense"):
        n_seeds = g["seed"].nunique()
        rows.append({
            "defense": defense, "n_seeds": n_seeds,
            "clean_acc_mean": round(g["clean_acc"].mean(), 4),
            "clean_acc_std": round(g["clean_acc"].std(), 4) if n_seeds > 1 else None,
            "robust_acc_mean": round(g["robust_acc_vs_guided"].mean(), 4),
            "robust_acc_std": round(g["robust_acc_vs_guided"].std(), 4) if n_seeds > 1 else None,
            "asr_mean": round(g["asr_vs_guided"].mean(), 4),
            "asr_std": round(g["asr_vs_guided"].std(), 4) if n_seeds > 1 else None,
            "seeds_included": sorted(g["seed"].unique().tolist()),
        })
    return pd.DataFrame(rows).sort_values("asr_mean")


def load_seed_table7(results_base: Path, seeds):
    frames, missing = _load_per_seed(results_base, seeds, "table7_results.csv")
    if missing:
        print(f"[warn] Table 7: no results for seed(s) {missing} -- aggregating over "
              f"the {len(frames)} seed(s) available.")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def aggregate_table7(df: pd.DataFrame):
    rows = []
    for defense, g in df.groupby("defense"):
        n_seeds = g["seed"].nunique()
        rows.append({
            "defense": defense, "n_seeds": n_seeds,
            "faithfulness_before_mean": round(g["faithfulness_before"].mean(), 4),
            "faithfulness_after_mean": round(g["faithfulness_after"].mean(), 4),
            "change_mean": round(g["change"].mean(), 4),
            "change_std": round(g["change"].std(), 4) if n_seeds > 1 else None,
            "seeds_included": sorted(g["seed"].unique().tolist()),
        })
    return pd.DataFrame(rows).sort_values("change_mean")


def load_seed_table5(results_base: Path, seeds):
    frames, missing = _load_per_seed(results_base, seeds, "table5_results.csv")
    if missing:
        print(f"[warn] Table 5: no results for seed(s) {missing} -- aggregating over "
              f"the {len(frames)} seed(s) available.")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def aggregate_table5(df: pd.DataFrame):
    """Table 5 cross-seed = same surrogate, tested for transfer against
    MULTIPLE independently trained TARGET seeds (not multiple surrogates --
    that would be a different, more expensive experiment). surrogate_asr and
    n_surrogate_success are constant across all target seeds for a given
    attack (they depend only on the surrogate + its own attack run), so only
    target_transfer_asr varies by seed and is what gets a mean/SD here."""
    rows = []
    for attack, g in df.groupby("attack"):
        n_seeds = g["seed"].nunique()
        surrogate_asr_vals = g["surrogate_asr"].unique()
        if len(surrogate_asr_vals) > 1:
            print(f"[warn] Table 5, {attack!r}: surrogate_asr differs across target seeds "
                  f"({surrogate_asr_vals}) -- expected constant (same surrogate, same attack "
                  f"run). Reporting the mean, but this is worth checking -- it may mean a "
                  f"different surrogate_results_dir was used for different seeds.")
        rows.append({
            "attack": attack, "n_seeds": n_seeds,
            "surrogate_asr": round(float(g["surrogate_asr"].mean()), 4),
            "n_surrogate_success": int(g["n_surrogate_success"].mean()),
            "target_transfer_asr_mean": round(g["target_transfer_asr"].mean(), 4),
            "target_transfer_asr_std": round(g["target_transfer_asr"].std(), 4) if n_seeds > 1 else None,
            "target_transfer_asr_untargeted_mean": round(g["target_transfer_asr_untargeted"].mean(), 4),
            "target_seeds_included": sorted(g["seed"].unique().tolist()),
        })
    return pd.DataFrame(rows).sort_values("target_transfer_asr_mean", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_base", required=True,
                     help="Directory containing seed<N>/ subfolders (e.g. ../results/tables).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--debug", action="store_true",
                     help="Print per-seed raw values and computed significance direction for "
                          "every Table 4 method, to diagnose any mismatch between expected and "
                          "computed significance counts.")
    args = ap.parse_args()

    results_base = Path(args.results_base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Table 2 ---
    t2 = load_seed_table2(results_base, args.seeds)
    if t2 is not None:
        agg2 = aggregate_table2(t2)
        agg2.to_csv(outdir / "table2_cross_seed.csv", index=False)
        n2 = t2["seed"].nunique()
        print(f"\n=== Table 2 cross-seed (n_seeds={n2}) ===")
        print(agg2.to_string(index=False))
        if n2 < 3:
            print(f"[NOTE] Only {n2} seed(s) -- SD unstable below 3. Preview only.")

    # --- Table 3 ---
    t3 = load_seed_table3(results_base, args.seeds)
    if t3 is not None:
        agg3 = aggregate_table3(t3)
        agg3.to_csv(outdir / "table3_cross_seed.csv", index=False)
        n3 = t3["seed"].nunique()
        print(f"\n=== Table 3 cross-seed (n_seeds={n3}) ===")
        print(agg3.to_string(index=False))
        if n3 < 3:
            print(f"[NOTE] Only {n3} seed(s) -- SD unstable below 3. Preview only.")

    # --- Table 4 ---
    t4 = load_seed_table4(results_base, args.seeds)
    if t4 is not None:
        agg4, seed_level_df, seed_corr, seed_corr_p, n_used = aggregate_table4(t4, debug=args.debug)
        agg4.to_csv(outdir / "table4_replication_summary.csv", index=False)
        n4 = t4["seed"].nunique()
        print(f"\n=== Table 4 replication summary (n_seeds={n4}) ===")
        print(agg4.to_string(index=False))
        print(f"\n=== Table 4 seed-level data (for the low-power cross-seed correlation) ===")
        print(seed_level_df.to_string(index=False))
        if seed_corr is not None:
            print(f"\nSeed-level Spearman correlation (mean deletion_auc vs. mean ASR, "
                  f"N={n_used} seeds with complete data): rho={seed_corr:.4f}, p={seed_corr_p:.4f}")
            print(f"[CAVEAT] N={n_used} gives this test essentially no power -- "
                  f"report descriptively, do not treat the p-value as evidence either way.")
        else:
            print(f"\n[NOTE] Fewer than 3 seeds with complete data -- seed-level correlation not computed.")
        seed_level_df.to_csv(outdir / "table4_seed_level.csv", index=False)

    # --- Table 5 (surrogate transfer to multiple target seeds) ---
    t5 = load_seed_table5(results_base, args.seeds)
    if t5 is not None:
        agg5 = aggregate_table5(t5)
        agg5.to_csv(outdir / "table5_cross_seed.csv", index=False)
        n5 = t5["seed"].nunique()
        print(f"\n=== Table 5 cross-seed: transfer to {n5} target seed(s) ===")
        print(agg5.to_string(index=False))
        if n5 < 3:
            print(f"[NOTE] Only {n5} target seed(s) -- SD unstable below 3. Preview only.")

    # --- Table 6 (defense evaluation) ---
    t6 = load_seed_table6(results_base, args.seeds)
    if t6 is not None:
        agg6 = aggregate_table6(t6)
        agg6.to_csv(outdir / "table6_cross_seed.csv", index=False)
        n6 = t6["seed"].nunique()
        print(f"\n=== Table 6 cross-seed (n_seeds={n6}) ===")
        print(agg6.to_string(index=False))
        if n6 < 3:
            print(f"[NOTE] Only {n6} seed(s) -- SD unstable below 3. Preview only.")

    # --- Table 7 (robustness-faithfulness) ---
    t7 = load_seed_table7(results_base, args.seeds)
    if t7 is not None:
        agg7 = aggregate_table7(t7)
        agg7.to_csv(outdir / "table7_cross_seed.csv", index=False)
        n7 = t7["seed"].nunique()
        print(f"\n=== Table 7 cross-seed (n_seeds={n7}) ===")
        print(agg7.to_string(index=False))
        if n7 < 3:
            print(f"[NOTE] Only {n7} seed(s) -- SD unstable below 3. Preview only.")


if __name__ == "__main__":
    main()

