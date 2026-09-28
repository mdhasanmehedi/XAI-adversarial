"""
Builds Table 4 (faithfulness-attackability analysis, Draft §6/§11): tests
H3 -- is independently measured attribution faithfulness associated with
attack efficiency?

This is a WITHIN-SEED, ACROSS-SAMPLE analysis (the seeds this checkpoint
represents = 1 so far; the full §11 plan also wants a cross-seed analysis
once multiple seeds each have their own trained model + faithfulness run +
attack run -- that's a larger undertaking for later, not blocked by this
script).

Methodology, and why it's built this way:
  - Faithfulness (deletion_auc, comprehensiveness, sufficiency) comes from
    faithfulness/deletion_auc.py, computed ONLY on the clean classifier --
    no attack involved. This is what keeps the correlation non-circular
    (§6: "Do not define faithfulness from attack success").
  - Attack efficiency, per §9, is only defined for SUCCESSFUL attacks
    (efficiency_i = 1 / (edits_i x queries_i)). This means the correlation
    is necessarily computed on the subset of URLs where the given attack
    method succeeded -- restricting to successes is a real selection-effect
    caveat (if faithfulness itself predicts success, conditioning on success
    could bias the correlation) and is reported as such, not hidden.
  - Spearman rank correlation (not Pearson), per the locked §11 plan --
    appropriate given the right-skewed faithfulness distributions observed.
  - 95% CI via bootstrap resampling of the paired (faithfulness, efficiency)
    list, per §11.

Run (from src/), after faithfulness.deletion_auc AND at least one of
attacks.unguided / attacks.guided have produced results in the same --outdir:
    python -m faithfulness.correlate_with_attacks --outdir ../results/tables \
        --n_bootstrap 2000 --seed 0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, rankdata


def bootstrap_spearman_ci(x, y, n_bootstrap=2000, seed=0, ci=95):
    """95% (or `ci`%) bootstrap CI for Spearman's rho on paired (x, y)."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    if n < 5:
        return None, None, None
    rho, _ = spearmanr(x, y)
    boot_rhos = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            boot_rhos[b] = np.nan  # degenerate resample, e.g. all-identical values
        else:
            boot_rhos[b], _ = spearmanr(xb, yb)
    lo = np.nanpercentile(boot_rhos, (100 - ci) / 2)
    hi = np.nanpercentile(boot_rhos, 100 - (100 - ci) / 2)
    return float(rho), float(lo), float(hi)


def class_adjusted_partial_spearman(x, y, classes):
    """Spearman correlation between x and y, controlling for a categorical
    confound (class), via rank-residualization: rank-transform both variables
    (the standard Spearman step), subtract each class's OWN mean rank from
    every point in that class (removing between-class differences from both
    variables), then Pearson-correlate what's left. This is the standard
    rank-based partial-correlation approach for a categorical control
    variable -- mathematically, subtracting group means from ranks removes
    exactly the between-group component of variation, so the correlation of
    the residuals reflects only the within-group (class-adjusted) association.

    Returns None if any class has too few points to estimate a stable mean,
    or if there's only one class present (nothing to adjust for)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    classes = np.asarray(classes)
    unique_classes = np.unique(classes)
    if len(unique_classes) < 2:
        return None

    x_r = rankdata(x)
    y_r = rankdata(y)
    x_resid = x_r.copy()
    y_resid = y_r.copy()
    for c in unique_classes:
        mask = classes == c
        if mask.sum() < 2:
            return None  # a class with <2 points can't contribute a stable mean
        x_resid[mask] = x_r[mask] - x_r[mask].mean()
        y_resid[mask] = y_r[mask] - y_r[mask].mean()

    if np.std(x_resid) == 0 or np.std(y_resid) == 0:
        return None
    rho, _ = pearsonr(x_resid, y_resid)
    return float(rho)


def bootstrap_class_adjusted_ci(x, y, classes, n_bootstrap=2000, seed=0, ci=95):
    rng = np.random.default_rng(seed)
    x, y, classes = np.asarray(x), np.asarray(y), np.asarray(classes)
    n = len(x)
    point_est = class_adjusted_partial_spearman(x, y, classes)
    if point_est is None or n < 8:
        return None, None, None
    boot_vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        val = class_adjusted_partial_spearman(x[idx], y[idx], classes[idx])
        if val is not None:
            boot_vals.append(val)
    if len(boot_vals) < n_bootstrap * 0.5:  # too many degenerate resamples to trust the CI
        return point_est, None, None
    lo = np.percentile(boot_vals, (100 - ci) / 2)
    hi = np.percentile(boot_vals, 100 - (100 - ci) / 2)
    return point_est, float(lo), float(hi)


def load_attack_results(outdir: Path):
    frames = []
    unguided_path = outdir / "unguided_attack_results.csv"
    guided_path = outdir / "guided_attack_results.csv"
    if unguided_path.exists():
        df = pd.read_csv(unguided_path).rename(columns={"method": "attack_name"})
        df["family"] = "unguided"
        frames.append(df)
    if guided_path.exists():
        df = pd.read_csv(guided_path).rename(columns={"mode": "attack_name"})
        df["family"] = "guided"
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            "Neither unguided_attack_results.csv nor guided_attack_results.csv found "
            f"in {outdir} -- run attacks.unguided and/or attacks.guided first.")
    return pd.concat(frames, ignore_index=True)


def analyze_one_method(name, attack_df, faith_df, n_bootstrap, seed):
    merged = faith_df.merge(attack_df, on="url", how="inner", suffixes=("", "_atk"))
    n_total = len(merged)
    if n_total == 0:
        return None, None
    # faith_df's true_label is integer-encoded; attack_df's is the readable class
    # string (suffixed _atk after the merge). Use the readable one consistently
    # for all class-grouping/adjustment below.
    merged["class"] = merged["true_label_atk"] if "true_label_atk" in merged.columns else merged["true_label"]

    success = merged["targeted_success"].astype(bool)
    n_success = int(success.sum())
    asr = n_success / n_total if n_total else None

    succ_df = merged.loc[success].copy()
    succ_df["efficiency"] = 1.0 / (succ_df["targeted_edits"] * succ_df["queries"])

    row = {
        "group": name,
        "n_total": n_total,
        "n_success": n_success,
        "asr": round(asr, 4) if asr is not None else None,
        "deletion_auc_mean": round(float(merged["deletion_auc"].mean()), 4),
        "comprehensiveness_mean": round(float(merged["comprehensiveness"].mean()), 4),
        "sufficiency_mean": round(float(merged["sufficiency"].mean()), 4),
        "median_edits_on_success": (float(succ_df["targeted_edits"].median())
                                     if n_success else None),
        "median_queries_on_success": (float(succ_df["queries"].median())
                                       if n_success else None),
    }

    by_class_rows = []
    for faith_col, label in [("deletion_auc", "deletion_auc"),
                              ("comprehensiveness", "comprehensiveness"),
                              ("sufficiency", "sufficiency")]:
        if n_success >= 5:
            rho, lo, hi = bootstrap_spearman_ci(
                succ_df[faith_col], succ_df["efficiency"], n_bootstrap=n_bootstrap, seed=seed)
        else:
            rho, lo, hi = None, None, None
        row[f"spearman_rho_{label}_vs_efficiency"] = round(rho, 4) if rho is not None else None
        row[f"ci95_lo_{label}"] = round(lo, 4) if lo is not None else None
        row[f"ci95_hi_{label}"] = round(hi, 4) if hi is not None else None

        # Class-adjusted (confound-controlled) version -- only computed for
        # the primary metric (deletion_auc) to keep the table readable; the
        # secondary metrics' pooled numbers are still reported above.
        if label == "deletion_auc" and n_success >= 8 and "class" in succ_df.columns:
            adj_rho, adj_lo, adj_hi = bootstrap_class_adjusted_ci(
                succ_df[faith_col].to_numpy(), succ_df["efficiency"].to_numpy(),
                succ_df["class"].to_numpy(), n_bootstrap=n_bootstrap, seed=seed)
            row["class_adjusted_rho_deletion_auc"] = round(adj_rho, 4) if adj_rho is not None else None
            row["class_adjusted_ci95_lo"] = round(adj_lo, 4) if adj_lo is not None else None
            row["class_adjusted_ci95_hi"] = round(adj_hi, 4) if adj_hi is not None else None
        elif label == "deletion_auc":
            row["class_adjusted_rho_deletion_auc"] = None
            row["class_adjusted_ci95_lo"] = None
            row["class_adjusted_ci95_hi"] = None

        # Per-class breakdown (deletion_auc only, same readability reasoning).
        if label == "deletion_auc" and "class" in succ_df.columns:
            for cls, cls_df in succ_df.groupby("class"):
                if len(cls_df) >= 5:
                    c_rho, c_lo, c_hi = bootstrap_spearman_ci(
                        cls_df[faith_col], cls_df["efficiency"], n_bootstrap=n_bootstrap, seed=seed)
                else:
                    c_rho, c_lo, c_hi = None, None, None
                by_class_rows.append({
                    "group": name, "class": cls, "n_success_in_class": len(cls_df),
                    "spearman_rho_deletion_auc": round(c_rho, 4) if c_rho is not None else None,
                    "ci95_lo": round(c_lo, 4) if c_lo is not None else None,
                    "ci95_hi": round(c_hi, 4) if c_hi is not None else None,
                })

    return row, by_class_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True,
                     help="Directory containing faithfulness_results.csv AND at least "
                          "one of unguided_attack_results.csv / guided_attack_results.csv.")
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=None,
                     help="Restrict to these attack method/mode names. Default: all found.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    faith_path = outdir / "faithfulness_results.csv"
    if not faith_path.exists():
        raise FileNotFoundError(f"{faith_path} not found -- run faithfulness.deletion_auc first.")
    faith_df = pd.read_csv(faith_path)
    print(f"Loaded faithfulness results: {len(faith_df)} URLs")

    all_attacks = load_attack_results(outdir)
    names = args.methods or sorted(all_attacks["attack_name"].unique())
    print(f"Attack methods found: {sorted(all_attacks['attack_name'].unique())}")
    print(f"Analyzing: {names}")

    rows = []
    by_class_all = []
    for name in names:
        attack_df = all_attacks[all_attacks.attack_name == name]
        row, by_class_rows = analyze_one_method(name, attack_df, faith_df, args.n_bootstrap, args.seed)
        if row is not None:
            rows.append(row)
            by_class_all.extend(by_class_rows)
        else:
            print(f"[warn] no overlap between faithfulness sample and {name!r}'s attacked "
                  f"URLs -- skipping (faithfulness and attack runs may need a larger overlapping "
                  f"--n_urls to produce a usable joined sample for this method).")

    table4 = pd.DataFrame(rows)
    out_path = outdir / "table4_results.csv"
    table4.to_csv(out_path, index=False)

    by_class_df = pd.DataFrame(by_class_all)
    by_class_path = outdir / "table4_by_class.csv"
    by_class_df.to_csv(by_class_path, index=False)

    print(f"\n=== Table 4: Faithfulness-attackability analysis (n_bootstrap={args.n_bootstrap}) ===")
    display_cols = ["group", "n_total", "n_success", "asr", "deletion_auc_mean",
                     "spearman_rho_deletion_auc_vs_efficiency", "ci95_lo_deletion_auc",
                     "ci95_hi_deletion_auc", "class_adjusted_rho_deletion_auc",
                     "class_adjusted_ci95_lo", "class_adjusted_ci95_hi"]
    print(table4[display_cols].to_string(index=False))
    print(f"\nWritten to {out_path}")
    print(f"Per-class breakdown written to {by_class_path}")
    print("\nCLASS-ADJUSTED COLUMNS: rank-residualize deletion_auc and efficiency by removing "
          "each true-class's own mean rank, then correlate what's left -- this is the "
          "confound-controlled version of the pooled correlation (§6.1's instruction to "
          "control for class). Compare it against the unadjusted rho: if they tell a "
          "similar story, class isn't driving the result; if the adjusted rho collapses "
          "toward 0 while the pooled one doesn't, the pooled correlation was likely "
          "confounded by class rather than reflecting a direct faithfulness-efficiency link.")
    print("\nCAVEAT (state this in the paper): the correlation is computed only on URLs "
          "where the given method SUCCEEDED, since efficiency is undefined for failures. "
          "If faithfulness itself predicts success, conditioning on success could bias this "
          "correlation -- this is a real limitation of the per-sample analysis, not an "
          "oversight, and should be named explicitly in §6.1/§14.")
    print("\nThis is a WITHIN-SEED (seed represented by the checkpoint used for both the "
          "faithfulness and attack runs), ACROSS-SAMPLE analysis. The full §11 plan also "
          "wants a CROSS-SEED analysis once multiple seeds each have their own faithfulness "
          "+ attack runs -- a larger undertaking for later.")


if __name__ == "__main__":
    main()
