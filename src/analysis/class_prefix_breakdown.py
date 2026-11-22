"""
Per-class ASR and prefix diagnostics for the main corpus (Supplementary tables).

For every seed, setting (V0 = variants/, V1 = variants_protected/) and ranking
method, reports targeted ASR at the budget
  * per true class (phishing, malware, defacement), and
  * split by whether the original URL has a non-empty prefix (scheme and/or a
    leading "www."),
and, for V0, the share of successful and of failed attacks whose final URL
changed the prefix, together with the base rate of prefixed URLs.

Usage (from the repo root):
    python src/analysis/class_prefix_breakdown.py --results_base results/tables --out per_class.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd

PREFIX = re.compile(r"^(?:[^:/?#]*://)?(?:www\.)?", re.I)


def prefix(u):
    return PREFIX.match(str(u)).group(0)


def hit(df, budget):
    return df["targeted_edits"].apply(lambda e: pd.notna(e) and e <= budget)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_base", default="results/tables")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for s in a.seeds:
        d = Path(a.results_base) / f"seed{s}"
        for setting, sub in (("V0", "variants"), ("V1", "variants_protected")):
            for p in sorted((d / sub).glob("*.csv")):
                df = pd.read_csv(p)
                df["hit"] = hit(df, a.budget)
                df["pfx"] = df["url"].map(prefix)
                df["has_prefix"] = df["pfx"].str.len() > 0
                df["prefix_changed"] = [not str(f).startswith(px) for f, px in zip(df["final_url"], df["pfx"])]
                base = dict(seed=s, setting=setting, ranker=df["ranker"].iloc[0], mode=df["mode"].iloc[0])
                rows.append(dict(base, group="all", n=len(df), asr=df.hit.mean(),
                                 share_with_prefix=df.has_prefix.mean(),
                                 prefix_changed_success=df.loc[df.hit & df.has_prefix, "prefix_changed"].mean(),
                                 prefix_changed_failure=df.loc[~df.hit & df.has_prefix, "prefix_changed"].mean()))
                for c, g in df.groupby("true_label"):
                    rows.append(dict(base, group=f"class={c}", n=len(g), asr=g.hit.mean(),
                                     share_with_prefix=g.has_prefix.mean()))
                for hp, g in df.groupby("has_prefix"):
                    rows.append(dict(base, group="prefix" if hp else "no prefix", n=len(g), asr=g.hit.mean()))
    R = pd.DataFrame(rows)
    R.to_csv(a.out, index=False)
    S = (R.groupby(["setting", "mode", "ranker", "group"])
           [["n", "asr", "share_with_prefix", "prefix_changed_success", "prefix_changed_failure"]].mean())
    S.to_csv(Path(a.out).with_name(Path(a.out).stem + "_summary.csv"))
    pd.set_option("display.width", 220)
    print(S.round(3).to_string())


if __name__ == "__main__":
    main()
