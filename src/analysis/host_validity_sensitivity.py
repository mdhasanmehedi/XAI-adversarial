"""
Host-validity sensitivity analysis (Supplementary Table S7 of the manuscript).

Condition (vi) of the URL-validity predicate (attacks/validity.py) requires the
host's final label to be purely alphabetic and 2-24 characters long. Some ORIGINAL
URLs in the evaluation sample (mainly IP-address hosts, e.g. http://192.168.0.1/x)
already fail that condition, so no perturbation of them can ever be accepted
(except by the domain/path/TLD family, which is exempt). This script reports how much
targeted ASR changes if those URLs are excluded.

For every attack method and seed it computes
    full  : targeted ASR at the budget over all attacked URLs
    valid : targeted ASR at the budget over URLs whose original host satisfies the TLD pattern
and then the mean and SD across seeds and the relative gain (valid / full - 1).

Usage (run from the repository root; each seed directory must contain
unguided_attack_results.csv and guided_attack_results.csv as written by
attacks/unguided.py and attacks/guided.py):

    python src/analysis/host_validity_sensitivity.py \
        --seed_dirs results/seed0 results/seed1 results/seed2 results/seed3 results/seed4 \
        --budget 10 --out results/tables/host_validity_sensitivity.csv
"""
import argparse
import sys
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--seed_dirs", nargs="+", required=True)
ap.add_argument("--budget", type=int, default=10)
ap.add_argument("--src_dir", default=str(HERE.parent),
                help="Directory that contains the 'attacks' package (default: the parent of this script).")
ap.add_argument("--out", default="host_validity_sensitivity.csv")
args = ap.parse_args()
sys.path.insert(0, args.src_dir)
from attacks.validity import is_valid_tld_pattern  # noqa: E402


def original_host(url: str) -> str:
    """Host exactly as attacks/validity.py extracts it (authority minus userinfo and port)."""
    try:
        parts = urlsplit(url if "://" in url else "http://" + url)
    except Exception:
        return ""
    return parts.netloc.split("@")[-1].split(":")[0]


def hit(df: pd.DataFrame, budget: int) -> pd.Series:
    return df["targeted_edits"].apply(lambda e: pd.notna(e) and e <= budget)


per_seed = []
for s, d in enumerate(args.seed_dirs):
    d = Path(d)
    u = pd.read_csv(d / "unguided_attack_results.csv").rename(columns={"method": "attack"})
    g = pd.read_csv(d / "guided_attack_results.csv").rename(columns={"mode": "attack"})
    allr = pd.concat([u, g], ignore_index=True)
    allr["host_ok"] = allr["url"].map(lambda x: is_valid_tld_pattern(original_host(x)))
    for name, sub in allr.groupby("attack"):
        h = hit(sub, args.budget)
        per_seed.append(dict(seed=s, attack=name, n=len(sub), n_invalid=int((~sub.host_ok).sum()),
                             full=float(h.mean()), valid=float(h[sub.host_ok].mean())))
R = pd.DataFrame(per_seed)
R["rel_gain_pct"] = (R.valid / R.full - 1) * 100
rows = []
for name, g in R.groupby("attack"):
    rows.append(dict(attack=name, full_mean=g.full.mean(), full_sd=g.full.std(ddof=1),
                     valid_mean=g.valid.mean(), valid_sd=g.valid.std(ddof=1),
                     rel_gain_min_pct=g.rel_gain_pct.min(), rel_gain_max_pct=g.rel_gain_pct.max(),
                     n_invalid_min=int(g.n_invalid.min()), n_invalid_max=int(g.n_invalid.max())))
out = pd.DataFrame(rows).sort_values("attack")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(args.out, index=False); R.to_csv(Path(args.out).with_name(Path(args.out).stem + "_per_seed.csv"), index=False)
pd.set_option("display.width", 200)
print(out.round(4).to_string(index=False))
