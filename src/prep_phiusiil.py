"""
Prepares the PhiUSIIL Phishing URL dataset (UCI 967; Prasad & Chandra, 2024; CC BY 4.0)
in EXACTLY the format the existing pipeline reads, so no attack, training or
faithfulness code has to change:

    split_dir/{train,val,test}.csv   columns: url,type     (type in {benign, phishing})

Choices (mirrors the Kaggle-dataset protocol of the paper, Section 4.1):
  * exact-URL deduplication (conflicting labels would be dropped; none exist)
  * near-duplicate grouping: group key = (registered domain = last two host labels,
    path template with digit / hex-like runs of >= 4 characters collapsed)
  * group-level split stratified by each group's majority label, 70/10/20, seed 0
  * label mapping: PhiUSIIL label 1 = legitimate -> "benign"; label 0 -> "phishing"

The model keeps the 4-class output head of the existing code; the classes "malware" and
"defacement" simply never occur. Nothing else needs to change. (Macro-F1 printed by
train.py is therefore over the two classes present.)

Run (from src/):
    python prep_phiusiil.py --csv ../data/raw/PhiUSIIL_Phishing_URL_Dataset.csv \
        --outdir ../data/processed/split_phiusiil --seed 0
"""
import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

_RUN = re.compile(r"[0-9a-fA-F]{4,}")


def group_key(url):
    u = url if "://" in url else "http://" + url
    try:
        p = urlsplit(u)
        host = (p.hostname or "").lower()
    except ValueError:
        host = ""
    reg = ".".join(host.split(".")[-2:])
    return reg + "|" + _RUN.sub("#", p.path if host else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = pd.read_csv(args.csv, usecols=["URL", "label"])
    n_raw = len(d)
    d["type"] = d["label"].map({1: "benign", 0: "phishing"})
    d = d.rename(columns={"URL": "url"})[["url", "type"]].dropna()
    conflict = d.groupby("url")["type"].nunique()
    d = d[~d["url"].isin(conflict[conflict > 1].index)].drop_duplicates("url").reset_index(drop=True)
    print(f"raw {n_raw} -> {len(d)} unique URLs ({n_raw - len(d)} removed); "
          f"{d['type'].value_counts().to_dict()}")

    d["g"] = d["url"].map(group_key)
    groups = defaultdict(list)
    for i, g in enumerate(d["g"]):
        groups[g].append(i)
    maj = {g: Counter(d.loc[ix, "type"]).most_common(1)[0][0] for g, ix in groups.items()}
    rng = random.Random(args.seed)
    assign = {}
    for cls in ["benign", "phishing"]:
        gl = sorted(g for g in groups if maj[g] == cls)
        rng.shuffle(gl)
        tot = sum(len(groups[g]) for g in gl)
        acc, cuts = 0, (0.7 * tot, 0.8 * tot)
        for g in gl:
            assign[g] = "train" if acc < cuts[0] else ("val" if acc < cuts[1] else "test")
            acc += len(groups[g])
    d["split"] = d["g"].map(assign)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for s in ["train", "val", "test"]:
        part = d[d["split"] == s][["url", "type"]].sample(frac=1, random_state=args.seed)
        part.to_csv(out / f"{s}.csv", index=False)
        print(s, len(part), part["type"].value_counts().to_dict())
    lens = d["url"].str.len()
    print(f"URL length: median {lens.median():.0f}, max {lens.max()}; groups {len(groups)}")


if __name__ == "__main__":
    main()
