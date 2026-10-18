"""
Prepares the GramBeddings dataset (Bozkir et al., 2024; train.csv + test.csv, no header, "label,url",
label 1 = phishing, 2 = legitimate; URLs may contain commas)
in EXACTLY the format the existing pipeline reads, so no attack, training or
faithfulness code has to change:

    split_dir/{train,val,test}.csv   columns: url,type     (type in {benign, phishing})

Choices (mirrors the Kaggle-dataset protocol of the paper, Section 4.1):
  * exact-URL deduplication (conflicting labels would be dropped; none exist)
  * near-duplicate grouping: group key = (registered domain = last two host labels,
    path template with digit / hex-like runs of >= 4 characters collapsed)
  * group-level split stratified by each group's majority label, 70/10/20, seed 0
  * train.csv and test.csv are POOLED first (the authors' test split shares registered domains
    with train: ~49% of test hosts also occur in train), then re-split by group
  * each split is then randomly subsampled to the PhiUSIIL sizes (164,760 / 23,961 / 46,649) so
    training time and protocol are identical; label mapping 1 -> "phishing", 2 -> "benign"

The model keeps the 4-class output head of the existing code; the classes "malware" and
"defacement" simply never occur. Nothing else needs to change. (Macro-F1 printed by
train.py is therefore over the two classes present.)

Run (from src/):
    python prep_grambeddings.py --train ../data/raw/train.csv --test ../data/raw/test.csv \
        --outdir ../data/processed/split_gram --seed 0
"""
import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

TARGET = {"train": 164760, "val": 23961, "test": 46648}
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
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for p in (args.train, args.test):
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if line:
                    a, _, b = line.partition(",")
                    rows.append((a.strip(), b.strip()))
    d = pd.DataFrame(rows, columns=["label", "url"])
    n_raw = len(d)
    d["type"] = d["label"].map({"1": "phishing", "2": "benign"})
    d = d[["url", "type"]].dropna()
    d = d[d["url"] != ""]
    conflict = d.groupby("url")["type"].nunique()
    d = d[~d["url"].isin(conflict[conflict > 1].index)].drop_duplicates("url").reset_index(drop=True)
    print(f"raw {n_raw} -> {len(d)} unique URLs ({n_raw - len(d)} removed); "
          f"{d['type'].value_counts().to_dict()}")

    d["g"] = d["url"].map(group_key)
    groups = d.groupby("g").size().to_dict()
    maj = (d.groupby("g")["type"].agg(lambda s: s.value_counts().idxmax())).to_dict()
    rng = random.Random(args.seed)
    assign = {}
    for cls in ["benign", "phishing"]:
        gl = sorted(g for g in groups if maj[g] == cls)
        rng.shuffle(gl)
        tot = sum(groups[g] for g in gl)
        acc, cuts = 0, (0.7 * tot, 0.8 * tot)
        for g in gl:
            assign[g] = "train" if acc < cuts[0] else ("val" if acc < cuts[1] else "test")
            acc += groups[g]
    d["split"] = d["g"].map(assign)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for s in ["train", "val", "test"]:
        part = d[d["split"] == s][["url", "type"]]
        part = pd.concat([x.sample(n=min(len(x), TARGET[s] // 2), random_state=args.seed)
                          for _, x in part.groupby("type")]).sample(frac=1, random_state=args.seed)
        part.to_csv(out / f"{s}.csv", index=False)
        print(s, len(part), part["type"].value_counts().to_dict())
    lens = d["url"].str.len()
    print(f"URL length: median {lens.median():.0f}, max {lens.max()}; groups {len(groups)}")


if __name__ == "__main__":
    main()
