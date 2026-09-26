"""
Dataset & split protocol per Draft §4.1.

Input: a CSV with columns ['url', 'type'] where type in {benign, phishing, malware, defacement}
       (this matches the standard 651,191-URL Kaggle malicious-URL dataset).

What this does, in order:
  1. Exact-duplicate removal on the raw URL string.
  2. Near-duplicate grouping (registered domain + path template) so near-duplicates
     land in the same split -> prevents leakage across train/val/test.
  3. Stratified 70/10/20 train/val/test split by class label, respecting the groups.
  4. Saves the frozen split to disk (three CSVs + a manifest) so every model seed
     trains/evals on byte-identical data.

Run:
    python data_prep.py --input raw_urls.csv --outdir split_v1
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from sklearn.model_selection import train_test_split


CLASSES = ["benign", "phishing", "malware", "defacement"]


def registered_domain(url: str) -> str:
    """Coarse registered-domain extraction (host minus leading subdomains).
    Not PSL-aware; good enough for grouping duplicates, not for security decisions."""
    try:
        host = urlsplit(url if "://" in url else "http://" + url).netloc.split(":")[0].lower()
    except Exception:
        host = url.lower()
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def path_template(url: str) -> str:
    """Collapse digits/hex-like tokens in the path so near-duplicate paths
    (e.g. /invoice/1029 vs /invoice/4471) group together."""
    try:
        path = urlsplit(url if "://" in url else "http://" + url).path
    except Exception:
        path = ""
    return re.sub(r"[0-9a-fA-F]{4,}|\d+", "#", path)


def group_key(url: str) -> str:
    key = registered_domain(url) + "|" + path_template(url)
    return hashlib.sha1(key.encode("utf-8", "ignore")).hexdigest()


def load_and_dedup(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert {"url", "type"}.issubset(df.columns), f"expected columns url,type; got {list(df.columns)}"
    df["type"] = df["type"].str.lower().str.strip()
    df = df[df["type"].isin(CLASSES)].copy()

    before = len(df)
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    print(f"[dedup] exact duplicates removed: {before - len(df)} (kept {len(df)})")

    df["group"] = df["url"].map(group_key)
    df["url_len"] = df["url"].str.len()
    return df


def _safe_split(groups, labels, train_size, seed):
    """Stratified split with a graceful fallback: if any class has fewer than 2
    groups (can happen for minority classes after near-duplicate grouping),
    fall back to an unstratified split for that call and warn loudly, so the
    pipeline never silently drops a class."""
    try:
        return train_test_split(groups, labels, train_size=train_size,
                                 random_state=seed, stratify=labels)
    except ValueError as e:
        print(f"[warn] stratified split failed ({e}); falling back to unstratified "
              f"split for this partition. Check class balance in manifest.json.")
        return train_test_split(groups, labels, train_size=train_size, random_state=seed)


def grouped_stratified_split(df: pd.DataFrame, seed: int = 0,
                              train_frac=0.7, val_frac=0.1):
    """Split at the GROUP level, stratified by the group's majority class,
    so no group straddles two splits (prevents near-duplicate leakage)."""
    group_labels = df.groupby("group")["type"].agg(lambda s: s.value_counts().idxmax())
    groups = group_labels.index.to_numpy()
    labels = group_labels.to_numpy()

    train_groups, temp_groups, train_y, temp_y = _safe_split(groups, labels, train_frac, seed)
    val_size_of_temp = val_frac / (1 - train_frac)
    val_groups, test_groups, _, _ = _safe_split(temp_groups, temp_y, val_size_of_temp, seed)

    train_df = df[df["group"].isin(train_groups)].drop(columns="group")
    val_df = df[df["group"].isin(val_groups)].drop(columns="group")
    test_df = df[df["group"].isin(test_groups)].drop(columns="group")
    return train_df, val_df, test_df


def report(name, d: pd.DataFrame):
    counts = d["type"].value_counts().reindex(CLASSES).fillna(0).astype(int)
    print(f"[{name}] n={len(d)}  " + "  ".join(f"{c}={counts[c]}" for c in CLASSES))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_and_dedup(args.input)
    train_df, val_df, test_df = grouped_stratified_split(df, seed=args.seed)

    manifest = {
        "source": args.input,
        "split_seed": args.seed,
        "unique_urls_after_dedup": len(df),
        "url_length_stats": {
            "min": int(df.url_len.min()), "median": int(df.url_len.median()),
            "max": int(df.url_len.max()),
        },
        "splits": {
            "train": report("train", train_df).to_dict(),
            "val": report("val", val_df).to_dict(),
            "test": report("test", test_df).to_dict(),
        },
    }

    train_df.to_csv(outdir / "train.csv", index=False)
    val_df.to_csv(outdir / "val.csv", index=False)
    test_df.to_csv(outdir / "test.csv", index=False)
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nFrozen split written to {outdir}/ (train.csv, val.csv, test.csv, manifest.json)")
    print("This manifest.json is what goes into your paper's §4.1 reporting.")


if __name__ == "__main__":
    main()
