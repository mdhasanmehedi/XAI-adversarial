"""
Freezes ONE fixed attack-evaluation sample, reused by every attack method in
this study: unguided baselines (§5.1), attribution-guided attacks (§5.2),
surrogate transfer (§7), and defense evaluation (§8).

Why this matters: Table 3's guided-vs-unguided paired comparison, and every
McNemar's/Wilcoxon test in §11, requires the SAME set of URLs to be attacked
by every method being compared. Without a frozen sample, each script's
internal random sampling could silently drift, breaking the pairing.

Sample size default (10,000) is chosen as a rigor/compute tradeoff: large
enough for well-powered paired tests and stable bootstrap CIs on the
faithfulness-efficiency correlation (§6/§11), small enough to stay tractable
once attribution-guided attacks (far more expensive per URL than unguided
random search) and surrogate transfer are layered on top. Stratified
proportionally by true class so class-conditional ASR (§9) keeps reasonable
per-class power too.

Run (from src/):
    python -m attacks.build_eval_sample --split_dir ../data/processed/split_v1 \
        --checkpoint ../runs/runs_v1/model_seed0.pt \
        --outdir ../data/processed --n 10000 --seed 0
"""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import torch  # noqa: F401 (imported for side effects / device backend availability)

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add src/ to path
from model import build_char_vocab  # noqa: E402
from train import get_device  # noqa: E402
from attacks.unguided import load_model, predict_batch, CLASSES, LABEL2IDX, BENIGN_IDX  # noqa: E402


def build_eligible(model, test_df, vocab, max_len, device, batch_size=256):
    eligible = []
    urls = test_df["url"].tolist()
    labels = [LABEL2IDX[t] for t in test_df["type"].tolist()]
    for i in range(0, len(urls), batch_size):
        b_urls = urls[i:i + batch_size]
        b_labels = labels[i:i + batch_size]
        preds = predict_batch(model, b_urls, vocab, max_len, device)
        for u, y, p in zip(b_urls, b_labels, preds):
            if y != BENIGN_IDX and p == y:
                eligible.append((u, y))
    return eligible


def stratified_sample(eligible, n, seed):
    """Proportional stratification by true class, deterministic given seed."""
    rng = random.Random(seed)
    by_class = {}
    for u, y in eligible:
        by_class.setdefault(y, []).append((u, y))
    total = len(eligible)
    sample = []
    for y, items in by_class.items():
        k = min(round(n * len(items) / total), len(items))
        sample.extend(rng.sample(items, k))
    rng.shuffle(sample)
    return sample[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--filename", default="attack_eval_sample_v1.csv",
                     help="Output filename. Default matches the target's frozen sample -- "
                          "override this (e.g. 'surrogate_eval_sample_v1.csv') when building "
                          "an eligible-population sample for a DIFFERENT checkpoint (such as "
                          "the §7 surrogate), so it doesn't overwrite the target's sample.")
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    vocab = build_char_vocab()
    test_df = pd.read_csv(Path(args.split_dir) / "test.csv")
    model = load_model(args.checkpoint, len(vocab), args.max_len, device)

    eligible = build_eligible(model, test_df, vocab, args.max_len, device)
    n_malicious = int((test_df["type"] != "benign").sum())
    print(f"Attack-eligible (malicious + correctly classified): "
          f"{len(eligible)} / {n_malicious} malicious test URLs")

    n = min(args.n, len(eligible))
    if n < args.n:
        print(f"[warn] requested n={args.n} exceeds eligible population; using n={n}")
    sample = stratified_sample(eligible, n, args.seed)

    out_df = pd.DataFrame({"url": [u for u, _ in sample],
                            "type": [CLASSES[y] for _, y in sample]})
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / args.filename
    out_df.to_csv(out_path, index=False)

    counts = out_df["type"].value_counts().to_dict()
    print(f"\nFrozen attack-evaluation sample written to: {out_path}")
    print(f"n={len(out_df)}  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print("\nUse THIS SAME FILE (--eval_sample) for every attack method going "
          "forward: unguided baselines (re-run §5.1), attribution-guided (§5.2), "
          "surrogate transfer (§7), defense evaluation (§8). Do not regenerate "
          "this file with a different seed/n once downstream results depend on it.")


if __name__ == "__main__":
    main()
