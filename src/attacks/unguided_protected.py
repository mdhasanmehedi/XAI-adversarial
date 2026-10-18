"""
Unguided baselines under the prefix-protected setting (companion to
guided_variants.py --protect_prefix).

Runs the SAME six unguided baselines of attacks/unguided.py (same transforms,
same validity predicate, same 25 retries, same budget) on the SAME 500 URLs the
protected guided variants attack, with one change: a candidate edit is rejected
if it changes the frozen prefix (leading '<scheme>://' if present, then a
leading 'www.'). Because the unguided methods choose positions at random and
retry on rejection, the validity wrapper is all that is needed.

Run (from src/), once per seed N (seed 0 uses attack_eval_sample_v1.csv):
    python -m attacks.unguided_protected --checkpoint ../runs/runs_v1/model_seedN.pt \
        --eval_sample ../data/processed/attack_eval_sample_seedN_v1.csv \
        --outdir ../results/tables/seedN --n_urls 500 --max_budget 10 --seed 0
Output: <outdir>/unguided_protected/<method>.csv
"""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model import build_char_vocab  # noqa: E402
from train import get_device  # noqa: E402
from attacks import unguided  # noqa: E402
from attacks.unguided import load_model, CLASSES, LABEL2IDX, METHODS  # noqa: E402
from attacks.guided_variants import protected_prefix, _state  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval_sample", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_urls", type=int, default=500)
    ap.add_argument("--max_budget", type=int, default=10)
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--split_dir", default=None)  # CLI parity, unused
    args = ap.parse_args()

    device = get_device()
    vocab = build_char_vocab()
    model = load_model(args.checkpoint, len(vocab), args.max_len, device)
    df = pd.read_csv(args.eval_sample)
    eligible = [(r.url, LABEL2IDX[r.type]) for r in df.itertuples()]
    if args.n_urls and len(eligible) > args.n_urls:
        eligible = random.Random(args.seed).sample(eligible, args.n_urls)  # same rule as guided.py
    print(f"{len(eligible)} URLs")

    part = Path(args.outdir) / "unguided_protected"
    part.mkdir(parents=True, exist_ok=True)

    original_valid = unguided.is_valid_url

    def valid(candidate, **kw):
        return candidate.startswith(_state["prefix"]) and original_valid(candidate, **kw)

    unguided.is_valid_url = valid
    try:
        for name, fn in METHODS.items():
            path = part / f"{name}.csv"
            if path.exists() and not args.force:
                print(f"[{name}] exists, skipping")
                continue
            rng = random.Random(f"{args.seed}-unguided-protected-{name}")
            rows = []
            for url, y in eligible:
                _state["prefix"] = protected_prefix(url)
                r = unguided.run_attack(url, y, model, vocab, args.max_len, device, name, fn,
                                        args.max_budget, rng)
                r.update({"method": name, "url": url, "true_label": CLASSES[y]})
                rows.append(r)
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"[{name}] targeted ASR@{args.max_budget} = "
                  f"{sum(r['targeted_success'] for r in rows) / len(rows):.4f}")
    finally:
        unguided.is_valid_url = original_valid
        _state["prefix"] = ""


if __name__ == "__main__":
    main()
