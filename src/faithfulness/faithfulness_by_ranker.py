"""
Faithfulness of EACH ranking method (new, post-submission).

Why
---
The ranking-control experiment (attacks/guided_variants.py) shows that attack
efficiency depends on WHICH method ranks the positions (occlusion > IG >
gradxinput ~ saliency > random). This script measures, on the clean model and
with no attack involved, how faithful each method's ordering is, so that
faithfulness can be related to attack efficiency ACROSS methods (not only
across URLs within IG, as in the original H3 analysis).

For every URL and every ranker the script computes, from that ranker's full
ordering of positions:
  deletion_auc       (lower = more faithful)  same definition as deletion_auc.py
  comprehensiveness  (higher = more faithful) top-k masked, k = 20% of positions
  sufficiency        (lower  = more faithful) only top-k kept

IMPORTANT interpretation caveat (state it in the paper): occlusion ranks
positions by the same operation (zeroing one position embedding) that the
deletion metric uses, so its high faithfulness under this metric is partly by
construction. Random ranking is the faithfulness floor. Report the metric
together with this caveat, and also report the sufficiency / comprehensiveness
results, which are less tied to single-step deletion order.

The URLs are the SAME 500 that guided_variants.py attacks (identical
subsampling rule, --seed 0), so per-URL joins with attack results are valid.

Run (from src/), once per seed N  (seed 0 uses attack_eval_sample_v1.csv):
    python -m faithfulness.faithfulness_by_ranker \
        --checkpoint ../runs/runs_v1/model_seedN.pt \
        --eval_sample ../data/processed/attack_eval_sample_seedN_v1.csv \
        --outdir ../results/tables/seedN \
        --rankers random saliency gradxinput ig occlusion \
        --n_urls 500 --ig_steps 20 --topk_fraction 0.2 --seed 0

Output: <outdir>/faithfulness_by_ranker.csv  (one row per URL x ranker)
"""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add src/ to path
from model import build_char_vocab, encode_url, PAD_IDX  # noqa: E402
from train import get_device  # noqa: E402
from attacks import guided  # noqa: E402
from attacks.guided_variants import make_ranker, RANKERS, protected_prefix  # noqa: E402
from attacks.unguided import load_model, CLASSES, LABEL2IDX  # noqa: E402
from faithfulness.deletion_auc import masked_confidence, trapezoidal_auc_unit_interval  # noqa: E402


def faithfulness_of_order(model, url, true_label, vocab, max_len, device, ranked, topk_fraction,
                          allowed=None):
    """allowed: if given, only these positions are ever masked (prefix-protected setting);
    the deletion curve then runs over the allowed positions and top-k is a fraction of them."""
    ids = torch.tensor([encode_url(url, vocab, max_len)], device=device)
    pad_mask = ids.eq(PAD_IDX)
    with torch.no_grad():
        embeds = model.embed(ids)
    real_len = min(len(url), max_len)
    if allowed is not None:
        ranked = [p for p in ranked if p in allowed]
        real_len = len(allowed)
        if real_len == 0:
            return None

    conf = [masked_confidence(model, embeds, pad_mask, set(), true_label)]
    masked = set()
    for pos in ranked:
        masked.add(pos)
        conf.append(masked_confidence(model, embeds, pad_mask, masked, true_label))
    k = max(1, round(topk_fraction * real_len))
    top = set(ranked[:k])
    allp = set(allowed) if allowed is not None else set(range(real_len))
    return {
        "real_len": real_len,
        "conf_full": conf[0],
        "deletion_auc": trapezoidal_auc_unit_interval(conf),
        "comprehensiveness": conf[0] - masked_confidence(model, embeds, pad_mask, top, true_label),
        "sufficiency": conf[0] - masked_confidence(model, embeds, pad_mask, allp - top, true_label),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval_sample", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rankers", nargs="+", default=["random", "saliency", "gradxinput", "ig", "occlusion"],
                    choices=RANKERS)
    ap.add_argument("--n_urls", type=int, default=500)
    ap.add_argument("--ig_steps", type=int, default=20)
    ap.add_argument("--topk_fraction", type=float, default=0.2)
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protect_prefix", action="store_true",
                    help="Exclude the scheme and a leading 'www.' from every ordering and from "
                         "the masking curve (matches guided_variants.py --protect_prefix). "
                         "Writes faithfulness_by_ranker_protected.csv.")
    ap.add_argument("--split_dir", default=None)  # accepted for CLI parity, unused
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    vocab = build_char_vocab()
    model = load_model(args.checkpoint, len(vocab), args.max_len, device)

    df = pd.read_csv(args.eval_sample)
    eligible = [(r.url, LABEL2IDX[r.type]) for r in df.itertuples()]
    if args.n_urls and len(eligible) > args.n_urls:
        eligible = random.Random(args.seed).sample(eligible, args.n_urls)  # same rule as guided.py
    print(f"{len(eligible)} URLs")

    guided.__dict__.setdefault("_original_rank_positions", guided.rank_positions)
    rows = []
    for name in args.rankers:
        rng = random.Random(f"{args.seed}-faith-{name}")
        rank_fn = make_ranker(name, rng)
        for i, (url, y) in enumerate(eligible):
            if not url:
                continue
            ranked = rank_fn(model, url, y, vocab, args.max_len, device, args.ig_steps)
            allowed = None
            if args.protect_prefix:
                k = len(protected_prefix(url))
                allowed = set(range(k, min(len(url), args.max_len)))
            r = faithfulness_of_order(model, url, y, vocab, args.max_len, device, ranked,
                                      args.topk_fraction, allowed)
            if r is None:
                continue
            r.update({"ranker": name, "url": url, "true_label": CLASSES[y]})
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"  [{name}] {i + 1}/{len(eligible)}")

    out = pd.DataFrame(rows)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = "faithfulness_by_ranker_protected.csv" if args.protect_prefix else "faithfulness_by_ranker.csv"
    out.to_csv(outdir / fname, index=False)
    print("\n=== mean faithfulness by ranker ===")
    print(out.groupby("ranker")[["deletion_auc", "comprehensiveness", "sufficiency"]].mean().round(4))
    print(f"\nWritten to {outdir / fname}")


if __name__ == "__main__":
    main()
