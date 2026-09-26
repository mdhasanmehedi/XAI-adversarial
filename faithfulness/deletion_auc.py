"""
Deletion AUC (primary) + Comprehensiveness/Sufficiency (secondary), Draft §6.

Both metrics are computed ONLY on the clean, undefended classifier -- no
attack is involved, which is what keeps H3's faithfulness-vs-attack-efficiency
correlation non-circular (§6: "Do not define faithfulness from attack
success; doing so would create circularity").

Masking convention: a character position is "masked" by replacing its
embedding with the same all-zero baseline Integrated Gradients uses
(attacks/attribution.py) -- NOT by physically deleting the character from the
string. This keeps sequence length and the padding mask constant across every
masking step, isolating "how much does removing this position's information
change the prediction" from confounds like sequence-length effects. This is
deliberately different from the ATTACK's physical-edit approach (which must
also satisfy URL validity, §5.5) -- faithfulness measurement has no such
constraint, since it's a clean-model interpretability diagnostic, not a
simulated deployable attack.

Deletion AUC (primary): rank character positions by |IG attribution|
descending (same convention as the guided attack, attacks/guided.py), then
mask them one at a time in that order, recording the model's confidence in
the URL's TRUE class after each step. A faithful attribution should cause
confidence to collapse quickly -- i.e. a LOW area under this deletion curve.
AUC is computed on a normalized x-axis (fraction of positions masked, 0 to 1)
so it's comparable across URLs of different lengths.

Comprehensiveness / Sufficiency (secondary, ERASER-style, DeYoung et al. 2020),
computed at a fixed top-k FRACTION of positions (default 20%):
  - Comprehensiveness = confidence(full) - confidence(top-k MASKED). Higher
    = more faithful (removing the "important" positions hurt confidence a lot).
  - Sufficiency = confidence(full) - confidence(everything EXCEPT top-k masked,
    i.e. only the top-k positions kept). Lower = more faithful (the top-k
    alone nearly reproduce the full-input confidence).

Uses the SAME frozen eval sample as the attack scripts (--eval_sample) so
faithfulness scores can later be correlated per-URL against attack efficiency
(Table 4, §6/§11) via a valid inner join -- this script only computes and
saves faithfulness; the correlation itself is a separate downstream step.

Run (from src/):
    python -m faithfulness.deletion_auc --split_dir ../data/processed/split_v1 \
        --checkpoint ../runs/runs_v1/model_seed0.pt \
        --eval_sample ../data/processed/attack_eval_sample_v1.csv \
        --outdir ../results/tables --ig_steps 20 --n_urls 1000 \
        --topk_fraction 0.2 --seed 0
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
from attacks.attribution import integrated_gradients  # noqa: E402
from attacks.unguided import load_model, LABEL2IDX  # noqa: E402


def masked_confidence(model, embeds, pad_mask, mask_positions, target_class_idx):
    """Zeros out embeddings at mask_positions (an iterable of ints), returns
    softmax confidence for target_class_idx. mask_positions empty -> full
    (unmasked) input confidence."""
    masked_embeds = embeds.clone()
    if mask_positions:
        idx = torch.tensor(list(mask_positions), device=embeds.device, dtype=torch.long)
        masked_embeds[0, idx, :] = 0.0
    with torch.no_grad():
        logits = model.forward_from_embeds(masked_embeds, pad_mask)
        probs = torch.softmax(logits, dim=-1)
    return probs[0, target_class_idx].item()


def trapezoidal_auc_unit_interval(ys):
    """AUC of ys sampled at evenly spaced x in [0, 1] (len(ys) points,
    x_i = i/(len(ys)-1)). Used for the deletion curve."""
    n = len(ys)
    if n < 2:
        return float(ys[0]) if ys else 0.0
    xs = [i / (n - 1) for i in range(n)]
    trapz_fn = getattr(torch, "trapezoid", None)
    if trapz_fn is not None:
        return float(trapz_fn(torch.tensor(ys), torch.tensor(xs)))
    # manual fallback
    area = 0.0
    for i in range(n - 1):
        area += (ys[i] + ys[i + 1]) / 2 * (xs[i + 1] - xs[i])
    return area


def compute_faithfulness_for_url(model, url, true_label, vocab, max_len, device,
                                  ig_steps, topk_fraction):
    ids = torch.tensor([encode_url(url, vocab, max_len)], device=device)
    pad_mask = ids.eq(PAD_IDX)
    with torch.no_grad():
        embeds = model.embed(ids)

    attributions = integrated_gradients(model, ids, true_label, steps=ig_steps)
    real_len = min(len(url), max_len)
    if real_len == 0:
        return None
    ranked = sorted(range(real_len), key=lambda i: -abs(attributions[i]))

    # --- Deletion AUC (primary) ---
    confidences = [masked_confidence(model, embeds, pad_mask, set(), true_label)]
    masked_so_far = set()
    for pos in ranked:
        masked_so_far.add(pos)
        confidences.append(masked_confidence(model, embeds, pad_mask, masked_so_far, true_label))
    deletion_auc = trapezoidal_auc_unit_interval(confidences)

    # --- Comprehensiveness / Sufficiency (secondary) ---
    k = max(1, round(topk_fraction * real_len))
    top_k_positions = set(ranked[:k])
    all_positions = set(range(real_len))
    conf_full = confidences[0]
    conf_topk_masked = masked_confidence(model, embeds, pad_mask, top_k_positions, true_label)
    conf_only_topk_kept = masked_confidence(
        model, embeds, pad_mask, all_positions - top_k_positions, true_label)
    comprehensiveness = conf_full - conf_topk_masked
    sufficiency = conf_full - conf_only_topk_kept

    return {
        "url": url, "true_label": true_label, "real_len": real_len,
        "conf_full": conf_full,
        "deletion_auc": deletion_auc,
        "comprehensiveness": comprehensiveness,
        "sufficiency": sufficiency,
        "topk_fraction": topk_fraction, "topk_n": k,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval_sample", required=True,
                     help="Frozen attack_eval_sample_v1.csv -- using the SAME sample as "
                          "the attack scripts is what makes a later per-URL correlation "
                          "with attack efficiency (Table 4) valid.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ig_steps", type=int, default=20)
    ap.add_argument("--topk_fraction", type=float, default=0.2)
    ap.add_argument("--n_urls", type=int, default=1000,
                     help="Subsample this many URLs (deterministic given --seed). "
                          "0 = use the full eval sample.")
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    vocab = build_char_vocab()
    model = load_model(args.checkpoint, len(vocab), args.max_len, device)

    sample_df = pd.read_csv(args.eval_sample)
    urls = [(row.url, LABEL2IDX[row.type]) for row in sample_df.itertuples()]
    print(f"Loaded frozen eval sample: {len(urls)} URLs from {args.eval_sample}")

    if args.n_urls and len(urls) > args.n_urls:
        rng = random.Random(args.seed)
        urls = rng.sample(urls, args.n_urls)
        print(f"Subsampled to {len(urls)} URLs (--n_urls={args.n_urls}); exact subset is "
              f"recoverable from the output CSV for later per-URL correlation with attack results.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (url, true_label) in enumerate(urls):
        result = compute_faithfulness_for_url(
            model, url, true_label, vocab, args.max_len, device, args.ig_steps, args.topk_fraction)
        if result is not None:
            rows.append(result)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(urls)} scored...")
            pd.DataFrame(rows).to_csv(outdir / "faithfulness_results.csv", index=False)  # incremental save

    results_df = pd.DataFrame(rows)
    results_df.to_csv(outdir / "faithfulness_results.csv", index=False)

    print(f"\n=== Faithfulness summary (n={len(results_df)}) ===")
    print(f"Deletion AUC:      mean={results_df.deletion_auc.mean():.4f}  "
          f"median={results_df.deletion_auc.median():.4f}  std={results_df.deletion_auc.std():.4f}")
    print(f"Comprehensiveness: mean={results_df.comprehensiveness.mean():.4f}  "
          f"median={results_df.comprehensiveness.median():.4f}  std={results_df.comprehensiveness.std():.4f}")
    print(f"Sufficiency:       mean={results_df.sufficiency.mean():.4f}  "
          f"median={results_df.sufficiency.median():.4f}  std={results_df.sufficiency.std():.4f}")
    print(f"\nWritten to {outdir}/faithfulness_results.csv")
    print("\nNOTE: this computes faithfulness only. Correlating it with attack efficiency "
          "(Table 4, Spearman rho per §6/§11) requires a separate script joining this "
          "against unguided_attack_results.csv / guided_attack_results.csv by URL.")


if __name__ == "__main__":
    main()
