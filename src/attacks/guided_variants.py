"""
Ranking-control and alternative-attribution experiments (new, post-submission).

Purpose
-------
The guided attacks of attacks/guided.py rank character positions by
|Integrated Gradients| and perturb the top-ranked valid position at each edit.
To test whether the ATTRIBUTION SIGNAL (and not the rest of the procedure) is
responsible for the gain over the unguided baselines, this script re-runs the
IDENTICAL guided procedure -- same action space (one_shot: substitution/swap;
adaptive: all four edit types), same validity predicate, same per-edit position
cap C = 10, same budget, same evaluation URLs -- and changes ONLY the function
that ranks positions:

    ig            Integrated Gradients (reproduces guided.py; sanity check)
    random        uniformly random ranking of positions   <- ranking control
    gradxinput    gradient x input on the true-class logit (single pass)
    saliency      L2 norm of the true-class-logit gradient (single pass)
    occlusion     |drop in true-class logit| when one position's embedding is zeroed

Implementation note: the attack loop is NOT re-implemented. guided.run_guided_attack
looks up guided.rank_positions at call time, so this script temporarily swaps
that one function. Every other line of the attack is shared code.

Run (from src/), once per seed N, with the SAME arguments you used for guided.py:
    python -m attacks.guided_variants \
        --split_dir ../data/processed/split_v1 \
        --checkpoint ../runs/runs_v1/model_seedN.pt \
        --eval_sample ../data/processed/attack_eval_sample_v1.csv \
        --outdir ../results/tables/seedN \
        --rankers random gradxinput saliency occlusion ig \
        --modes one_shot adaptive --max_budget 10 --n_urls 500 --ig_steps 20 --seed N

Outputs: <outdir>/variants/<ranker>_<mode>.csv (per-URL results; resumable) and
<outdir>/guided_variant_results.csv (all merged).
"""
import argparse
import random
import re
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add src/ to path
from model import build_char_vocab, encode_url, PAD_IDX  # noqa: E402
from train import get_device  # noqa: E402
from attacks import guided  # noqa: E402
from attacks.unguided import load_model, CLASSES, LABEL2IDX  # noqa: E402

RANKERS = ["ig", "random", "gradxinput", "saliency", "occlusion"]

# --protect_prefix: the leading "<scheme>://" (if present) and a following "www." are
# frozen. No edit may change them, and positions inside them are removed from every
# ranker's ordering (so protected positions do not burn the per-edit position cap).
_PREFIX_RE = re.compile(r"^(?:[^:/?#]*://)?(?:www\.)?", re.I)
_state = {"prefix": ""}


def protected_prefix(url):
    return _PREFIX_RE.match(url).group(0)


def with_prefix_protection(rank_fn):
    def rank(model, url, true_label, vocab, max_len, device, ig_steps):
        order = rank_fn(model, url, true_label, vocab, max_len, device, ig_steps)
        k = len(_state["prefix"])
        return [p for p in order if p >= k]
    return rank


def _prep(model, url, vocab, max_len, device):
    ids = torch.tensor([encode_url(url, vocab, max_len)], device=device)
    return ids, ids.eq(PAD_IDX), min(len(url), max_len)


def _order(scores, real_len):
    """Descending |score|, ties broken toward the lower index (same as guided.py)."""
    scored = [(i, abs(float(scores[i]))) for i in range(real_len)]
    scored.sort(key=lambda t: -t[1])
    return [p for p, _ in scored]


def make_ranker(name, rng):
    """Returns a function with the same signature as guided.rank_positions."""
    if name == "ig":
        return guided.__dict__["_original_rank_positions"]

    if name == "random":
        def rank(model, url, true_label, vocab, max_len, device, ig_steps):
            real_len = min(len(url), max_len)
            return rng.sample(range(real_len), real_len)
        return rank

    if name in ("gradxinput", "saliency"):
        def rank(model, url, true_label, vocab, max_len, device, ig_steps):
            model.eval()
            ids, pad_mask, real_len = _prep(model, url, vocab, max_len, device)
            with torch.no_grad():
                embeds = model.embed(ids).detach()
            embeds = embeds.clone().requires_grad_(True)
            logit = model.forward_from_embeds(embeds, pad_mask)[0, true_label]
            grad, = torch.autograd.grad(logit, embeds)
            if name == "gradxinput":
                s = (embeds.detach() * grad).sum(-1)[0]
            else:
                s = grad.norm(dim=-1)[0]
            return _order(s.detach().cpu().tolist(), real_len)
        return rank

    if name == "occlusion":
        def rank(model, url, true_label, vocab, max_len, device, ig_steps):
            model.eval()
            ids, pad_mask, real_len = _prep(model, url, vocab, max_len, device)
            with torch.no_grad():
                embeds = model.embed(ids)                       # [1, L, E]
                base = model.forward_from_embeds(embeds, pad_mask)[0, true_label]
                batch = embeds.repeat(real_len, 1, 1)           # [real_len, L, E]
                idx = torch.arange(real_len, device=device)
                batch[idx, idx, :] = 0.0                        # zero one position per row
                logits = model.forward_from_embeds(batch, pad_mask.repeat(real_len, 1))
                drops = (base - logits[:, true_label]).cpu().tolist()
            return _order(drops, real_len)
        return rank

    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)  # kept for CLI parity with guided.py
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--eval_sample", required=True)
    ap.add_argument("--rankers", nargs="+", default=["random", "gradxinput", "saliency", "occlusion"],
                    choices=RANKERS)
    ap.add_argument("--modes", nargs="+", default=["one_shot", "adaptive"],
                    choices=["one_shot", "adaptive"])
    ap.add_argument("--max_budget", type=int, default=10)
    ap.add_argument("--ig_steps", type=int, default=20)
    ap.add_argument("--n_urls", type=int, default=500)
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protect_prefix", action="store_true",
                    help="Freeze the URL scheme and a leading 'www.' (robustness setting). "
                         "Outputs go to variants_protected/ and guided_variant_results_protected.csv.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    vocab = build_char_vocab()
    model = load_model(args.checkpoint, len(vocab), args.max_len, device)

    sample_df = pd.read_csv(args.eval_sample)
    eligible = [(r.url, LABEL2IDX[r.type]) for r in sample_df.itertuples()]
    if args.n_urls and len(eligible) > args.n_urls:
        # IDENTICAL subsampling rule to guided.py -> the same URLs as the IG results.
        eligible = random.Random(args.seed).sample(eligible, args.n_urls)
    print(f"{len(eligible)} URLs (same subsample rule as guided.py)")

    outdir = Path(args.outdir)
    suffix = "_protected" if args.protect_prefix else ""
    part = outdir / f"variants{suffix}"
    part.mkdir(parents=True, exist_ok=True)

    guided.__dict__["_original_rank_positions"] = guided.rank_positions
    original_valid = guided.is_valid_url
    if args.protect_prefix:
        def valid(candidate, **kw):
            return candidate.startswith(_state["prefix"]) and original_valid(candidate, **kw)
        guided.is_valid_url = valid
    try:
        for ranker_name in args.rankers:
            for mode in args.modes:
                path = part / f"{ranker_name}_{mode}.csv"
                if path.exists() and not args.force:
                    print(f"[{ranker_name}/{mode}] exists, skipping (use --force)")
                    continue
                # Fresh, ranker-specific RNG so results do not depend on run order.
                rng = random.Random(f"{args.seed}-{ranker_name}-{mode}")
                rank_fn = make_ranker(ranker_name, rng)
                if args.protect_prefix:
                    rank_fn = with_prefix_protection(rank_fn)
                guided.rank_positions = rank_fn
                rows = []
                for i, (url, y) in enumerate(eligible):
                    _state["prefix"] = protected_prefix(url) if args.protect_prefix else ""
                    r = guided.run_guided_attack(url, y, model, vocab, args.max_len, device,
                                                 mode, args.max_budget, rng,
                                                 ig_steps=args.ig_steps)
                    r.update({"ranker": ranker_name, "mode": mode, "url": url,
                              "true_label": CLASSES[y]})
                    rows.append(r)
                    if (i + 1) % 100 == 0:
                        print(f"  [{ranker_name}/{mode}] {i + 1}/{len(eligible)}")
                pd.DataFrame(rows).to_csv(path, index=False)
                asr = sum(r["targeted_success"] for r in rows) / len(rows)
                print(f"[{ranker_name}/{mode}] targeted ASR@{args.max_budget} = {asr:.4f}")
    finally:
        guided.rank_positions = guided.__dict__["_original_rank_positions"]
        guided.is_valid_url = original_valid

    files = sorted(part.glob("*.csv"))
    merged = outdir / f"guided_variant_results{suffix}.csv"
    pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv(merged, index=False)
    print(f"Merged {len(files)} files -> {merged}")


if __name__ == "__main__":
    main()
