"""
Graded degradation of the attribution (new): does attack success fall as the ranking
becomes LESS faithful, when faithfulness is manipulated instead of merely observed?

The paper relates independently measured faithfulness to attack success across ranking
methods (a correlation). Here faithfulness is set by an intervention. For every URL the
Integrated Gradients (IG) ranking is blended with a random ranking:

      key(position) = (1 - w) * IG_rank_percentile + w * U(0, 1),   ranking = ascending key

w = 0 is the IG ranking (reproduces the IG re-run of guided_variants.py exactly), w = 1 is
a uniformly random ranking (the control), and w in {0.25, 0.5, 0.75} are intermediate
degradations. For each w the script records (a) the targeted ASR of the guided attack with
that ranking (the identical procedure of guided.py: same action space, validity predicate,
position cap, budget, URLs; only the ranking function changes) and (b) the Deletion AUC of
that same ranking on the clean model (definition of Section 6: top-ranked positions masked
by zeroing their embeddings, true-class probability recorded after every step, trapezoid
area on a [0, 1] axis; lower = more faithful). Under --protect_prefix the prefix positions
are excluded from the ranking and are never masked (setting V1).

Run (from src/), once per seed N, with the SAME --seed value you used for guided_variants.py:
    python -m attacks.degraded_attribution --checkpoint ../runs/runs_v1/model_seedN.pt \
        --eval_sample ../data/processed/attack_eval_sample_v1.csv \
        --outdir ../results/tables/seedN --n_urls 500 --seed 0 [--protect_prefix]
Outputs: <outdir>/degraded[_protected]/attack_w<w>_<mode>.csv and faith_w<w>.csv
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add src/ to path
from model import build_char_vocab, encode_url, PAD_IDX  # noqa: E402
from train import get_device  # noqa: E402
from attacks import guided  # noqa: E402
from attacks.unguided import load_model, CLASSES, LABEL2IDX  # noqa: E402
from attacks.guided_variants import protected_prefix, with_prefix_protection, _state  # noqa: E402

W_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


def degraded_rank_fn(w, nrng):
    base = guided.__dict__["_original_rank_positions"]  # Integrated Gradients ranking

    def rank(model, url, true_label, vocab, max_len, device, ig_steps):
        real_len = min(len(url), max_len)
        if w >= 1.0:
            return nrng.sample(range(real_len), real_len)
        order = base(model, url, true_label, vocab, max_len, device, ig_steps)
        if w <= 0.0:
            return order
        denom = max(real_len - 1, 1)
        pct = {p: i / denom for i, p in enumerate(order)}
        key = [(1.0 - w) * pct[p] + w * nrng.random() for p in range(real_len)]
        return sorted(range(real_len), key=lambda p: key[p])
    return rank


@torch.no_grad()
def deletion_auc(model, url, y, order, vocab, max_len, device):
    """Section-6 Deletion AUC for a given ranking (list of positions, most important first)."""
    n = len(order)
    if n == 0:
        return float("nan")
    ids = torch.tensor([encode_url(url, vocab, max_len)], device=device)
    pad = ids.eq(PAD_IDX)
    emb = model.embed(ids)                                   # [1, L, E]
    batch = emb.repeat(n + 1, 1, 1)                          # row k: first k positions masked
    for k in range(1, n + 1):
        batch[k:, order[k - 1], :] = 0.0
    logits = model.forward_from_embeds(batch, pad.repeat(n + 1, 1))
    probs = torch.softmax(logits, dim=-1)[:, y].cpu().numpy()
    return float(((probs[1:] + probs[:-1]) / 2.0).sum() / n)  # trapezoid on a [0, 1] axis, step 1/n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval_sample", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--levels", type=float, nargs="+", default=W_LEVELS)
    ap.add_argument("--modes", nargs="+", default=["one_shot", "adaptive"],
                    choices=["one_shot", "adaptive"])
    ap.add_argument("--max_budget", type=int, default=10)
    ap.add_argument("--n_urls", type=int, default=500)
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--ig_steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protect_prefix", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    vocab = build_char_vocab()
    model = load_model(args.checkpoint, len(vocab), args.max_len, device)
    guided.__dict__.setdefault("_original_rank_positions", guided.rank_positions)

    df = pd.read_csv(args.eval_sample)
    eligible = [(r.url, LABEL2IDX[r.type]) for r in df.itertuples()]
    if args.n_urls and len(eligible) > args.n_urls:
        eligible = random.Random(args.seed).sample(eligible, args.n_urls)  # same rule as guided.py
    print(f"{len(eligible)} URLs")

    suffix = "_protected" if args.protect_prefix else ""
    part = Path(args.outdir) / f"degraded{suffix}"
    part.mkdir(parents=True, exist_ok=True)

    original_rank = guided.rank_positions
    original_valid = guided.is_valid_url
    if args.protect_prefix:
        def valid(candidate, **kw):
            return candidate.startswith(_state["prefix"]) and original_valid(candidate, **kw)
        guided.is_valid_url = valid
    try:
        for w in args.levels:
            tag = f"w{w:g}"
            # (b) faithfulness of the degraded ranking on the clean model (one ranking per URL)
            fpath = part / f"faith_{tag}.csv"
            if not fpath.exists() or args.force:
                nrng = random.Random(f"{args.seed}-degrade-faith-{w}")
                fn = degraded_rank_fn(w, nrng)
                if args.protect_prefix:
                    fn = with_prefix_protection(fn)
                rows = []
                for url, y in eligible:
                    _state["prefix"] = protected_prefix(url) if args.protect_prefix else ""
                    order = fn(model, url, y, vocab, args.max_len, device, args.ig_steps)
                    rows.append({"url": url, "deletion_auc": deletion_auc(model, url, y, order, vocab,
                                                                         args.max_len, device)})
                pd.DataFrame(rows).to_csv(fpath, index=False)
                print(f"[{tag}] mean Deletion AUC = {np.nanmean([r['deletion_auc'] for r in rows]):.4f}")
            # (a) attack success with the degraded ranking
            for mode in args.modes:
                apath = part / f"attack_{tag}_{mode}.csv"
                if apath.exists() and not args.force:
                    print(f"[{tag}/{mode}] exists, skipping")
                    continue
                rng = random.Random(f"{args.seed}-ig-{mode}")   # same stream name as the IG re-run
                nrng = random.Random(f"{args.seed}-degrade-{w}-{mode}")
                fn = degraded_rank_fn(w, nrng)
                if args.protect_prefix:
                    fn = with_prefix_protection(fn)
                guided.rank_positions = fn
                rows = []
                for i, (url, y) in enumerate(eligible):
                    _state["prefix"] = protected_prefix(url) if args.protect_prefix else ""
                    r = guided.run_guided_attack(url, y, model, vocab, args.max_len, device, mode,
                                                 args.max_budget, rng, ig_steps=args.ig_steps)
                    r.update({"w": w, "mode": mode, "url": url, "true_label": CLASSES[y]})
                    rows.append(r)
                    if (i + 1) % 100 == 0:
                        print(f"  [{tag}/{mode}] {i + 1}/{len(eligible)}")
                pd.DataFrame(rows).to_csv(apath, index=False)
                asr = sum(r["targeted_success"] for r in rows) / len(rows)
                print(f"[{tag}/{mode}] targeted ASR@{args.max_budget} = {asr:.4f}")
    finally:
        guided.rank_positions = original_rank
        guided.is_valid_url = original_valid
        _state["prefix"] = ""


if __name__ == "__main__":
    main()
