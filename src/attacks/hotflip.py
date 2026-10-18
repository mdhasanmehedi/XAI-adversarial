"""
HotFlip-style character-flip attack adapted to the URL setting (external
published-attack baseline; new, post-submission).

Reference: Ebrahimi, Rao, Lowd & Dou (2018), "HotFlip: White-Box Adversarial
Examples for Text Classification", ACL.

What is HotFlip's idea
----------------------
For a character at position i with embedding e_i, flipping it to character c is
estimated, to first order, to change the loss L by
        dL ~= (e_c - e_i) . dL/de_i .
HotFlip evaluates this for EVERY (position, replacement character) pair from ONE
backward pass and picks the pair with the largest estimated increase of the loss
of the true label. It therefore chooses the position AND the replacement
character jointly, unlike the rankers of guided_variants.py, which choose only
the position and draw the character at random.

How it is adapted here, so that the comparison with the five rankers stays fair
-------------------------------------------------------------------------------
Same URLs (identical subsampling rule as guided.py / guided_variants.py), same
checkpoint, same maximum budget (10 edits), same validity predicate
(attacks.validity.is_valid_url, TLD pattern preserved), same success rule
(targeted = classified benign), same per-edit position cap (10 positions tried
per edit), same prefix rule with --protect_prefix (setting V1).

Differences that are NOT removed, and must be stated in the paper:
  * edit type: HotFlip's character FLIP = substitution only (no insertion,
    deletion or swap), with the replacement character chosen by the gradient
    estimate over the same 95 printable characters the other attacks draw from;
  * objective: gradient of the cross-entropy loss of the true class
    (--objective loss, HotFlip's original choice); --objective logit uses the
    true-class logit instead, the quantity the rankers attribute;
  * greedy (no beam search): one flip per edit.
Two guidance modes mirror guided.py:
  one_shot  gradients computed ONCE on the original URL; each edit takes the next
            best candidate at a position not edited before;
  adaptive  gradients recomputed on the current URL before every edit (this is
            HotFlip's usual greedy loop), positions may be edited again.

Run (from src/), once per seed N, with the SAME arguments you used for
guided_variants.py:
    python -m attacks.hotflip \
        --checkpoint ../runs/runs_v1/model_seedN.pt \
        --eval_sample ../data/processed/attack_eval_sample_v1.csv \
        --outdir ../results/tables/seedN \
        --modes one_shot adaptive --max_budget 10 --n_urls 500 --seed N
Add --protect_prefix for setting V1.

Outputs: <outdir>/hotflip/hotflip_<mode>.csv  (or hotflip_protected/ with
--protect_prefix). Existing result files are not touched or merged.
"""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add src/ to path
from model import build_char_vocab, encode_url, PAD_IDX  # noqa: E402
from train import get_device  # noqa: E402
from attacks.validity import is_valid_url  # noqa: E402
from attacks.transforms import PRINTABLE  # noqa: E402
from attacks.unguided import load_model, predict_one, CLASSES, LABEL2IDX, BENIGN_IDX  # noqa: E402
from attacks.guided_variants import protected_prefix  # noqa: E402


def flip_candidates(model, url, true_label, vocab, max_len, device, objective, print_ids):
    """One backward pass. Returns, for every real position i, the list of
    printable characters sorted by estimated gain (best first), and the best gain
    per position. gain > 0 means the first-order estimate says the flip helps the
    attacker (raises the loss / lowers the true-class logit)."""
    model.eval()
    ids = torch.tensor([encode_url(url, vocab, max_len)], device=device)
    pad_mask = ids.eq(PAD_IDX)
    real_len = min(len(url), max_len)
    with torch.no_grad():
        embeds = model.embed(ids).detach()
    embeds = embeds.clone().requires_grad_(True)
    logits = model.forward_from_embeds(embeds, pad_mask)
    if objective == "loss":
        obj = F.cross_entropy(logits, torch.tensor([true_label], device=device))
        sign = 1.0           # larger loss = better for the attacker
    else:
        obj = logits[0, true_label]
        sign = -1.0          # smaller true-class logit = better for the attacker
    g, = torch.autograd.grad(obj, embeds)
    g = g[0, :real_len]                                   # [n, E]
    e = embeds.detach()[0, :real_len]                     # [n, E]
    W = model.embed.weight.detach()[print_ids]            # [95, E]
    # estimated change of the objective for flipping position i to character c
    delta = g @ W.T - (e * g).sum(-1, keepdim=True)       # [n, 95]
    gain = sign * delta
    order = torch.argsort(gain, dim=1, descending=True).cpu().tolist()
    best = gain.max(dim=1).values.cpu().tolist()
    return order, best


def run_hotflip_attack(url, true_label, model, vocab, max_len, device, mode, max_budget,
                        objective, print_ids, valid_fn, prefix_len=0, max_positions_per_step=10):
    """ONE attack, up to max_budget edits. Mirrors guided.run_guided_attack()'s
    return shape (grad_calls replaces ig_calls)."""
    current = url
    queries = 0
    grad_calls = 0
    untargeted_edit = None
    targeted_edit = None
    used_positions = set()
    order = best = None

    for edit_num in range(1, max_budget + 1):
        if order is None or mode == "adaptive":
            order, best = flip_candidates(model, current, true_label, vocab, max_len, device,
                                          objective, print_ids)
            grad_calls += 1
            if mode == "adaptive":
                used_positions = set()
        # positions ranked by their best estimated gain (HotFlip's own ordering);
        # a position's characters are then tried best-first
        positions = sorted(range(len(best)), key=lambda p: -best[p])
        applied = False
        positions_tried = 0
        for pos in positions:
            if pos < prefix_len or pos in used_positions or pos >= len(current):
                continue
            for ci in order[pos]:
                ch = PRINTABLE[ci]
                if ch == current[pos]:
                    continue
                candidate = current[:pos] + ch + current[pos + 1:]
                if not valid_fn(candidate):
                    continue
                current = candidate
                used_positions.add(pos)
                applied = True
                break
            positions_tried += 1
            if applied or positions_tried >= max_positions_per_step:
                break
        if not applied:
            break

        pred = predict_one(model, current, vocab, max_len, device)
        queries += 1
        if untargeted_edit is None and pred != true_label:
            untargeted_edit = edit_num
        if pred == BENIGN_IDX:
            targeted_edit = edit_num
            break

    return {
        "final_url": current,
        "queries": queries,
        "grad_calls": grad_calls,
        "untargeted_success": untargeted_edit is not None,
        "untargeted_edits": untargeted_edit,
        "targeted_success": targeted_edit is not None,
        "targeted_edits": targeted_edit,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval_sample", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--modes", nargs="+", default=["one_shot", "adaptive"],
                    choices=["one_shot", "adaptive"])
    ap.add_argument("--objective", choices=["loss", "logit"], default="loss")
    ap.add_argument("--max_budget", type=int, default=10)
    ap.add_argument("--n_urls", type=int, default=500)
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protect_prefix", action="store_true",
                    help="Freeze the URL scheme and a leading 'www.' (setting V1).")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--split_dir", default=None)  # CLI parity with guided_variants.py, unused
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    vocab = build_char_vocab()
    model = load_model(args.checkpoint, len(vocab), args.max_len, device)
    print_ids = torch.tensor([vocab[c] for c in PRINTABLE], device=device)

    sample_df = pd.read_csv(args.eval_sample)
    eligible = [(r.url, LABEL2IDX[r.type]) for r in sample_df.itertuples()]
    if args.n_urls and len(eligible) > args.n_urls:
        # IDENTICAL subsampling rule to guided.py / guided_variants.py -> the same URLs.
        eligible = random.Random(args.seed).sample(eligible, args.n_urls)
    print(f"{len(eligible)} URLs (same subsample rule as guided.py)")

    suffix = "_protected" if args.protect_prefix else ""
    part = Path(args.outdir) / f"hotflip{suffix}"
    part.mkdir(parents=True, exist_ok=True)

    for mode in args.modes:
        path = part / f"hotflip_{mode}.csv"
        if path.exists() and not args.force:
            print(f"[hotflip/{mode}] exists, skipping (use --force)")
            continue
        rows = []
        for i, (url, y) in enumerate(eligible):
            prefix = protected_prefix(url) if args.protect_prefix else ""

            def valid_fn(candidate, _p=prefix):
                return candidate.startswith(_p) and is_valid_url(candidate, allow_tld_mutation=False)

            r = run_hotflip_attack(url, y, model, vocab, args.max_len, device, mode,
                                   args.max_budget, args.objective, print_ids, valid_fn,
                                   prefix_len=len(prefix))
            r.update({"ranker": "hotflip", "mode": mode, "url": url, "true_label": CLASSES[y]})
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"  [hotflip/{mode}] {i + 1}/{len(eligible)}")
        pd.DataFrame(rows).to_csv(path, index=False)
        asr = sum(r["targeted_success"] for r in rows) / len(rows)
        print(f"[hotflip/{mode}] targeted ASR@{args.max_budget} = {asr:.4f}  -> {path}")


if __name__ == "__main__":
    main()
