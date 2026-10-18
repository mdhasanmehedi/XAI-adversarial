"""
Position x character factorial experiment (new): WHERE to edit versus WHAT to write.

The rankers of guided_variants.py choose only the position of an edit and draw the
replacement character at random; HotFlip (attacks/hotflip.py) chooses both from a
first-order gradient estimate. Their success differs (paper, Table 22), but the two
differ in two things at once. This script crosses them:

    position rule   random | ig | occlusion | hotflip   (hotflip = positions ordered by
                                                         their best estimated flip gain)
    character rule  random | grad                       (grad = replacement character with
                                                         the largest estimated loss gain,
                                                         tried best-first until valid)

Everything else is held constant and identical to hotflip.py: substitution only (so the
position ranking never goes stale), same URLs (same subsampling rule), checkpoint, budget
of 10 edits, validity predicate, success rule, 10 positions tried per edit, and the
frozen prefix under --protect_prefix. The cell (hotflip, grad) IS the HotFlip attack and
must reproduce attacks/hotflip.py exactly (a built-in check for the analysis script).
Modes: one_shot (positions and gradients computed once; each position edited once) and
adaptive (recomputed before every edit).

Run (from src/), once per seed N, with the SAME --seed value you used for hotflip.py:
    python -m attacks.factorial --checkpoint ../runs/runs_v1/model_seedN.pt \
        --eval_sample ../data/processed/attack_eval_sample_v1.csv \
        --outdir ../results/tables/seedN --n_urls 500 --seed 0 [--protect_prefix]
Outputs: <outdir>/factorial[_protected]/<position>_<character>_<mode>.csv
"""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add src/ to path
from model import build_char_vocab  # noqa: E402
from train import get_device  # noqa: E402
from attacks import guided  # noqa: E402
from attacks.validity import is_valid_url  # noqa: E402
from attacks.transforms import PRINTABLE  # noqa: E402
from attacks.unguided import load_model, predict_one, CLASSES, LABEL2IDX, BENIGN_IDX  # noqa: E402
from attacks.guided_variants import make_ranker, protected_prefix  # noqa: E402
from attacks.hotflip import flip_candidates  # noqa: E402

POSITIONS = ["random", "ig", "occlusion", "hotflip"]
CHARS = ["random", "grad"]


def run_cell(url, y, model, vocab, max_len, device, pos_rule, char_rule, mode, max_budget,
             rng, rank_fn, print_ids, valid_fn, prefix_len=0, ig_steps=20, cap=10):
    current = url
    queries = 0
    untargeted_edit = targeted_edit = None
    used = set()
    positions = char_order = None
    for edit_num in range(1, max_budget + 1):
        if positions is None or mode == "adaptive":
            order = best = None
            if pos_rule == "hotflip" or char_rule == "grad":
                order, best = flip_candidates(model, current, y, vocab, max_len, device,
                                              "loss", print_ids)
            if pos_rule == "hotflip":
                positions = sorted(range(len(best)), key=lambda p: -best[p])
            else:
                positions = rank_fn(model, current, y, vocab, max_len, device, ig_steps)
            char_order = order
            if mode == "adaptive":
                used = set()
        applied = False
        tried = 0
        for pos in positions:
            if pos < prefix_len or pos in used or pos >= len(current):
                continue
            if char_rule == "grad":
                chars = [PRINTABLE[ci] for ci in char_order[pos]]
            else:
                chars = [rng.choice(PRINTABLE)]
            for ch in chars:
                if ch == current[pos]:
                    continue
                cand = current[:pos] + ch + current[pos + 1:]
                if not valid_fn(cand):
                    continue
                current = cand
                used.add(pos)
                applied = True
                break
            tried += 1
            if applied or tried >= cap:
                break
        if not applied:
            break
        pred = predict_one(model, current, vocab, max_len, device)
        queries += 1
        if untargeted_edit is None and pred != y:
            untargeted_edit = edit_num
        if pred == BENIGN_IDX:
            targeted_edit = edit_num
            break
    return {"final_url": current, "queries": queries,
            "untargeted_success": untargeted_edit is not None, "untargeted_edits": untargeted_edit,
            "targeted_success": targeted_edit is not None, "targeted_edits": targeted_edit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval_sample", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--positions", nargs="+", default=POSITIONS, choices=POSITIONS)
    ap.add_argument("--chars", nargs="+", default=CHARS, choices=CHARS)
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
    print_ids = torch.tensor([vocab[c] for c in PRINTABLE], device=device)
    guided.__dict__.setdefault("_original_rank_positions", guided.rank_positions)

    df = pd.read_csv(args.eval_sample)
    eligible = [(r.url, LABEL2IDX[r.type]) for r in df.itertuples()]
    if args.n_urls and len(eligible) > args.n_urls:
        eligible = random.Random(args.seed).sample(eligible, args.n_urls)  # same rule as guided.py
    print(f"{len(eligible)} URLs")

    suffix = "_protected" if args.protect_prefix else ""
    part = Path(args.outdir) / f"factorial{suffix}"
    part.mkdir(parents=True, exist_ok=True)

    for mode in args.modes:
        for pos_rule in args.positions:
            for char_rule in args.chars:
                path = part / f"{pos_rule}_{char_rule}_{mode}.csv"
                if path.exists() and not args.force:
                    print(f"[{pos_rule}/{char_rule}/{mode}] exists, skipping")
                    continue
                rng = random.Random(f"{args.seed}-fact-{pos_rule}-{char_rule}-{mode}")
                rank_fn = (make_ranker(pos_rule, random.Random(f"{args.seed}-fact-rank-{pos_rule}-{mode}"))
                           if pos_rule != "hotflip" else None)
                rows = []
                for i, (url, y) in enumerate(eligible):
                    prefix = protected_prefix(url) if args.protect_prefix else ""

                    def valid_fn(c, _p=prefix):
                        return c.startswith(_p) and is_valid_url(c, allow_tld_mutation=False)

                    r = run_cell(url, y, model, vocab, args.max_len, device, pos_rule, char_rule,
                                 mode, args.max_budget, rng, rank_fn, print_ids, valid_fn,
                                 prefix_len=len(prefix), ig_steps=args.ig_steps)
                    r.update({"position_rule": pos_rule, "char_rule": char_rule, "mode": mode,
                              "url": url, "true_label": CLASSES[y]})
                    rows.append(r)
                    if (i + 1) % 100 == 0:
                        print(f"  [{pos_rule}/{char_rule}/{mode}] {i + 1}/{len(eligible)}")
                pd.DataFrame(rows).to_csv(path, index=False)
                asr = sum(r["targeted_success"] for r in rows) / len(rows)
                print(f"[{pos_rule}/{char_rule}/{mode}] targeted ASR@{args.max_budget} = {asr:.4f}")


if __name__ == "__main__":
    main()
