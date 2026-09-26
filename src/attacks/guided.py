"""
Attribution-guided adversarial attack (Draft §5.2), with one-shot and
adaptive guidance variants (§5.3).

Attribution: Integrated Gradients (attacks/attribution.py) computed w.r.t.
the URL's TRUE (malicious) class score -- high |attribution| at a character
position means that position is currently decision-relevant to the model
believing this URL is malicious, i.e. where a guided attacker would
concentrate perturbations.

One-shot vs adaptive (§5.3):
  - one-shot: attribution computed ONCE, on the original URL. The resulting
    position ranking is then reused for the whole attack, consuming one
    (unused) ranked position per accepted edit. Because inserting or
    deleting a character shifts every later index, one-shot is restricted to
    LENGTH-PRESERVING edits (substitution, swap) so the fixed ranking stays
    valid for the whole attack -- a deliberate methodological choice, not an
    oversight (see transforms.LENGTH_PRESERVING_TARGETED_METHODS).
  - adaptive: attribution is recomputed after EVERY accepted edit, so it
    always operates on fresh positions for the CURRENT string. Free to use
    all four targeted edit types (substitution, insertion, deletion, swap).

Success criterion, TLD-mutation policy, and validity checks are identical to
the unguided baselines (§3, §5.5) -- see attacks/unguided.py for the shared
conventions this file follows.

Cost note: attribution-guided attacks are far more expensive per URL than
unguided random search -- one-shot needs 1 IG call (ig_steps forward+backward
passes) per URL; adaptive needs up to max_budget IG calls per URL. Start with
a modest --n_urls and --ig_steps before scaling up.

Run (from src/):
    python -m attacks.guided --split_dir ../data/processed/split_v1 \
        --checkpoint ../runs/runs_v1/model_seed0.pt \
        --outdir ../results/tables \
        --eval_sample ../data/processed/attack_eval_sample_v1.csv \
        --modes one_shot adaptive --max_budget 10 --budgets 1 2 3 5 8 10 \
        --ig_steps 20 --n_urls 1000 --seed 0
"""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add src/ to path
from model import build_char_vocab, encode_url  # noqa: E402
from train import get_device  # noqa: E402
from attacks.validity import is_valid_url  # noqa: E402
from attacks.transforms import TARGETED_METHODS, LENGTH_PRESERVING_TARGETED_METHODS  # noqa: E402
from attacks.attribution import integrated_gradients  # noqa: E402
from attacks.unguided import load_model, predict_one, CLASSES, LABEL2IDX, BENIGN_IDX  # noqa: E402


def rank_positions(model, url, true_label, vocab, max_len, device, ig_steps):
    ids = torch.tensor([encode_url(url, vocab, max_len)], device=device)
    attributions = integrated_gradients(model, ids, true_label, steps=ig_steps)
    real_len = min(len(url), max_len)
    scored = [(i, abs(attributions[i])) for i in range(real_len)]
    scored.sort(key=lambda t: -t[1])
    return [p for p, _ in scored]


def run_guided_attack(url, true_label, model, vocab, max_len, device, mode,
                       max_budget, rng, ig_steps=20, max_positions_per_step=10):
    """Runs ONE guided attack up to max_budget edits. Mirrors unguided.py's
    run_attack() return shape (plus ig_calls, for tracking the extra compute
    cost this method incurs vs. the unguided baselines)."""
    current = url
    queries = 0
    ig_calls = 0
    untargeted_edit = None
    targeted_edit = None

    allowed_types = (LENGTH_PRESERVING_TARGETED_METHODS if mode == "one_shot"
                      else list(TARGETED_METHODS.keys()))

    ranked = rank_positions(model, current, true_label, vocab, max_len, device, ig_steps)
    ig_calls += 1
    used_positions = set()

    for edit_num in range(1, max_budget + 1):
        if mode == "adaptive" and edit_num > 1:
            # Fresh attribution on the CURRENT (already-edited) string --
            # this is what makes adaptive guidance safe with length-changing
            # edits: positions are always recomputed against the string
            # they'll actually be applied to.
            ranked = rank_positions(model, current, true_label, vocab, max_len, device, ig_steps)
            ig_calls += 1
            used_positions = set()

        applied = False
        positions_tried = 0
        for pos in ranked:
            if pos in used_positions or pos >= len(current):
                continue
            types_order = rng.sample(allowed_types, len(allowed_types))
            for type_name in types_order:
                candidate = TARGETED_METHODS[type_name](current, pos, rng)
                if candidate is None:
                    continue
                if not is_valid_url(candidate, allow_tld_mutation=False):
                    continue
                current = candidate
                used_positions.add(pos)
                applied = True
                break
            positions_tried += 1
            if applied or positions_tried >= max_positions_per_step:
                break
        if not applied:
            break  # no valid move found near any top-ranked position; stop early

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
        "ig_calls": ig_calls,
        "untargeted_success": untargeted_edit is not None,
        "untargeted_edits": untargeted_edit,
        "targeted_success": targeted_edit is not None,
        "targeted_edits": targeted_edit,
    }


def build_table_guided(results_df: pd.DataFrame, budgets):
    rows = []
    for mode in results_df["mode"].unique():
        sub = results_df[results_df["mode"] == mode]
        n = len(sub)
        for budget in budgets:
            t_hit = sub["targeted_edits"].apply(lambda e: pd.notna(e) and e <= budget)
            u_hit = sub["untargeted_edits"].apply(lambda e: pd.notna(e) and e <= budget)
            edits_succ = sub.loc[t_hit, "targeted_edits"]
            queries_succ = sub.loc[t_hit, "queries"]
            rows.append({
                "mode": mode,
                "budget": budget,
                "asr_targeted": round(float(t_hit.mean()), 4) if n else None,
                "asr_untargeted": round(float(u_hit.mean()), 4) if n else None,
                "median_edits": float(edits_succ.median()) if len(edits_succ) else None,
                "median_queries": float(queries_succ.median()) if len(queries_succ) else None,
                "n_attacked": n,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--eval_sample", required=True,
                     help="Frozen attack_eval_sample_v1.csv from build_eval_sample.py. "
                          "Required here (not optional) -- guided results must be directly "
                          "comparable to unguided results for the eventual Table 3.")
    ap.add_argument("--max_budget", type=int, default=10)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10])
    ap.add_argument("--modes", nargs="+", default=["one_shot", "adaptive"],
                     choices=["one_shot", "adaptive"])
    ap.add_argument("--ig_steps", type=int, default=20,
                     help="IG interpolation steps. Higher = more accurate attribution, "
                          "proportionally slower (each step = 1 forward + 1 backward pass).")
    ap.add_argument("--n_urls", type=int, default=1000,
                     help="Subsample this many URLs from --eval_sample (deterministic "
                          "given --seed). Guided attacks cost far more per URL than "
                          "unguided; start small and scale up once timing is known. "
                          "0 = use the full eval sample.")
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                     help="Re-run a mode even if its partial-results file already exists.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    vocab = build_char_vocab()
    model = load_model(args.checkpoint, len(vocab), args.max_len, device)

    sample_df = pd.read_csv(args.eval_sample)
    eligible = [(row.url, LABEL2IDX[row.type]) for row in sample_df.itertuples()]
    print(f"Loaded frozen eval sample: {len(eligible)} URLs from {args.eval_sample}")

    if args.n_urls and len(eligible) > args.n_urls:
        subsample_rng = random.Random(args.seed)
        eligible = subsample_rng.sample(eligible, args.n_urls)
        print(f"Subsampled to {len(eligible)} URLs for this guided run (--n_urls="
              f"{args.n_urls}); the exact subset used is recoverable from the output CSV, "
              f"so a later comparison script can inner-join against unguided results by URL "
              f"regardless of subsample size.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    partial_dir = outdir / "_partial_guided"
    partial_dir.mkdir(exist_ok=True)

    for mode in args.modes:
        partial_path = partial_dir / f"{mode}.csv"
        if partial_path.exists() and not args.force:
            print(f"[{mode}] already completed (found {partial_path}), skipping. "
                  f"Use --force to re-run it.")
            continue

        mode_rows = []
        for i, (url, true_label) in enumerate(eligible):
            result = run_guided_attack(url, true_label, model, vocab, args.max_len, device,
                                        mode, args.max_budget, rng, ig_steps=args.ig_steps)
            result.update({"mode": mode, "url": url, "true_label": CLASSES[true_label]})
            mode_rows.append(result)
            if (i + 1) % 200 == 0:
                print(f"  [{mode}] {i + 1}/{len(eligible)} attacked...")

        pd.DataFrame(mode_rows).to_csv(partial_path, index=False)
        succ = sum(1 for r in mode_rows if r["targeted_success"])
        total_ig_calls = sum(r["ig_calls"] for r in mode_rows)
        print(f"[{mode}] done: {len(eligible)} attacked, {succ} targeted successes within "
              f"budget {args.max_budget}, {total_ig_calls} total IG calls "
              f"(saved to {partial_path})")

    all_partial_files = sorted(partial_dir.glob("*.csv"))
    if not all_partial_files:
        print("No completed modes found -- nothing to merge.")
        return
    results_df = pd.concat([pd.read_csv(f) for f in all_partial_files], ignore_index=True)
    results_df.to_csv(outdir / "guided_attack_results.csv", index=False)

    table_df = build_table_guided(results_df, args.budgets)
    table_df.to_csv(outdir / "table3_guided_only.csv", index=False)

    print("\n=== Guided attack performance by budget (one_shot / adaptive) ===")
    print(table_df.to_string(index=False))
    print(f"\nWritten to {outdir}/guided_attack_results.csv and table3_guided_only.csv")
    print("\nNOTE: this is guided-ONLY. Building the real Table 3 (guided vs. unguided "
          "paired comparison with McNemar/Wilcoxon/effect sizes per §11) requires a "
          "separate comparison script that joins this against unguided_attack_results.csv "
          "by URL -- that's the next step after this run.")


if __name__ == "__main__":
    main()
