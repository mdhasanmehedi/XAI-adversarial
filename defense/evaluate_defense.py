"""
Defense evaluation, Draft §8 (Table 6) and §8.1 (Table 7), testing H5.

Reuses the EXACT SAME attack functions already built and tested for §5/§7 --
attacks.unguided.run_attack and attacks.guided.run_guided_attack -- rather
than reimplementing attack logic here. This script only orchestrates: load a
defended checkpoint, determine ITS OWN attack-eligible population (§4.1's
definition applied to this specific model, not blindly reused from the
undefended target's eligible list -- a defended model may classify a
slightly different subset of URLs correctly), run a fixed battery of attacks
against it, and report Clean/Robust accuracy, ASR, edit cost, and
faithfulness before/after.

Generalization test (§8's explicit requirement -- "evaluate defended models
against attack families...not identical to the training perturbations"):
the attack battery deliberately includes attacks from THREE distinct
relationships to what charaug training saw:
  - random_substitution: a "seen-type" attack (charaug trains against this
    exact transform family) -- expect the strongest defense effect here for
    charaug/combined, since this is closest to memorization territory.
  - segment_manipulation: an "unseen-structural" attack (deliberately
    excluded from charaug's training pool -- see defense/train_defended.py)
  - one_shot (attribution-guided): unseen regardless of defense type, since
    nothing about random or embedding-space training resembles
    attribution-directed position selection.
Comparing robustness across these three tells you whether a defense
generalizes or merely memorizes its own training perturbations.

Run (from src/), once you have defended checkpoints from train_defended.py:
    python -m defense.evaluate_defense \
        --split_dir ../data/processed/split_v1 \
        --defense_checkpoint ../runs/defense/charaug_seed0.pt \
        --undefended_checkpoint ../runs/runs_v1/model_seed0.pt \
        --eval_sample ../data/processed/attack_eval_sample_v1.csv \
        --outdir ../results/tables --max_budget 10 --n_urls 500 --seed 0
"""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add src/ to path
from model import build_char_vocab, PAD_IDX  # noqa: E402
from train import get_device, evaluate as compute_metrics  # noqa: E402
from attacks.unguided import (  # noqa: E402
    load_model, predict_batch, predict_one, build_eligible_population,
    run_attack, METHODS as ALL_UNGUIDED_METHODS, CLASSES, LABEL2IDX,
)
from attacks.guided import run_guided_attack  # noqa: E402
from faithfulness.deletion_auc import compute_faithfulness_for_url  # noqa: E402

EVAL_ATTACKS_UNGUIDED = ["random_substitution", "segment_manipulation"]


def clean_accuracy(model, test_df, vocab, max_len, device, batch_size=256):
    urls = test_df["url"].tolist()
    labels = [LABEL2IDX[t] for t in test_df["type"].tolist()]
    all_preds = []
    for i in range(0, len(urls), batch_size):
        all_preds.extend(predict_batch(model, urls[i:i + batch_size], vocab, max_len, device))
    return compute_metrics(all_preds, labels)


def own_eligible_population(model, candidate_urls, vocab, max_len, device):
    """§4.1's eligibility definition, applied to THIS model specifically:
    true label malicious AND this model currently classifies it correctly.
    A defended model may not classify the exact same subset correctly as
    the undefended target did, so this is re-derived per model rather than
    reused blindly."""
    eligible = []
    for url, true_label_str in candidate_urls:
        true_idx = LABEL2IDX[true_label_str]
        if true_idx == 0:  # benign, skip (0 = benign per CLASSES ordering)
            continue
        pred = predict_one(model, url, vocab, max_len, device)
        if pred == true_idx:
            eligible.append((url, true_idx))
    return eligible


def run_unguided_eval(model, method_name, eligible, vocab, max_len, device, max_budget, rng):
    method_fn = ALL_UNGUIDED_METHODS[method_name]
    successes, edits = 0, []
    for url, true_label in eligible:
        result = run_attack(url, true_label, model, vocab, max_len, device,
                             method_name, method_fn, max_budget, rng)
        if result["targeted_success"]:
            successes += 1
            edits.append(result["targeted_edits"])
    n = len(eligible)
    return {
        "attack": method_name, "n_eligible": n, "n_success": successes,
        "asr": round(successes / n, 4) if n else None,
        "median_edits": (pd.Series(edits).median() if edits else None),
    }


def run_guided_eval(model, eligible, vocab, max_len, device, max_budget, ig_steps, rng):
    successes, edits = 0, []
    for url, true_label in eligible:
        result = run_guided_attack(url, true_label, model, vocab, max_len, device,
                                    "one_shot", max_budget, rng, ig_steps=ig_steps)
        if result["targeted_success"]:
            successes += 1
            edits.append(result["targeted_edits"])
    n = len(eligible)
    return {
        "attack": "one_shot (guided)", "n_eligible": n, "n_success": successes,
        "asr": round(successes / n, 4) if n else None,
        "median_edits": (pd.Series(edits).median() if edits else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--defense_checkpoint", required=True)
    ap.add_argument("--undefended_checkpoint", required=True,
                     help="The original target checkpoint (e.g. runs_v1/model_seed0.pt), "
                          "used as the 'before' point for Table 7's faithfulness comparison.")
    ap.add_argument("--eval_sample", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--defense_name", default=None,
                     help="Label for this defense in the output tables. Default: inferred "
                          "from the checkpoint filename.")
    ap.add_argument("--max_budget", type=int, default=10)
    ap.add_argument("--n_urls", type=int, default=500,
                     help="Subsample of --eval_sample's candidates to check eligibility on "
                          "and attack. Full attack batteries are expensive; start modest.")
    ap.add_argument("--ig_steps", type=int, default=20)
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    defense_name = args.defense_name or Path(args.defense_checkpoint).stem
    rng = random.Random(args.seed)
    device = get_device()
    print(f"Using device: {device}  |  evaluating defense: {defense_name}")
    vocab = build_char_vocab()

    defended_model = load_model(args.defense_checkpoint, len(vocab), args.max_len, device)
    undefended_model = load_model(args.undefended_checkpoint, len(vocab), args.max_len, device)

    # --- Clean accuracy ---
    test_df = pd.read_csv(Path(args.split_dir) / "test.csv")
    clean_metrics = clean_accuracy(defended_model, test_df, vocab, args.max_len, device)
    print(f"Clean accuracy: {clean_metrics['accuracy']:.4f}  macro_f1: {clean_metrics['macro_f1']:.4f}")

    # --- This model's own eligible population, from the frozen sample's candidates ---
    sample_df = pd.read_csv(args.eval_sample)
    candidates = list(zip(sample_df["url"], sample_df["type"]))
    if args.n_urls and len(candidates) > args.n_urls:
        candidates = rng.sample(candidates, args.n_urls)
    eligible = own_eligible_population(defended_model, candidates, vocab, args.max_len, device)
    print(f"Defended model's own eligible population: {len(eligible)} / {len(candidates)} candidates")

    # --- Attack battery: seen-type, unseen-structural, unseen-guided ---
    attack_rows = []
    for method_name in EVAL_ATTACKS_UNGUIDED:
        row = run_unguided_eval(defended_model, method_name, eligible, vocab, args.max_len,
                                 device, args.max_budget, rng)
        row["defense"] = defense_name
        attack_rows.append(row)
        print(f"  [{method_name}] ASR={row['asr']}  n_eligible={row['n_eligible']}")

    guided_row = run_guided_eval(defended_model, eligible, vocab, args.max_len, device,
                                  args.max_budget, args.ig_steps, rng)
    guided_row["defense"] = defense_name
    attack_rows.append(guided_row)
    print(f"  [one_shot guided] ASR={guided_row['asr']}  n_eligible={guided_row['n_eligible']}")

    # Table 6 row: use the LOWEST ASR / highest robust-acc? No -- report each
    # attack separately (more informative), plus one summary row using the
    # unseen-guided attack as the headline "Robust Acc." per §8's emphasis on
    # generalization to unseen/adaptive attacks specifically.
    headline_asr = guided_row["asr"]
    table6_row = {
        "defense": defense_name,
        "clean_acc": round(clean_metrics["accuracy"], 4),
        "robust_acc_vs_guided": round(1 - headline_asr, 4) if headline_asr is not None else None,
        "asr_vs_guided": headline_asr,
        "median_edit_cost_vs_guided": guided_row["median_edits"],
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    attack_battery_df = pd.DataFrame(attack_rows)
    attack_battery_path = outdir / f"table6_attack_battery_{defense_name}.csv"
    attack_battery_df.to_csv(attack_battery_path, index=False)

    table6_path = outdir / "table6_results.csv"
    if table6_path.exists():
        existing = pd.read_csv(table6_path)
        existing = existing[existing["defense"] != defense_name]  # replace if re-run
        table6_df = pd.concat([existing, pd.DataFrame([table6_row])], ignore_index=True)
    else:
        table6_df = pd.DataFrame([table6_row])
    table6_df.to_csv(table6_path, index=False)

    # --- Table 7: faithfulness before (undefended) vs after (this defense) ---
    faith_rows = []
    for url, true_label in eligible:
        before = compute_faithfulness_for_url(undefended_model, url, true_label, vocab,
                                                args.max_len, device, args.ig_steps, 0.2)
        after = compute_faithfulness_for_url(defended_model, url, true_label, vocab,
                                               args.max_len, device, args.ig_steps, 0.2)
        if before is not None and after is not None:
            faith_rows.append({"url": url, "deletion_auc_before": before["deletion_auc"],
                                "deletion_auc_after": after["deletion_auc"]})
    faith_df = pd.DataFrame(faith_rows)
    mean_before = faith_df["deletion_auc_before"].mean() if len(faith_df) else None
    mean_after = faith_df["deletion_auc_after"].mean() if len(faith_df) else None
    table7_row = {
        "defense": defense_name,
        "faithfulness_before": round(mean_before, 4) if mean_before is not None else None,
        "faithfulness_after": round(mean_after, 4) if mean_after is not None else None,
        "change": round(mean_after - mean_before, 4) if (mean_before is not None and mean_after is not None) else None,
        "n": len(faith_df),
    }
    table7_path = outdir / "table7_results.csv"
    if table7_path.exists():
        existing = pd.read_csv(table7_path)
        existing = existing[existing["defense"] != defense_name]
        table7_df = pd.concat([existing, pd.DataFrame([table7_row])], ignore_index=True)
    else:
        table7_df = pd.DataFrame([table7_row])
    table7_df.to_csv(table7_path, index=False)

    print(f"\n=== Table 6 row (defense={defense_name}) ===")
    print(table6_row)
    print(f"\n=== Table 7 row (defense={defense_name}) ===")
    print(table7_row)
    print(f"\nAttack battery detail written to {attack_battery_path}")
    print(f"Table 6 written to {table6_path}")
    print(f"Table 7 written to {table7_path}")
    print("\nNOTE: robust_acc_vs_guided/asr_vs_guided are the HEADLINE Table 6 numbers, "
          "using the unseen attribution-guided attack per §8's emphasis on generalization. "
          "The per-attack-type breakdown in the battery CSV (including the 'seen-type' "
          "random_substitution result) is what actually shows whether this defense "
          "generalizes or just memorizes -- compare seen-type ASR vs unseen-type ASR there.")


if __name__ == "__main__":
    main()
