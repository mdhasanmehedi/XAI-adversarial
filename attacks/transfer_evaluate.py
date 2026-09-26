"""
Surrogate transfer evaluation, Draft §7 (Table 5), testing H4.

Threat model (§3, surrogate/transfer level): the attacker has ZERO access to
the target model's internals or predictions, and instead attacks a surrogate
they trained themselves, then transfers the resulting perturbed URLs to the
target UNMODIFIED -- no target queries are spent adapting the attack.

This script does NOT run any new attacks itself. It takes attack results
ALREADY produced by attacks.unguided / attacks.guided run against a SURROGATE
checkpoint (point --outdir at a surrogate-specific results folder when running
those scripts, so they don't overwrite the target's results), then:
  1. Restricts to URLs where the attack SUCCEEDED against the surrogate
     (surrogate ASR is exactly the ASR already reported by those scripts --
     no recomputation needed, just read it off their output).
  2. Re-queries the TARGET model on each surrogate-successful URL's final
     (perturbed) string, completely unmodified -- this is what "transfer"
     means: the attacker never touches the target during the attack itself.
  3. Reports Target transfer ASR = the fraction of surrogate-successes that
     ALSO fool the target. This is the standard transferability-rate
     definition in the adversarial ML literature (conditioning on the source
     model actually being fooled, not on the full attempted set).

Run (from src/), after producing surrogate-side attack results:
    # 1. Train a surrogate with a genuinely different architecture (train.py):
    python train.py --split_dir ../data/processed/split_v1 \
        --outdir ../runs/surrogate --seeds 100 --epochs 10 \
        --embed_dim 32 --conv_channels 64 --transformer_layers 1 --transformer_heads 2

    # 2. Build the surrogate's OWN eligible-population sample (attacker only
    #    knows what THEIR model gets right, not the target's):
    python -m attacks.build_eval_sample --split_dir ../data/processed/split_v1 \
        --checkpoint ../runs/surrogate/model_seed100.pt \
        --outdir ../data/processed --filename surrogate_eval_sample_v1.csv \
        --n 2000 --seed 0

    # 3. Attack the surrogate (reusing existing scripts, separate --outdir):
    python -m attacks.unguided --split_dir ../data/processed/split_v1 \
        --checkpoint ../runs/surrogate/model_seed100.pt \
        --outdir ../results/tables/surrogate \
        --eval_sample ../data/processed/surrogate_eval_sample_v1.csv \
        --max_budget 10 --budgets 1 2 3 5 8 10 --seed 0
    python -m attacks.guided --split_dir ../data/processed/split_v1 \
        --checkpoint ../runs/surrogate/model_seed100.pt \
        --outdir ../results/tables/surrogate \
        --eval_sample ../data/processed/surrogate_eval_sample_v1.csv \
        --modes one_shot --max_budget 10 --budgets 1 2 3 5 8 10 --seed 0

    # 4. THIS script: measure transfer to the real target.
    python -m attacks.transfer_evaluate \
        --target_checkpoint ../runs/runs_v1/model_seed0.pt \
        --surrogate_results_dir ../results/tables/surrogate \
        --outdir ../results/tables --max_len 200
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add src/ to path
from model import build_char_vocab  # noqa: E402
from train import get_device  # noqa: E402
from attacks.unguided import load_model, predict_one, LABEL2IDX, BENIGN_IDX  # noqa: E402


def load_surrogate_results(surrogate_results_dir: Path):
    frames = []
    unguided_path = surrogate_results_dir / "unguided_attack_results.csv"
    guided_path = surrogate_results_dir / "guided_attack_results.csv"
    if unguided_path.exists():
        df = pd.read_csv(unguided_path).rename(columns={"method": "attack_name"})
        frames.append(df)
    if guided_path.exists():
        df = pd.read_csv(guided_path).rename(columns={"mode": "attack_name"})
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No unguided_attack_results.csv or guided_attack_results.csv found in "
            f"{surrogate_results_dir} -- run attacks.unguided / attacks.guided against "
            f"the surrogate checkpoint first, with --outdir pointed at this directory.")
    return pd.concat(frames, ignore_index=True)


def evaluate_transfer_for_attack(name, attack_df, target_model, vocab, max_len, device):
    surrogate_success = attack_df[attack_df["targeted_success"].astype(bool)].copy()
    n_surrogate_success = len(surrogate_success)
    if n_surrogate_success == 0:
        return {"attack": name, "surrogate_asr": 0.0, "n_surrogate_success": 0,
                "target_transfer_asr": None, "target_transfer_asr_untargeted": None,
                "median_edits": None}

    target_targeted_hits, target_untargeted_hits = 0, 0
    for row in surrogate_success.itertuples():
        pred = predict_one(target_model, row.final_url, vocab, max_len, device)
        true_idx = LABEL2IDX[row.true_label]
        if pred == BENIGN_IDX:
            target_targeted_hits += 1
        if pred != true_idx:
            target_untargeted_hits += 1

    n_total_attempted = len(attack_df)
    return {
        "attack": name,
        "n_attempted": n_total_attempted,
        "surrogate_asr": round(n_surrogate_success / n_total_attempted, 4),
        "n_surrogate_success": n_surrogate_success,
        "target_transfer_asr": round(target_targeted_hits / n_surrogate_success, 4),
        "target_transfer_asr_untargeted": round(target_untargeted_hits / n_surrogate_success, 4),
        "median_edits": float(surrogate_success["targeted_edits"].median()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_checkpoint", required=True)
    ap.add_argument("--surrogate_results_dir", required=True,
                     help="Directory containing the surrogate's unguided_attack_results.csv "
                          "and/or guided_attack_results.csv (from running attacks.unguided / "
                          "attacks.guided against the SURROGATE checkpoint).")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max_len", type=int, default=200)
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    vocab = build_char_vocab()
    target_model = load_model(args.target_checkpoint, len(vocab), args.max_len, device)

    surrogate_results_dir = Path(args.surrogate_results_dir)
    all_results = load_surrogate_results(surrogate_results_dir)
    names = sorted(all_results["attack_name"].unique())
    print(f"Surrogate-side attack methods found: {names}")

    rows = []
    for name in names:
        attack_df = all_results[all_results.attack_name == name]
        row = evaluate_transfer_for_attack(name, attack_df, target_model, vocab, args.max_len, device)
        rows.append(row)
        print(f"[{name}] surrogate ASR={row['surrogate_asr']:.4f} "
              f"({row['n_surrogate_success']} successes) -> "
              f"target transfer ASR={row['target_transfer_asr']}")

    table5 = pd.DataFrame(rows)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "table5_results.csv"
    table5.to_csv(out_path, index=False)

    print(f"\n=== Table 5: Surrogate transfer ===")
    print(table5.to_string(index=False))
    print(f"\nWritten to {out_path}")
    print("\nNOTE: target_transfer_asr is computed ONLY over URLs where the attack "
          "already succeeded against the surrogate (the standard transferability-rate "
          "definition), not over all attempted URLs -- surrogate_asr (over all attempted "
          "URLs) and target_transfer_asr (over surrogate successes only) answer different "
          "questions and should not be directly compared as if on the same denominator.")


if __name__ == "__main__":
    main()
