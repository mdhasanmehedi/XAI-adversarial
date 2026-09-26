"""
Runs the six §5.1 unguided baseline attacks against a trained checkpoint and
produces the raw per-attack results plus a Table 2 summary (ASR-targeted,
ASR-untargeted, edits, queries, at each requested budget).

Success criterion follows Draft §3 (locked v2.1):
  - targeted (primary): classified specifically as benign
  - untargeted (secondary): any misclassification away from the true label

Attack-eligible population, per §4.1: test-set URLs whose true label is
malicious (phishing/malware/defacement) AND which the clean model classifies
correctly.

Run (from src/):
    python -m attacks.unguided --split_dir ../data/processed/split_v1 \
        --checkpoint ../runs/runs_v1/model_seed0.pt \
        --outdir ../results/tables --max_budget 10 --max_eligible 2000
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add src/ to path
from model import CharCNNTransformer, build_char_vocab, encode_url  # noqa: E402
from train import get_device  # noqa: E402
from attacks.validity import is_valid_url  # noqa: E402
from attacks.transforms import METHODS  # noqa: E402

CLASSES = ["benign", "phishing", "malware", "defacement"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
BENIGN_IDX = LABEL2IDX["benign"]


def load_model(checkpoint_path, vocab_size, max_len, device):
    """Loads a checkpoint, auto-detecting its architecture from a sidecar JSON
    (model_seedN.json next to model_seedN.pt) if one exists -- this is what
    lets attack scripts load a surrogate model (§7), which is trained with a
    deliberately different architecture from the target, without the caller
    needing to know or pass its hyperparameters manually. Falls back to the
    CharCNNTransformer defaults if no sidecar is found (e.g. for checkpoints
    trained before this sidecar mechanism existed)."""
    arch_path = Path(checkpoint_path).with_suffix(".json")
    arch_kwargs = {}
    if arch_path.exists():
        arch = json.loads(arch_path.read_text())
        arch_kwargs = {k: v for k, v in arch.items() if k != "max_len"}
        print(f"[load_model] using architecture from {arch_path.name}: {arch_kwargs}")
    model = CharCNNTransformer(vocab_size=vocab_size, num_classes=len(CLASSES), max_len=max_len,
                                **arch_kwargs).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_batch(model, urls, vocab, max_len, device):
    ids = torch.tensor([encode_url(u, vocab, max_len) for u in urls], dtype=torch.long, device=device)
    return model(ids).argmax(-1).cpu().tolist()


@torch.no_grad()
def predict_one(model, url, vocab, max_len, device):
    return predict_batch(model, [url], vocab, max_len, device)[0]


def build_eligible_population(model, test_df, vocab, max_len, device, batch_size=256):
    """URLs the attacker starts from: true label malicious AND correctly classified."""
    eligible = []
    urls = test_df["url"].tolist()
    labels = [LABEL2IDX[t] for t in test_df["type"].tolist()]
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        preds = predict_batch(model, batch_urls, vocab, max_len, device)
        for u, y, p in zip(batch_urls, batch_labels, preds):
            if y != BENIGN_IDX and p == y:
                eligible.append((u, y))
    return eligible


def run_attack(url, true_label, model, vocab, max_len, device, method_name, method_fn,
               max_budget, rng, max_retries=25):
    """Runs ONE attack up to max_budget edits, logging the first edit index at
    which each success level is reached. Stops early once targeted (primary)
    success is reached -- no reason to keep perturbing past the attacker's goal.

    allow_tld_mutation is passed to the validity check ONLY for the
    domain_path_tld_transform baseline (per §5.5) -- every other baseline must
    preserve the original TLD pattern."""
    allow_tld_mutation = (method_name == "domain_path_tld_transform")
    current = url
    queries = 0
    untargeted_edit = None
    targeted_edit = None
    for edit_num in range(1, max_budget + 1):
        applied = False
        for _ in range(max_retries):
            candidate = method_fn(current, rng)
            if candidate is None:
                continue
            if not is_valid_url(candidate, allow_tld_mutation=allow_tld_mutation):
                continue
            current = candidate
            applied = True
            break
        if not applied:
            break  # exhausted retries without finding a valid move; stop this attack
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
        "untargeted_success": untargeted_edit is not None,
        "untargeted_edits": untargeted_edit,
        "targeted_success": targeted_edit is not None,
        "targeted_edits": targeted_edit,
    }


def build_table2(results_df: pd.DataFrame, budgets):
    rows = []
    for method_name in results_df["method"].unique():
        sub = results_df[results_df.method == method_name]
        n = len(sub)
        for budget in budgets:
            t_hit = sub["targeted_edits"].apply(lambda e: e is not None and e <= budget)
            u_hit = sub["untargeted_edits"].apply(lambda e: e is not None and e <= budget)
            edits_succ = sub.loc[t_hit, "targeted_edits"]
            queries_succ = sub.loc[t_hit, "queries"]
            rows.append({
                "attack": method_name,
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
    ap.add_argument("--max_budget", type=int, default=10)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10])
    ap.add_argument("--eval_sample", default=None,
                     help="Path to a frozen attack_eval_sample_v1.csv from "
                          "build_eval_sample.py. STRONGLY preferred over "
                          "--max_eligible: guarantees this run attacks the exact "
                          "same URLs as every other attack method, which paired "
                          "tests in §11 and Table 3 require.")
    ap.add_argument("--max_eligible", type=int, default=2000,
                     help="Fallback ONLY used when --eval_sample is not given: caps "
                          "a freshly (re-)sampled eligible population; 0 = full test set. "
                          "Prefer --eval_sample for anything feeding into Table 3+.")
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=None, choices=list(METHODS.keys()),
                     help="Run only these methods (default: all six). Combine with the "
                          "auto-skip behavior below to resume after a crash without "
                          "re-running methods that already finished.")
    ap.add_argument("--force", action="store_true",
                     help="Re-run a method even if its partial-results file already "
                          "exists (default: skip methods already completed on disk).")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    vocab = build_char_vocab()
    model = load_model(args.checkpoint, len(vocab), args.max_len, device)

    if args.eval_sample:
        sample_df = pd.read_csv(args.eval_sample)
        eligible = [(row.url, LABEL2IDX[row.type]) for row in sample_df.itertuples()]
        print(f"Loaded frozen eval sample: {len(eligible)} URLs from {args.eval_sample}")
    else:
        print("[warn] --eval_sample not given; deriving an ad-hoc sample from the test "
              "split. This is fine for a quick sanity check, but results from this run "
              "will NOT be guaranteed comparable to other attack methods' results -- "
              "use build_eval_sample.py + --eval_sample for anything going into Table 3+.")
        test_df = pd.read_csv(Path(args.split_dir) / "test.csv")
        eligible = build_eligible_population(model, test_df, vocab, args.max_len, device)
        print(f"Attack-eligible population: {len(eligible)} / {len(test_df)} test URLs "
              f"(malicious + correctly classified)")
        if args.max_eligible and len(eligible) > args.max_eligible:
            eligible = rng.sample(eligible, args.max_eligible)
            print(f"Sampled down to {len(eligible)} (--max_eligible={args.max_eligible})")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    partial_dir = outdir / "_partial"
    partial_dir.mkdir(exist_ok=True)

    methods_to_run = args.methods or list(METHODS.keys())
    for method_name in methods_to_run:
        method_fn = METHODS[method_name]
        partial_path = partial_dir / f"{method_name}.csv"
        if partial_path.exists() and not args.force:
            print(f"[{method_name}] already completed (found {partial_path}), "
                  f"skipping. Use --force to re-run it.")
            continue

        method_rows = []
        for url, true_label in eligible:
            result = run_attack(url, true_label, model, vocab, args.max_len, device,
                                 method_name, method_fn, args.max_budget, rng)
            result.update({"method": method_name, "url": url, "true_label": CLASSES[true_label]})
            method_rows.append(result)

        # Save THIS method's results to disk immediately -- if a later method
        # crashes, everything up to here is already safe on disk and does not
        # need to be re-attacked.
        pd.DataFrame(method_rows).to_csv(partial_path, index=False)
        succ = sum(1 for r in method_rows if r["targeted_success"])
        print(f"[{method_name}] done: {len(eligible)} attacked, {succ} targeted successes "
              f"within budget {args.max_budget} (saved to {partial_path})")

    # Merge every completed method's partial file (from this run AND any prior
    # run) into the combined results -- this is what makes resuming safe.
    all_partial_files = sorted(partial_dir.glob("*.csv"))
    if not all_partial_files:
        print("No completed methods found -- nothing to merge.")
        return
    results_df = pd.concat([pd.read_csv(f) for f in all_partial_files], ignore_index=True)
    results_df.to_csv(outdir / "unguided_attack_results.csv", index=False)

    table2_df = build_table2(results_df, args.budgets)
    table2_df.to_csv(outdir / "table2_results.csv", index=False)

    print("\n=== Table 2: Attack performance by budget (unguided baselines) ===")
    print(table2_df.to_string(index=False))
    print(f"\nWritten to {outdir}/unguided_attack_results.csv and table2_results.csv")


if __name__ == "__main__":
    main()
