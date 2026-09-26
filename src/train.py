"""
Trains the target model across N independent seeds on the FROZEN split from
data_prep.py, evaluates each on the held-out test set, and writes exactly the
numbers Table 1 needs: per-seed Accuracy, Macro-F1, and per-class F1.

Only the split (train.csv/val.csv/test.csv) is shared across seeds -- model
init and data shuffling differ per seed, which is what lets Table 1 report
real seed-to-seed variance instead of a single-run point estimate (per the
draft's §11 statistical plan: minimum 3, target 5 seeds).

Run:
    python train.py --split_dir split_v1 --outdir runs --seeds 0 1 2 3 4
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

from model import CharCNNTransformer, build_char_vocab, encode_url

CLASSES = ["benign", "phishing", "malware", "defacement"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}


class URLDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: dict, max_len: int):
        self.urls = df["url"].tolist()
        self.labels = [LABEL2IDX[t] for t in df["type"].tolist()]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        ids = encode_url(self.urls[idx], self.vocab, self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def get_device():
    """Prefer CUDA (if you ever run this on a non-Mac/cloud GPU box), then MPS
    (Apple Silicon GPU — M1/M2/M3), then fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def run_one_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    total_loss, all_preds, all_labels = 0.0, [], []
    criterion = nn.CrossEntropyLoss()
    for ids, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            logits = model(ids)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * ids.size(0)
        all_preds.extend(logits.argmax(-1).detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_preds, all_labels


def evaluate(preds, labels):
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    class_f1 = f1_score(labels, preds, average=None, labels=list(range(len(CLASSES))), zero_division=0)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "class_f1": {c: float(class_f1[i]) for i, c in enumerate(CLASSES)},
    }


def train_one_seed(seed, split_dir, outdir, max_len, epochs, batch_size, lr, device, arch):
    set_seed(seed)
    vocab = build_char_vocab()

    train_df = pd.read_csv(Path(split_dir) / "train.csv")
    val_df = pd.read_csv(Path(split_dir) / "val.csv")
    test_df = pd.read_csv(Path(split_dir) / "test.csv")

    train_ds = URLDataset(train_df, vocab, max_len)
    val_ds = URLDataset(val_df, vocab, max_len)
    test_ds = URLDataset(test_df, vocab, max_len)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = CharCNNTransformer(vocab_size=len(vocab), num_classes=len(CLASSES), max_len=max_len, **arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_macro_f1 = -1.0
    best_state = None
    for epoch in range(epochs):
        train_loss, _, _ = run_one_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_preds, val_labels = run_one_epoch(model, val_loader, optimizer, device, train=False)
        val_metrics = evaluate(val_preds, val_labels)
        print(f"[seed {seed}] epoch {epoch+1}/{epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    _, test_preds, test_labels = run_one_epoch(model, test_loader, optimizer, device, train=False)
    test_metrics = evaluate(test_preds, test_labels)
    print(f"[seed {seed}] TEST  acc={test_metrics['accuracy']:.4f} macro_f1={test_metrics['macro_f1']:.4f}")

    ckpt_path = Path(outdir) / f"model_seed{seed}.pt"
    torch.save(best_state, ckpt_path)
    # Save architecture metadata alongside the checkpoint so any script loading
    # it later (attacks.unguided, attacks.guided, faithfulness.*) can rebuild
    # the exact right model shape automatically -- this matters most for a
    # surrogate model (§7), which is trained with a DELIBERATELY different
    # architecture from the target; without this sidecar, loading a surrogate
    # checkpoint with the default architecture would crash with a shape
    # mismatch, or (worse, if shapes happened to coincide) silently load wrong.
    arch_path = Path(outdir) / f"model_seed{seed}.json"
    arch_path.write_text(json.dumps({"max_len": max_len, **arch}, indent=2))
    return test_metrics, str(ckpt_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed_dim", type=int, default=64,
                     help="Char embedding dim. Override for a surrogate model (§7) to give "
                          "it a genuinely different architecture from the target, not just a "
                          "different random seed of the same model.")
    ap.add_argument("--conv_channels", type=int, default=128)
    ap.add_argument("--transformer_layers", type=int, default=2)
    ap.add_argument("--transformer_heads", type=int, default=4)
    args = ap.parse_args()
    arch = {"embed_dim": args.embed_dim, "conv_channels": args.conv_channels,
            "transformer_layers": args.transformer_layers, "transformer_heads": args.transformer_heads}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")
    if device.type == "mps":
        print("Apple Silicon GPU (MPS) detected. If you hit an "
              "'operator not implemented for MPS' error on an uncommon op, "
              "re-run with: PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py ...")

    table1_rows = []
    for seed in args.seeds:
        metrics, ckpt_path = train_one_seed(
            seed, args.split_dir, outdir, args.max_len, args.epochs,
            args.batch_size, args.lr, device, arch,
        )
        table1_rows.append({
            "seed": seed,
            "accuracy": round(metrics["accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            **{f"f1_{c}": round(metrics["class_f1"][c], 4) for c in CLASSES},
            "checkpoint": ckpt_path,
        })

    table1_df = pd.DataFrame(table1_rows)
    mean_row = table1_df.drop(columns=["seed", "checkpoint"]).mean(numeric_only=True)
    std_row = table1_df.drop(columns=["seed", "checkpoint"]).std(numeric_only=True)
    print("\n=== Table 1: Clean target-model performance ===")
    print(table1_df.drop(columns="checkpoint").to_string(index=False))
    print(f"\nMean:  " + "  ".join(f"{k}={v:.4f}" for k, v in mean_row.items()))
    print(f"Std:   " + "  ".join(f"{k}={v:.4f}" for k, v in std_row.items()))

    table1_df.to_csv(outdir / "table1_results.csv", index=False)
    summary = {"per_seed": table1_rows,
               "mean": mean_row.round(4).to_dict(),
               "std": std_row.round(4).to_dict()}
    (outdir / "table1_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWritten to {outdir}/table1_results.csv and table1_summary.json")
    print("Paste these numbers directly into Table 1 of the draft.")


if __name__ == "__main__":
    main()
