"""
Defended-model training, Draft §8.

Implements two independent, togglable defense techniques, matching the
draft's four conditions (--defense none / fgm / charaug / combined):

  - FGM adversarial training (Miyato-style, adapted to the standard "perturb
    the embedding weight matrix" pattern widely used in text-classification
    adversarial training): after the normal forward/backward pass, perturb
    the embedding LAYER'S WEIGHTS by a small step in the direction of the
    gradient just computed, do a second forward/backward pass through the
    perturbed embeddings, then restore the original weights before the
    optimizer step (which uses the ACCUMULATED gradient from both passes).
    This operates purely in continuous embedding space during training; it
    produces no discrete "attack" and needs no §5.5 validity constraint --
    it's noise injected to shape the loss landscape, not attacker behavior.
  - Character-level augmentation: applies random DISCRETE unguided
    perturbations -- reusing the EXACT §5.1 transform primitives from
    attacks/transforms.py, not a reimplementation -- to a fraction of
    training examples each epoch, forcing invariance to the kinds of edits
    an unguided attacker would make.

§8's explicit generalization requirement ("evaluate defended models against
attack families...not identical to the training perturbations") is enforced
at EVALUATION time (defense/evaluate_defense.py), not here -- this script
only trains models; a separate script decides what they're tested against.

Checkpoints get the same architecture-sidecar treatment as train.py's target/
surrogate models, so every existing attack/faithfulness script can load a
defended model with zero changes.

Run (from src/):
    python -m defense.train_defended --split_dir ../data/processed/split_v1 \
        --outdir ../runs/defense --seed 0 --epochs 10 --defense fgm
    python -m defense.train_defended ... --defense charaug
    python -m defense.train_defended ... --defense combined
    python -m defense.train_defended ... --defense none   # standard-training control
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add src/ to path
from model import CharCNNTransformer, build_char_vocab, encode_url, PAD_IDX  # noqa: E402
from train import CLASSES, LABEL2IDX, get_device, set_seed, evaluate  # noqa: E402
from attacks.transforms import METHODS as ALL_UNGUIDED_METHODS  # noqa: E402

# Character-augmentation deliberately draws from a SUBSET of the §5.1
# baselines -- the pure character-level edits only. segment_manipulation and
# domain_path_tld_transform are excluded on purpose, so they remain genuinely
# unseen structural attack types at evaluation time (§8's explicit
# generalization requirement: test against attacks not identical to training
# perturbations). Attribution-guided attacks are unseen regardless, since
# nothing about random augmentation resembles attribution-directed selection.
CHARAUG_METHODS = {
    k: v for k, v in ALL_UNGUIDED_METHODS.items()
    if k in ("random_substitution", "random_insertion", "random_deletion", "adjacent_swap")
}


class AugmentedURLDataset(Dataset):
    """Same as train.py's URLDataset, but optionally applies a random §5.1
    unguided transform to a fraction of examples EACH TIME they're drawn
    (not once at dataset-construction time), so augmentation varies epoch to
    epoch rather than being a fixed, memorizable set of noisy variants."""

    def __init__(self, df: pd.DataFrame, vocab: dict, max_len: int,
                 augment_prob: float = 0.0, seed: int = 0):
        self.urls = df["url"].tolist()
        self.labels = [LABEL2IDX[t] for t in df["type"].tolist()]
        self.vocab = vocab
        self.max_len = max_len
        self.augment_prob = augment_prob
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        if self.augment_prob > 0 and self.rng.random() < self.augment_prob:
            method_name = self.rng.choice(list(CHARAUG_METHODS.keys()))
            candidate = CHARAUG_METHODS[method_name](url, self.rng)
            if candidate is not None:
                url = candidate  # no §5.5 validity check needed -- this is
                                  # training-time noise, not simulated attacker
                                  # output, so realism/deployability is irrelevant.
        ids = encode_url(url, self.vocab, self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class FGM:
    """Standard embedding-weight FGM adversarial training helper. attack()
    perturbs the embedding weight matrix using its just-computed gradient;
    restore() undoes it. Between the two, a second forward/backward pass is
    run so the optimizer step uses the SUM of clean and adversarial gradients."""

    def __init__(self, model, epsilon=1.0, emb_name="embed.weight"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(self.epsilon * param.grad / norm)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def run_one_epoch_defended(model, loader, optimizer, device, train, defense, fgm=None):
    model.train(train)
    total_loss, all_preds, all_labels = 0.0, [], []
    criterion = nn.CrossEntropyLoss()
    for ids, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
            logits = model(ids)
            loss = criterion(logits, labels)
            loss.backward()

            if defense in ("fgm", "combined"):
                fgm.attack()
                adv_logits = model(ids)
                adv_loss = criterion(adv_logits, labels)
                adv_loss.backward()
                fgm.restore()

            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(ids)
                loss = criterion(logits, labels)

        total_loss += loss.item() * ids.size(0)
        all_preds.extend(logits.argmax(-1).detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
    return total_loss / len(loader.dataset), all_preds, all_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--defense", choices=["none", "fgm", "charaug", "combined"], default="none")
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fgm_epsilon", type=float, default=1.0)
    ap.add_argument("--augment_prob", type=float, default=0.3,
                     help="Per-example probability of applying a random §5.1 unguided "
                          "transform during training, for charaug/combined defenses.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}  |  defense={args.defense}")
    vocab = build_char_vocab()

    train_df = pd.read_csv(Path(args.split_dir) / "train.csv")
    val_df = pd.read_csv(Path(args.split_dir) / "val.csv")
    test_df = pd.read_csv(Path(args.split_dir) / "test.csv")

    use_aug = args.defense in ("charaug", "combined")
    train_ds = AugmentedURLDataset(train_df, vocab, args.max_len,
                                    augment_prob=args.augment_prob if use_aug else 0.0,
                                    seed=args.seed)
    val_ds = AugmentedURLDataset(val_df, vocab, args.max_len, augment_prob=0.0)
    test_ds = AugmentedURLDataset(test_df, vocab, args.max_len, augment_prob=0.0)

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = CharCNNTransformer(vocab_size=len(vocab), num_classes=len(CLASSES),
                                max_len=args.max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    fgm = FGM(model, epsilon=args.fgm_epsilon) if args.defense in ("fgm", "combined") else None

    best_val_macro_f1, best_state = -1.0, None
    for epoch in range(args.epochs):
        train_loss, _, _ = run_one_epoch_defended(model, train_loader, optimizer, device,
                                                    train=True, defense=args.defense, fgm=fgm)
        val_loss, val_preds, val_labels = run_one_epoch_defended(
            model, val_loader, optimizer, device, train=False, defense=args.defense, fgm=fgm)
        val_metrics = evaluate(val_preds, val_labels)
        print(f"[{args.defense}] epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    _, test_preds, test_labels = run_one_epoch_defended(
        model, test_loader, optimizer, device, train=False, defense=args.defense, fgm=fgm)
    test_metrics = evaluate(test_preds, test_labels)
    print(f"[{args.defense}] TEST acc={test_metrics['accuracy']:.4f} "
          f"macro_f1={test_metrics['macro_f1']:.4f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"{args.defense}_seed{args.seed}"
    ckpt_path = outdir / f"{ckpt_name}.pt"
    torch.save(best_state, ckpt_path)
    (outdir / f"{ckpt_name}.json").write_text(json.dumps({
        "max_len": args.max_len, "embed_dim": 64, "conv_channels": 128,
        "transformer_layers": 2, "transformer_heads": 4,  # standard architecture,
        # unlike the surrogate -- defenses test TRAINING differences, not
        # architecture differences, so this is deliberately the target's own shape.
    }, indent=2))
    print(f"Written checkpoint to {ckpt_path}")
    print(f"Clean test accuracy={test_metrics['accuracy']:.4f} "
          f"macro_f1={test_metrics['macro_f1']:.4f} -- record this as this defense's "
          f"'Clean Acc.' in Table 6.")


if __name__ == "__main__":
    main()
