"""
Adds a second detector architecture (character-level BiLSTM with attention pooling) to the repo.
Run ONCE from the repository root:   python apply_bilstm_patch.py
It edits src/model.py, src/train.py and src/attacks/unguided.py in place (a .bak copy of each is kept)
and is safe to run twice. Nothing else in the repo changes, and checkpoints of the original
CNN-Transformer keep loading exactly as before (no sidecar arch_type -> CNN-Transformer).
"""
from pathlib import Path
import re, shutil, sys

ROOT = Path(".")
SRC = ROOT / "src"
if not (SRC / "model.py").exists():
    sys.exit("Run this from the repository root (the folder that contains src/).")

def backup(p):
    b = p.with_suffix(p.suffix + ".bak")
    if not b.exists():
        shutil.copy(p, b)

# ---------------- model.py ----------------
p = SRC / "model.py"; s = p.read_text()
if "class CharBiLSTM" not in s:
    backup(p)
    s = s.rstrip() + '''


class CharBiLSTM(nn.Module):
    """Second detector architecture: char embedding -> stacked bidirectional LSTM ->
    attention pooling over non-pad positions -> linear head. It has no convolution and no
    self-attention, and exposes the same interface as CharCNNTransformer (embed,
    forward_from_embeds, forward), so every attack, attribution and faithfulness script
    works with it unchanged. Sequences are padded, not packed (packing is not reliable on MPS);
    padded positions are excluded from the pooling."""

    def __init__(self, vocab_size, num_classes=4, embed_dim=64, hidden_dim=128, num_layers=2,
                 max_len=200, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attn_pool = nn.Linear(2 * hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden_dim, num_classes)
        self.max_len = max_len

    def forward_from_embeds(self, embeds, pad_mask):
        x, _ = self.lstm(embeds)                                  # [B, L, 2H]
        scores = self.attn_pool(x).squeeze(-1)                     # [B, L]
        scores = scores.masked_fill(pad_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)      # [B, L, 1]
        pooled = (x * weights).sum(dim=1)                          # [B, 2H]
        return self.head(self.dropout(pooled))

    def forward(self, input_ids):
        pad_mask = input_ids.eq(PAD_IDX)
        return self.forward_from_embeds(self.embed(input_ids), pad_mask)


ARCHS = {"cnn_transformer": CharCNNTransformer, "bilstm": CharBiLSTM}


def build_model(vocab_size, num_classes=4, max_len=200, arch_type="cnn_transformer", **kwargs):
    """Factory used by train.py and attacks.unguided.load_model. arch_type comes from the sidecar
    JSON next to a checkpoint; checkpoints without it are CNN-Transformers (backward compatible)."""
    return ARCHS[arch_type](vocab_size=vocab_size, num_classes=num_classes, max_len=max_len, **kwargs)
'''
    p.write_text(s + "\n")
    print("model.py patched")
else:
    print("model.py already patched")

# ---------------- train.py ----------------
p = SRC / "train.py"; s = p.read_text()
if "build_model" not in s:
    backup(p)
    a = "from model import CharCNNTransformer, build_char_vocab, encode_url"
    assert a in s, "train.py import line not found"
    s = s.replace(a, "from model import CharCNNTransformer, build_char_vocab, encode_url, build_model")
    a = "model = CharCNNTransformer(vocab_size=len(vocab), num_classes=len(CLASSES), max_len=max_len, **arch).to(device)"
    assert a in s, "train.py model line not found"
    s = s.replace(a, "model = build_model(vocab_size=len(vocab), num_classes=len(CLASSES), max_len=max_len, **arch).to(device)")
    a = '    ap.add_argument("--transformer_heads", type=int, default=4)\n'
    assert a in s, "train.py args not found"
    s = s.replace(a, a + '''    ap.add_argument("--arch_type", default="cnn_transformer", choices=["cnn_transformer", "bilstm"],
                     help="Detector architecture. bilstm = character-level BiLSTM with attention pooling.")
    ap.add_argument("--hidden_dim", type=int, default=128, help="BiLSTM hidden size per direction.")
    ap.add_argument("--num_layers", type=int, default=2, help="BiLSTM layers.")
''')
    a = '''    arch = {"embed_dim": args.embed_dim, "conv_channels": args.conv_channels,
            "transformer_layers": args.transformer_layers, "transformer_heads": args.transformer_heads}
'''
    assert a in s, "train.py arch dict not found"
    s = s.replace(a, a + '''    if args.arch_type == "bilstm":
        arch = {"arch_type": "bilstm", "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim, "num_layers": args.num_layers}
''')
    p.write_text(s); print("train.py patched")
else:
    print("train.py already patched")

# ---------------- attacks/unguided.py ----------------
p = SRC / "attacks" / "unguided.py"; s = p.read_text()
if "build_model" not in s:
    backup(p)
    a = "from model import CharCNNTransformer, build_char_vocab, encode_url"
    assert a in s, "unguided.py import line not found"
    s = s.replace(a, "from model import CharCNNTransformer, build_char_vocab, encode_url, build_model")
    a = "model = CharCNNTransformer(vocab_size=vocab_size, num_classes=len(CLASSES), max_len=max_len,"
    assert a in s, "unguided.py model line not found"
    s = s.replace(a, "model = build_model(vocab_size=vocab_size, num_classes=len(CLASSES), max_len=max_len,")
    p.write_text(s); print("attacks/unguided.py patched")
else:
    print("attacks/unguided.py already patched")

# ---------------- report any other place that builds the model directly ----------------
others = []
for f in SRC.rglob("*.py"):
    if f.name in ("model.py", "attribution.py") or ".venv" in f.parts:  # attribution.py: self-test only
        continue
    for i, line in enumerate(f.read_text().splitlines(), 1):
        if re.search(r"CharCNNTransformer\s*\(", line) and not line.lstrip().startswith("#"):
            others.append(f"{f}:{i}: {line.strip()}")
if others:
    print("\nNOTE: these lines still build the CNN-Transformer directly (send them to Claude):")
    print("\n".join(others))
else:
    print("\nNo other direct model construction found. Done.")
