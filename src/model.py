"""
Character-level CNN-Transformer malicious URL classifier.

Architecture: char embedding -> stack of 1D conv blocks (local n-gram features,
the "CNN" part) -> Transformer encoder layers over the conv output (captures
longer-range structural dependencies across the URL) -> attention pooling ->
linear classifier head over the 4 classes.

This is a reference implementation matched to the draft's "character-level
CNN-Transformer" description in §4 -- swap in your own if you already have
one; the training/eval harness in train.py only depends on model(input_ids)
returning logits of shape [batch, num_classes].
"""
import torch
import torch.nn as nn


PAD_IDX = 0


def build_char_vocab():
    """Printable ASCII covers virtually all real URL characters; index 0 is
    reserved for padding, index 1 for any out-of-vocab byte."""
    chars = [chr(i) for i in range(32, 127)]
    vocab = {"<pad>": PAD_IDX, "<unk>": 1}
    for c in chars:
        vocab[c] = len(vocab)
    return vocab


def encode_url(url: str, vocab: dict, max_len: int) -> list:
    ids = [vocab.get(ch, vocab["<unk>"]) for ch in url[:max_len]]
    ids += [PAD_IDX] * (max_len - len(ids))
    return ids


class ConvBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x):  # x: [B, C, L]
        return self.act(self.bn(self.conv(x))) + x  # residual


class CharCNNTransformer(nn.Module):
    def __init__(self, vocab_size, num_classes=4, embed_dim=64, conv_channels=128,
                 conv_kernels=(3, 5, 7), transformer_layers=2, transformer_heads=4,
                 max_len=200, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.proj = nn.Conv1d(embed_dim, conv_channels, kernel_size=1)
        self.conv_blocks = nn.ModuleList([ConvBlock(conv_channels, k) for k in conv_kernels])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_channels, nhead=transformer_heads,
            dim_feedforward=conv_channels * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers,
            enable_nested_tensor=False,  # nested-tensor fast path isn't reliably
                                          # supported on MPS (Apple Silicon); this
                                          # keeps behavior identical across CPU/MPS/CUDA
        )

        self.attn_pool = nn.Linear(conv_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(conv_channels, num_classes)
        self.max_len = max_len

    def forward_from_embeds(self, embeds, pad_mask):
        """Same forward pass as forward(), but starting from precomputed
        embeddings instead of token ids. This is what Integrated Gradients
        (attacks/attribution.py) needs: it interpolates BETWEEN a baseline
        embedding and the real embedding, which only makes sense in
        embedding space, not in discrete token-id space."""
        x = embeds.transpose(1, 2)                     # [B, E, L]
        x = self.proj(x)                                # [B, C, L]
        for block in self.conv_blocks:
            x = block(x)
        x = x.transpose(1, 2)                            # [B, L, C]

        x = self.transformer(x, src_key_padding_mask=pad_mask)

        # attention pooling over non-pad positions
        scores = self.attn_pool(x).squeeze(-1)            # [B, L]
        scores = scores.masked_fill(pad_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
        pooled = (x * weights).sum(dim=1)                  # [B, C]

        return self.head(self.dropout(pooled))

    def forward(self, input_ids):  # [B, L]
        pad_mask = input_ids.eq(PAD_IDX)              # [B, L] True where padded
        embeds = self.embed(input_ids)                  # [B, L, E]
        return self.forward_from_embeds(embeds, pad_mask)
