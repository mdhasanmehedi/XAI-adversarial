"""
Integrated Gradients (Sundararajan, Taly & Yan, 2017) over character
embeddings, used to rank which character positions the model relies on for
its current malicious-class score -- the attribution signal that drives the
guided attack in attacks/guided.py (§5.2), and reusable later for the
faithfulness metrics in §6 (Deletion AUC / Comprehensiveness / Sufficiency
all need the same kind of per-position ranking).

Why interpolate in EMBEDDING space, not token-id space: token ids are
discrete/categorical (there's no meaningful "halfway between character 'a'
and character 'b'"), so IG's straight-line path from a baseline to the input
is defined over the continuous embedding vectors instead. Baseline = the
all-zero embedding (standard choice; equivalent to "no information" since the
model's own padding embedding is the only other natural zero-ish reference
and using it as baseline would conflate "padding" with "absence of signal").
"""
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add src/ to path
from model import PAD_IDX  # noqa: E402


def integrated_gradients(model, input_ids, target_class_idx, steps=20):
    """
    input_ids: LongTensor [1, L] (single URL, already encoded + padded)
    target_class_idx: int, the class whose score is being attributed
                       (in the guided attack, this is the URL's TRUE
                       malicious class -- i.e. "what does the model rely on
                       to believe this URL is malicious right now")
    steps: number of interpolation steps along the IG path. More steps =
           more accurate attribution but proportionally more compute (each
           step is one forward + one backward pass). 20 is a reasonable
           default; the completeness axiom (attributions should sum to
           target_score - baseline_score) gets noticeably tighter above ~20-30
           for a model this size, with diminishing returns beyond that.

    Returns: 1D numpy-free list of per-position attribution scores, length L
             (same length as input_ids' sequence dim, including padded
             positions -- callers should slice to the real URL length).
    """
    model.eval()
    device = input_ids.device
    pad_mask = input_ids.eq(PAD_IDX)

    with torch.no_grad():
        embeds = model.embed(input_ids).detach()  # [1, L, E]
    baseline = torch.zeros_like(embeds)

    total_grads = torch.zeros_like(embeds)
    for step in range(1, steps + 1):
        alpha = step / steps
        interp = baseline + alpha * (embeds - baseline)
        interp = interp.clone().requires_grad_(True)
        logits = model.forward_from_embeds(interp, pad_mask)
        score = logits[0, target_class_idx]
        grad, = torch.autograd.grad(score, interp)
        total_grads = total_grads + grad.detach()

    avg_grads = total_grads / steps
    attributions = (embeds - baseline) * avg_grads          # [1, L, E]
    attributions = attributions.sum(dim=-1).squeeze(0)        # [L]
    return attributions.detach().cpu().tolist()


if __name__ == "__main__":
    # Quick sanity check: attribution should (a) run without error, (b) be
    # concentrated on non-padded positions, (c) roughly satisfy the
    # completeness axiom (sum of attributions ~= target_score - baseline_score).
    from model import CharCNNTransformer, build_char_vocab, encode_url

    vocab = build_char_vocab()
    torch.manual_seed(0)
    model = CharCNNTransformer(vocab_size=len(vocab), num_classes=4, max_len=40).eval()

    url = "secure-login-verify.xyz/signin"
    max_len = 40
    ids = torch.tensor([encode_url(url, vocab, max_len)])

    attributions = integrated_gradients(model, ids, target_class_idx=1, steps=20)

    real_len = min(len(url), max_len)
    real_attr = attributions[:real_len]
    pad_attr = attributions[real_len:]

    with torch.no_grad():
        pad_mask = ids.eq(PAD_IDX)
        target_score = model.forward_from_embeds(model.embed(ids), pad_mask)[0, 1].item()
        baseline_score = model.forward_from_embeds(torch.zeros_like(model.embed(ids)), pad_mask)[0, 1].item()

    print(f"URL: {url!r} (length {real_len})")
    print(f"Per-character attribution (real positions): {[round(a, 3) for a in real_attr]}")
    print(f"Sum over padded positions (should be ~0): {sum(pad_attr):.4f}")
    print(f"Completeness check: sum(attributions)={sum(attributions):.4f} vs "
          f"(target_score - baseline_score)={target_score - baseline_score:.4f} "
          f"(should be reasonably close, not exact due to finite steps)")
