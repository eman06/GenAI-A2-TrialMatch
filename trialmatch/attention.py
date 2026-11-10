"""Attention kernels used by the decoder (Task 4).

Three interchangeable backends, all computing *causal* scaled dot-product
attention for a block of queries that sit at the END of the key sequence
(the situation in every decoder forward pass, with or without a KV cache):

    query i (0-based inside the block) has absolute position  q_start + i
    key   j                            has absolute position  j
    query i may attend key j  <=>  j <= q_start + i

where ``q_start = S - T`` (S = #keys incl. cache, T = #queries).

* ``reference_attention``  - explicit scores matrix, explicit max-subtracted softmax.
* ``tiled_attention``      - FlashAttention-style blocked computation with an
                             online (running) softmax. Never materialises the
                             full T x S score matrix.
* ``sdpa_attention``       - PyTorch's fused ``F.scaled_dot_product_attention``
                             (used as the production / speed baseline).

Shapes: q (B, H, T, D), k/v (B, H, S, D) -> out (B, H, T, D).
GQA head expansion is done by the caller.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def causal_mask(T: int, S: int, device=None) -> torch.Tensor:
    """Boolean (T, S) mask, True = query may attend key."""
    q_start = S - T
    qi = torch.arange(T, device=device)[:, None] + q_start
    kj = torch.arange(S, device=device)[None, :]
    return kj <= qi


def reference_attention(q, k, v, scale: float | None = None) -> torch.Tensor:
    """Textbook softmax(QK^T / sqrt(d) + mask) V with an explicit stable softmax."""
    T, S, D = q.shape[-2], k.shape[-2], q.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale            # (B,H,T,S)
    mask = causal_mask(T, S, q.device)
    scores = scores.masked_fill(~mask, float("-inf"))
    # numerically stable softmax: subtract the row max before exponentiating
    row_max = scores.amax(dim=-1, keepdim=True)
    probs = torch.exp(scores - row_max)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.matmul(probs, v)


def tiled_attention(q, k, v, scale: float | None = None,
                    block_q: int = 64, block_k: int = 64) -> torch.Tensor:
    """Blocked causal attention with an online softmax (FlashAttention-1 forward).

    For every query tile we stream over key/value tiles and keep, per query row,
        m   - running maximum of the scores seen so far
        l   - running softmax normaliser  sum_j exp(s_j - m)
        acc - running un-normalised output sum_j exp(s_j - m) v_j
    When a new tile raises the maximum from m to m', the old statistics are
    rescaled by exp(m - m') so that everything stays expressed relative to
    the current maximum. The output is acc / l after the last tile.
    Key tiles lying completely above the causal diagonal are skipped.
    """
    B, H, T, D = q.shape
    S = k.shape[-2]
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    q_start = S - T
    out = torch.empty_like(q)
    neg_inf = float("-inf")
    for qs in range(0, T, block_q):
        qe = min(qs + block_q, T)
        qt = q[:, :, qs:qe] * scale                                    # (B,H,bq,D)
        bq = qe - qs
        m = torch.full((B, H, bq, 1), neg_inf, dtype=torch.float32, device=q.device)
        l = torch.zeros((B, H, bq, 1), dtype=torch.float32, device=q.device)
        acc = torch.zeros((B, H, bq, D), dtype=torch.float32, device=q.device)
        q_abs_last = q_start + qe - 1                                   # last visible key for this tile
        for ks in range(0, min(S, q_abs_last + 1), block_k):
            ke = min(ks + block_k, S)
            s = torch.matmul(qt, k[:, :, ks:ke].transpose(-1, -2)).float()  # (B,H,bq,bk)
            # causal mask only needed when the tile straddles the diagonal
            if ke - 1 > q_start + qs:
                qi = torch.arange(qs, qe, device=q.device)[:, None] + q_start
                kj = torch.arange(ks, ke, device=q.device)[None, :]
                s = s.masked_fill(kj > qi, neg_inf)
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            # guard rows that have seen nothing yet (m_new = -inf) against inf - inf = nan
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            p = torch.exp(s - m_safe)
            corr = torch.exp(m - m_safe)
            l = l * corr + p.sum(dim=-1, keepdim=True)
            acc = acc * corr + torch.matmul(p.to(v.dtype), v[:, :, ks:ke]).float()
            m = m_new
        out[:, :, qs:qe] = (acc / l).to(q.dtype)
    return out


def sdpa_attention(q, k, v, scale: float | None = None) -> torch.Tensor:
    T, S = q.shape[-2], k.shape[-2]
    if T == S:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    if T == 1:  # single decode step: every cached key is visible
        return F.scaled_dot_product_attention(q, k, v, scale=scale)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask(T, S, q.device), scale=scale)


BACKENDS = {
    "reference": reference_attention,
    "tiled": tiled_attention,
    "sdpa": sdpa_attention,
}


def attention(q, k, v, backend: str = "sdpa") -> torch.Tensor:
    try:
        fn = BACKENDS[backend]
    except KeyError as e:
        raise ValueError(f"unknown attention backend {backend!r}; choose from {list(BACKENDS)}") from e
    return fn(q, k, v)
