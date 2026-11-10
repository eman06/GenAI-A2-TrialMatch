"""E1-E3: reference / tiled / fused attention agree under causal masking."""
import pytest
import torch

from trialmatch.attention import causal_mask, reference_attention, sdpa_attention, tiled_attention


@pytest.mark.parametrize("B,H,T,S,D", [(1, 1, 1, 1, 8), (1, 2, 7, 7, 16), (2, 3, 33, 33, 64),
                                       (1, 4, 5, 40, 32), (1, 15, 1, 129, 64), (2, 5, 70, 70, 64),
                                       (1, 3, 100, 260, 64)])
@pytest.mark.parametrize("block", [(16, 16), (64, 32), (128, 128)])
def test_tiled_matches_reference(B, H, T, S, D, block):
    torch.manual_seed(0)
    q, k, v = torch.randn(B, H, T, D), torch.randn(B, H, S, D), torch.randn(B, H, S, D)
    gold = reference_attention(q.double(), k.double(), v.double())
    out = tiled_attention(q, k, v, block_q=block[0], block_k=block[1])
    assert torch.allclose(out.double(), gold, atol=1e-5, rtol=1e-4), (out.double() - gold).abs().max()
    assert torch.allclose(sdpa_attention(q, k, v).double(), gold, atol=1e-5, rtol=1e-4)


def test_reference_matches_naive_softmax():
    torch.manual_seed(1)
    q, k, v = torch.randn(1, 2, 9, 8), torch.randn(1, 2, 9, 8), torch.randn(1, 2, 9, 8)
    s = q @ k.transpose(-1, -2) / 8 ** 0.5
    s = s.masked_fill(~causal_mask(9, 9), float("-inf"))
    assert torch.allclose(reference_attention(q, k, v), torch.softmax(s, -1) @ v, atol=1e-6)


def test_causality_future_keys_do_not_matter():
    torch.manual_seed(2)
    q, k, v = torch.randn(1, 2, 6, 16), torch.randn(1, 2, 6, 16), torch.randn(1, 2, 6, 16)
    k2, v2 = k.clone(), v.clone()
    k2[..., 4:, :] += 100
    v2[..., 4:, :] -= 50                                   # perturb keys/values 4 and 5
    a = tiled_attention(q, k, v, block_q=2, block_k=2)
    b = tiled_attention(q, k2, v2, block_q=2, block_k=2)
    assert torch.allclose(a[..., :4, :], b[..., :4, :], atol=1e-6)   # queries 0..3 unaffected


def test_online_softmax_stable_for_huge_logits():
    torch.manual_seed(3)
    q, k, v = torch.randn(1, 1, 50, 32) * 100, torch.randn(1, 1, 50, 32), torch.randn(1, 1, 50, 32)
    out = tiled_attention(q, k, v, block_q=8, block_k=8)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.double(), reference_attention(q.double(), k.double(), v.double()), atol=1e-4)
