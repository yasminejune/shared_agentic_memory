"""The DClip mean and Top-k+ selection that InvisibleInk samples from."""

from __future__ import annotations

import pytest
import torch

from agent_memories.agent.invisible_ink.mechanism import (
    dclip_mean,
    sample_topk_plus,
    top_k_plus_mask,
)

pytestmark = pytest.mark.unit


def test_dclip_mean_bounds_each_coordinate_within_c() -> None:
    torch.manual_seed(0)
    b, vocab, c = 4, 32, 1.5
    z_public = torch.randn(vocab, dtype=torch.float32)
    z_private = z_public.unsqueeze(0) + torch.randn(b, vocab, dtype=torch.float32) * 3.0
    phi_bar = dclip_mean(z_private, z_public, c)
    assert isinstance(phi_bar, torch.Tensor)
    assert phi_bar.dtype == torch.float32
    assert phi_bar.shape == (vocab,)
    assert torch.all(phi_bar >= z_public - c - 1e-5)
    assert torch.all(phi_bar <= z_public + c + 1e-5)


def test_dclip_mean_is_identity_when_private_equals_public() -> None:
    z_public = torch.arange(16, dtype=torch.float32)
    z_private = z_public.unsqueeze(0).expand(3, -1).contiguous()
    phi_bar = dclip_mean(z_private, z_public, c=2.0)
    assert torch.allclose(phi_bar, z_public, atol=1e-6)


def test_top_k_plus_mask_selects_at_least_k() -> None:
    torch.manual_seed(1)
    vocab, k, c, b = 64, 8, 1.0, 10
    z_public = torch.randn(vocab, dtype=torch.float32)
    mask, expansion_idxs = top_k_plus_mask(z_public, k=k, c=c, b=b)
    assert mask.dtype == torch.bool
    assert int(mask.sum().item()) >= k
    # Expansion set is the band between the Top-k threshold and the expanded one.
    assert expansion_idxs.ndim == 1


def test_top_k_plus_width_shrinks_as_c_over_b_shrinks() -> None:
    torch.manual_seed(2)
    z_public = torch.randn(128, dtype=torch.float32)
    wide, _ = top_k_plus_mask(z_public, k=10, c=5.0, b=5)
    narrow, _ = top_k_plus_mask(z_public, k=10, c=0.1, b=50)
    assert int(narrow.sum().item()) <= int(wide.sum().item())


def test_sample_topk_plus_only_returns_masked_token() -> None:
    torch.manual_seed(3)
    vocab = 20
    phi_bar = torch.randn(vocab, dtype=torch.float32)
    mask = torch.zeros(vocab, dtype=torch.bool)
    allowed = torch.tensor([2, 5, 11], dtype=torch.long)
    mask[allowed] = True
    for _ in range(30):
        tok = sample_topk_plus(phi_bar, mask, tau=1.0)
        assert tok in {2, 5, 11}


def test_top_k_plus_mask_full_vocab_when_k_covers_all() -> None:
    z_public = torch.linspace(-1.0, 1.0, 16)
    mask, expansion_idxs = top_k_plus_mask(z_public, k=16, c=1.0, b=4)
    assert bool(mask.all().item())
    assert expansion_idxs.size == 0
