"""DClip and Top-k+ primitives for InvisibleInk Algorithm 1."""

from __future__ import annotations

import numpy as np
import torch
from invink.utils import difference_clip, get_topk


def dclip_mean(Z: torch.Tensor, z_public: torch.Tensor, c: float) -> torch.Tensor:  # noqa: N803
    """Mean of DClip rows: phi_pub + mean_i clip_C(phi_i - phi_pub).

    Changes inputs to a float32 before the clipping since NumPy
    cannot promote the three full-size temporaries to
    float64. Returns a float32 CPU tensor of shape (vocab,).
    """
    z32 = Z.detach().to(dtype=torch.float32)
    pub32 = z_public.detach().to(dtype=torch.float32)
    clipped = difference_clip(logit=z32, publogit=pub32, clip_norm=c)
    mean_np = np.mean(clipped, axis=0).astype(np.float32, copy=False)
    return torch.from_numpy(mean_np)


def top_k_plus_mask(
    z_public: torch.Tensor,
    *,
    k: int,
    c: float,
    b: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Boolean Top-k+ mask and expansion-band indices from public logits.

    When k covers the full vocabulary the mask is all-ones and the
    expansion index array is empty.
    Otherwise calls get_topk in line with the paper.
    Returns (mask, expansion_idxs), where expansion_idxs
    are the indices between the top-k threshold and the expanded
    threshold, used for the expansion_set_count metric.
    """
    pub_np = z_public.detach().to(dtype=torch.float32).cpu().numpy()
    if k >= pub_np.size:
        mask = torch.ones(pub_np.shape, dtype=torch.bool)
        return mask, np.array([], dtype=np.int64)
    mask_np, expansion_idxs = get_topk(pub_logits=pub_np, k=k, clip=c, batch=b)
    return torch.from_numpy(np.asarray(mask_np, dtype=bool)), np.asarray(expansion_idxs)


def sample_topk_plus(phi_bar: torch.Tensor, mask: torch.Tensor, tau: float) -> int:
    """Softmax-sample one token from the Top-k+ support only."""
    masked = phi_bar.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(masked / tau, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())
