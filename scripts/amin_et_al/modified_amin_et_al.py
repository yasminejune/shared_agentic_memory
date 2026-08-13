"""InvisibleInk (Vinod et al., arXiv:2507.02974) smoke harness.

Mirrors Algorithm 1 from InvisibleInk / the official ``invink`` package
on the same 10 toy rows as ``WP2_3.py``, while reusing this project's
Gemma 2 IT token plumbing and WP2 round-2 ``wrap`` prompt. Production
Amin code in ``src/agent_memories/agent/privacy/`` is left untouched.

Per-token steps (paper Alg. 1 / ``invink.generate`` inner loop):

1. Prefill ``B`` sensitive prompts plus one public prompt.
2. Build Top-k+ vocabulary from public logits only:
   ``{y : φ_pub(y) ≥ ℓ − 2C/B}`` (paper ``B``, not ``invink``'s
   ``batch_size = B+1`` call-site quirk).
3. DClip-aggregate private logits:
   ``φ_bar = φ_pub + (1/B) Σ clip_C(φ_i − φ_pub)``.
4. Sample from ``softmax(φ_bar[V_k+] / τ)``; spend privacy on every
   token (no Amin SVT public path).

Clip ``C`` is calibrated from target ``(ε, δ)`` via the tight zCDP
conversion used by ``invink.utils.cdp_rho`` / ``get_clip``:
``C = B τ √(2 ρ / T)`` with ``ρ`` inverted from ``ε`` at fixed ``δ``.
"""

from __future__ import annotations

import math

import pandas as pd
import torch

from agent_memories.agent.privacy import token_generation as tg
from agent_memories.agent.privacy.prompts import wrap

B = 10  # number of private references (paper B)
TAU = 1.0  # sampling temperature
TOP_K = 100  # top-k+ truncation parameter
T = 80  # max tokens; used to calibrate C
DELTA = 1e-5  # ADP failure probability (invink default)
EPSILON = 10.0  # target (ε, δ)-DP budget (loose WP2_3-comparable demo)
LABEL = "attending a recent event"

EXAMPLES_PATH = "scripts/amin_et_al/examples.csv"


def _cdp_delta(rho: float, eps: float) -> float:
    """Tight zCDP → δ at fixed ε (invink ``cdp_delta``)."""
    if rho == 0.0:
        return 0.0
    amin, amax = 1.0001, max(2.0, (eps + 1.0) / (2.0 * rho) + 2.0)
    alpha = 0.5 * (amin + amax)
    for _ in range(1000):
        alpha = 0.5 * (amin + amax)
        log1p_term = math.log1p(-1.0 / alpha)
        derivative = (2.0 * alpha - 1.0) * rho - eps + log1p_term
        if derivative < 0.0:
            amin = alpha
        else:
            amax = alpha
    exponent = (alpha - 1.0) * (alpha * rho - eps) + alpha * math.log1p(-1.0 / alpha)
    try:
        delta = math.exp(exponent) / (alpha - 1.0)
    except OverflowError:
        delta = 0.0
    return min(max(delta, 0.0), 1.0)


def eps_from_rho(rho: float, delta: float) -> float:
    """Smallest ε such that ρ-zCDP implies (ε, δ)-DP (invink ``cdp_eps``)."""
    if rho == 0.0:
        return 0.0
    epsmin, epsmax = 0.0, rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
    for _ in range(1000):
        eps = 0.5 * (epsmin + epsmax)
        if _cdp_delta(rho, eps) <= delta:
            epsmax = eps
        else:
            epsmin = eps
    return float(epsmax)


def rho_from_eps(eps: float, delta: float) -> float:
    """Smallest ρ such that ρ-zCDP implies (ε, δ)-DP (invink ``cdp_rho``)."""
    if eps == 0.0:
        return 0.0
    rhomin, rhomax = 0.0, max(1.0, eps + 1.0)
    for _ in range(2000):
        rho = 0.5 * (rhomin + rhomax)
        if _cdp_delta(rho, eps) <= delta:
            rhomin = rho
        else:
            rhomax = rho
    return float(rhomin)


def rho_for_tokens(num_toks: int, c: float, b: int, tau: float) -> float:
    """Theorem 2: ρ_seq = T · (C / (B τ))² / 2."""
    return float(num_toks) * 0.5 * (c / (float(b) * tau)) ** 2


def clip_norm_for_budget(epsilon: float, delta: float, num_toks: int, b: int, tau: float) -> float:
    """invink ``get_clip`` with paper ``B`` (private count)."""
    rho_tot = rho_from_eps(epsilon, delta)
    rho_tok = rho_tot / float(num_toks)
    return float(tau) * float(b) * math.sqrt(max(0.0, 2.0 * rho_tok))


def dclip_aggregate(Z: torch.Tensor, z_pub: torch.Tensor, c: float) -> torch.Tensor:
    """Mean of DClip rows: φ_pub + mean_i clip_C(φ_i − φ_pub)."""
    clipped = z_pub + torch.clamp(Z - z_pub, min=-c, max=c)
    return clipped.mean(dim=0)


def expanded_top_vocab(z_pub: torch.Tensor, k: int, c: float, b: int) -> torch.Tensor:
    """Boolean mask for Top-k+: φ_pub(y) ≥ ℓ − 2C/B (paper B)."""
    if k >= z_pub.numel():
        return torch.ones_like(z_pub, dtype=torch.bool)
    # k-th largest entry of φ_pub
    ell = torch.topk(z_pub, k).values[-1]
    return z_pub >= (ell - 2.0 * c / float(b))


def sample_topk_plus(bar_phi: torch.Tensor, mask: torch.Tensor, tau: float) -> int:
    """Softmax sample over the Top-k+ support only."""
    masked = bar_phi.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(masked / tau, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_invisible_ink(
    texts: list[str],
    *,
    label: str,
    b: int,
    c: float,
    tau: float,
    top_k: int,
    max_toks: int,
) -> tuple[str, int, float]:
    """Run InvisibleInk Alg. 1; return (text, tokens_used, mean |V_k+|)."""
    stop = tg.stop_ids()
    prompts = [wrap(items=text, label=label) for text in texts]
    public_prompt = wrap(label=label)
    prompt_ids = [tg.encode_chat(p) for p in prompts]
    public_ids = tg.encode_chat(public_prompt)

    logits, state = tg.prefill_padded(prompt_ids + [public_ids])
    x_ids: list[int] = []
    topk_sizes: list[int] = []

    while len(x_ids) < max_toks:
        Z = logits[: len(prompt_ids)]
        z_pub = logits[len(prompt_ids)]

        mask = expanded_top_vocab(z_pub, top_k, c, b)
        topk_sizes.append(int(mask.sum().item()))
        bar_phi = dclip_aggregate(Z, z_pub, c)
        tok = sample_topk_plus(bar_phi, mask, tau)
        x_ids.append(tok)
        if tok in stop:
            break

        logits, state = tg.continue_batched(state, [tok] * (len(prompt_ids) + 1))

    mean_vk = float(sum(topk_sizes) / len(topk_sizes)) if topk_sizes else 0.0
    return tg.decode(x_ids), len(x_ids), mean_vk


def main() -> None:
    df = pd.read_csv(EXAMPLES_PATH)
    items_blocks = df["text"].tolist()
    if len(items_blocks) != B:
        raise SystemExit(f"Expected {B} toy rows in {EXAMPLES_PATH}, got {len(items_blocks)}.")

    c = clip_norm_for_budget(EPSILON, DELTA, T, B, TAU)
    rho_budget = rho_for_tokens(T, c, B, TAU)
    eps_budget = eps_from_rho(rho_budget, DELTA)

    print(f"InvisibleInk harness on {EXAMPLES_PATH}")
    print(f"  B={B}, T={T}, tau={TAU}, top_k={TOP_K}")
    print(f"  target epsilon={EPSILON}, delta={DELTA}")
    print(f"  calibrated C={c:.6f}")
    print(f"  budgeted rho(T)={rho_budget:.6f}, epsilon(T)={eps_budget:.4f}")

    synthetic, n_toks, mean_vk = generate_invisible_ink(
        items_blocks,
        label=LABEL,
        b=B,
        c=c,
        tau=TAU,
        top_k=TOP_K,
        max_toks=T,
    )

    rho_spent = rho_for_tokens(max(n_toks, 1), c, B, TAU) if n_toks > 0 else 0.0
    eps_spent = eps_from_rho(rho_spent, DELTA) if n_toks > 0 else 0.0

    print()
    print("Synthetic text:")
    print(repr(synthetic))
    print()
    print("Privacy account:")
    print(f"  tokens generated       = {n_toks} / {T}")
    print(f"  mean |V_k+|            = {mean_vk:.1f}")
    print(f"  budgeted epsilon (T)   = {eps_budget:.4f}")
    print(f"  spent epsilon (len)    = {eps_spent:.4f}")
    print(f"  budgeted rho (T)       = {rho_budget:.6f}")
    print(f"  spent rho (len)        = {rho_spent:.6f}")


if __name__ == "__main__":
    main()
