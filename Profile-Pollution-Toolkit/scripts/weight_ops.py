"""Production DUE weights used by the profile-pollution evaluation runners."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def get_based_weight(loss_values: torch.Tensor) -> torch.Tensor:
    """Right-rank ECDF plotting position; callers supply negated entry losses."""
    if loss_values.numel() == 0:
        return torch.ones_like(loss_values)
    sorted_losses, _ = torch.sort(loss_values)
    ranks = torch.searchsorted(sorted_losses, loss_values, right=True).float()
    n_samples = loss_values.numel()
    return (ranks - 0.5) / n_samples


def production_get_weight(
    avg_loss: torch.Tensor,
    startValue: float,
    interaction_cnts: Optional[torch.Tensor] = None,
    floor_ratio: float = 0.0,
) -> torch.Tensor:
    """Loss-rank entity factors with support-gated asymmetric shrinkage."""
    effective_end = startValue * floor_ratio
    sorted_indices = torch.argsort(avg_loss)
    n = avg_loss.numel()
    weights = torch.linspace(
        startValue,
        effective_end,
        steps=n,
        device=avg_loss.device,
        dtype=avg_loss.dtype,
    )
    result = torch.zeros_like(avg_loss)
    result[sorted_indices] = weights

    if interaction_cnts is not None:
        neutral = (startValue + effective_end) * 0.5
        prior_n = interaction_cnts.median().clamp(min=1.0)
        half_range = abs(startValue - effective_end) * 0.5
        if half_range > 1e-8:
            extremity = (result - neutral).abs() / half_range
            suppression_depth = ((neutral - result) / half_range).clamp(min=0.0)
            risk = extremity * (1.0 + suppression_depth)
        else:
            risk = torch.zeros_like(result)
        effective_prior = prior_n * (1.0 + risk)
        raw_confidence = interaction_cnts / (interaction_cnts + effective_prior)
        confidence = raw_confidence.sqrt()
        result = confidence * result + (1.0 - confidence) * neutral
    return result


def fuse_with_switches(
    w_base: torch.Tensor,
    w_user: torch.Tensor,
    w_item: torch.Tensor,
    u_indices: torch.Tensor,
    i_indices: torch.Tensor,
    *,
    use_bw: bool,
    use_uf: bool,
    use_if: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse entry/user/item factors; a disabled factor is replaced by one."""
    if not use_bw:
        w_base = torch.ones_like(w_base)
    if not use_uf:
        w_user = torch.ones_like(w_user)
    if not use_if:
        w_item = torch.ones_like(w_item)
    w_u = w_user[u_indices]
    w_i = w_item[i_indices]
    return w_base * w_u * w_i, w_base, w_u, w_i
