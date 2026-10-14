"""Five-way (clean / noise / filler / selected / target) epoch aggregation."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from config import ROLE_NAMES


def _mean_masked(x: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    n = int(mask.sum().item())
    if n == 0:
        return float("nan")
    return float(x[mask].mean().item())


def aggregate_five_way(
    *,
    loss: torch.Tensor,
    w_final: torch.Tensor,
    w_base: torch.Tensor,
    w_user: torch.Tensor,
    w_item: torch.Tensor,
    role_id: torch.Tensor,
    include_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    role_id: 0 noise, 1 clean, 2 filler, 3 selected, 4 target
    (see config.ROLE_IDS). include_mask: positives only for NCF.
    """
    rid = role_id.long()
    if include_mask is not None:
        inc = include_mask.bool()
        rid = torch.where(inc, rid, torch.full_like(rid, -1))

    stats: Dict[str, float] = {}
    name_to_id = {
        "noise": 0,
        "clean": 1,
        "filler": 2,
        "selected": 3,
        "target": 4,
    }
    for name in ROLE_NAMES:
        m = rid == name_to_id[name]
        stats[f"n_{name}"] = float(m.sum().item())
        stats[f"loss_{name}"] = _mean_masked(loss, m)
        stats[f"w_final_{name}"] = _mean_masked(w_final, m)
        stats[f"w_base_{name}"] = _mean_masked(w_base, m)
        stats[f"w_user_{name}"] = _mean_masked(w_user, m)
        stats[f"w_item_{name}"] = _mean_masked(w_item, m)
    return stats
