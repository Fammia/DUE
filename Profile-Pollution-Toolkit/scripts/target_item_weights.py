"""Append per-target-movie entity weights each epoch."""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Sequence

import torch

from config import ROLE_IDS, TARGET_ITEM_COLUMNS


class TargetItemWeightRecorder:
    def __init__(self, path: str, columns: Sequence[str] = TARGET_ITEM_COLUMNS):
        self.path = path
        self.columns = list(columns)
        self.rows: List[Dict[str, float]] = []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def add_rows(self, epoch: int, rows: Sequence[Dict[str, float]]) -> None:
        for rec in rows:
            out = {"epoch": float(epoch)}
            out.update(rec)
            self.rows.append(out)
        self.flush()

    def flush(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: row.get(k, "") for k in self.columns})
        os.replace(tmp, self.path)


def snapshot_target_items(
    *,
    target_ids: Sequence[int],
    item_entity_w: torch.Tensor,
    items: torch.Tensor,
    w_final: torch.Tensor,
    role_id: torch.Tensor,
) -> List[Dict[str, float]]:
    """item_entity_w: [n_items] applied this epoch. items/w_final/role_id aligned on positives."""
    tid_role = int(ROLE_IDS["target"])
    rows: List[Dict[str, float]] = []
    n_items = int(item_entity_w.numel())
    for tid in target_ids:
        tid = int(tid)
        if 0 <= tid < n_items:
            w_ent = float(item_entity_w[tid].item())
        else:
            w_ent = float("nan")
        mask = (items.long() == tid) & (role_id.long() == tid_role)
        n = int(mask.sum().item())
        if n > 0:
            wf_mean = float(w_final[mask].mean().item())
        else:
            wf_mean = float("nan")
        rows.append(
            {
                "item_id": float(tid),
                "w_item": w_ent,
                "w_final_target_mean": wf_mean,
                "n_target_edges": float(n),
            }
        )
    return rows
