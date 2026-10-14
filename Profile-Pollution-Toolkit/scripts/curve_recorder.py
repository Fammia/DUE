"""Record one row per epoch, using an atomic replacement for each CSV flush."""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Sequence

from config import CURVE_COLUMNS


class CurveRecorder:
    def __init__(
        self,
        path: str,
        columns: Sequence[str] = CURVE_COLUMNS,
        *,
        enabled: bool = True,
    ):
        self.path = path
        self.columns = list(columns)
        self.enabled = bool(enabled)
        self.rows: List[Dict[str, float]] = []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def add(self, epoch: int, stats: Dict[str, float]) -> None:
        if not self.enabled:
            return
        row = {"epoch": float(epoch)}
        row.update(stats)
        self.rows.append(row)
        self.flush()

    def flush(self) -> None:
        if not self.enabled:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: row.get(k, "") for k in self.columns})
        os.replace(tmp, self.path)
