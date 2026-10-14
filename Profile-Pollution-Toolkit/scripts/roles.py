"""Map (user, item) → role id for dump stats."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

from config import ROLE_IDS


def load_role_lookup(data_path: str) -> Dict[Tuple[int, int], int]:
    path = os.path.join(data_path, "fake_roles.csv")
    lookup: Dict[Tuple[int, int], int] = {}
    if not os.path.isfile(path):
        return lookup
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            if not line.strip():
                continue
            u, i, role, *_ = line.strip().split("\t")
            lookup[(int(u), int(i))] = ROLE_IDS[role]
    return lookup


def role_ids_for_features(
    features: List[List[int]],
    kinds: List[int],
    lookup: Dict[Tuple[int, int], int],
) -> List[int]:
    out = []
    for (u, i), k in zip(features, kinds):
        if k == 1:
            out.append(ROLE_IDS["clean"])
        elif k == 0:
            out.append(ROLE_IDS["noise"])
        else:
            out.append(lookup.get((int(u), int(i)), ROLE_IDS["filler"]))
    return out
