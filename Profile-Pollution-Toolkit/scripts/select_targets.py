#!/usr/bin/env python3
"""Pick 10 cold MovieLens movies that have genres.

Confirmation that T-HR@50 is ~0 on GMF/NeuMF/LightGCN happens in the
6 no-attack score runs (same scoring code). This script only chooses
candidates from the original split + movies.dat.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from data_io import load_original_bundle  # noqa: E402
from config import (  # noqa: E402
    N_TARGETS,
    ORIG_DATA_DIR,
    SEED,
    meta_dir,
    targets_path,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=ORIG_DATA_DIR)
    p.add_argument("--n_targets", type=int, default=N_TARGETS)
    p.add_argument("--min_train", type=int, default=5)
    p.add_argument("--max_train", type=int, default=40)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    bundle = load_original_bundle(args.data_dir)
    movies = bundle["movies"]
    item_pop = bundle["item_pop"]
    catalog = bundle["catalog"]

    cands = []
    for mid in catalog:
        info = movies.get(mid)
        if not info or not info["genres"]:
            continue
        pop = int(item_pop.get(mid, 0))
        if pop < args.min_train or pop > args.max_train:
            continue
        cands.append((pop, mid, info["title"], info["genres"]))
    cands.sort(key=lambda x: (x[0], x[1]))
    if len(cands) < args.n_targets:
        raise SystemExit(
            f"only {len(cands)} items with train count in "
            f"[{args.min_train}, {args.max_train}] and genres; relax bounds"
        )

    # Greedy genre diversity: skip if this exact genre-set already taken twice.
    picked = []
    genre_set_count = {}
    for pop, mid, title, genres in cands:
        key = tuple(sorted(genres))
        if genre_set_count.get(key, 0) >= 2:
            continue
        picked.append(
            {
                "item": int(mid),
                "title": title,
                "genres": list(genres),
                "train_count": int(pop),
            }
        )
        genre_set_count[key] = genre_set_count.get(key, 0) + 1
        if len(picked) >= args.n_targets:
            break
    if len(picked) < args.n_targets:
        for pop, mid, title, genres in cands:
            if any(x["item"] == mid for x in picked):
                continue
            picked.append(
                {
                    "item": int(mid),
                    "title": title,
                    "genres": list(genres),
                    "train_count": int(pop),
                }
            )
            if len(picked) >= args.n_targets:
                break

    picked = picked[: args.n_targets]
    payload = {
        "seed": args.seed,
        "min_train": args.min_train,
        "max_train": args.max_train,
        "n_targets": len(picked),
        "targets": picked,
        "item_ids": [x["item"] for x in picked],
        "note": (
            "T-HR@50 ≈ 0 must be confirmed on GMF/NeuMF/LightGCN "
            "by the 6 no-attack score jobs, not by this heuristic alone."
        ),
    }
    os.makedirs(meta_dir(), exist_ok=True)
    path = targets_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"wrote {path}")
    for i, row in enumerate(picked):
        print(
            f"  [{i}] item={row['item']} train_count={row['train_count']} "
            f"genres={','.join(row['genres'])}  {row['title']}"
        )


if __name__ == "__main__":
    main()
