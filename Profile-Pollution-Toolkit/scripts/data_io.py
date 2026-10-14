"""Load MovieLens splits and movies.dat without touching the three split files."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd

from config import DATASET_NAME, ORIG_DATA_DIR


def split_paths(data_dir: str = ORIG_DATA_DIR, dataset: str = DATASET_NAME):
    return (
        os.path.join(data_dir, f"{dataset}.train.rating"),
        os.path.join(data_dir, f"{dataset}.valid.rating"),
        os.path.join(data_dir, f"{dataset}.test.negative"),
    )


def load_triplets(path: str) -> List[Tuple[int, int, int]]:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["user", "item", "kind"],
        usecols=[0, 1, 2],
        dtype={0: int, 1: int, 2: int},
    )
    return list(df.itertuples(index=False, name=None))


def load_test_pos(path: str) -> Dict[int, List[int]]:
    pos: Dict[int, List[int]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            arr = line.split("\t")
            u = int(arr[0])
            i = int(arr[1])
            pos[u].append(i)
    return dict(pos)


def load_movies(path: str) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            mid_s, title, genres_s = line.split("::")
            mid = int(mid_s)
            genres = [g for g in genres_s.split("|") if g and g != "(no genres listed)"]
            out[mid] = {"title": title, "genres": genres}
    return out


def user_item_sets(triplets) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = defaultdict(set)
    for u, i, _ in triplets:
        out[int(u)].add(int(i))
    return dict(out)


def load_original_bundle(data_dir: str = ORIG_DATA_DIR):
    train_p, valid_p, test_p = split_paths(data_dir)
    train = load_triplets(train_p)
    valid = load_triplets(valid_p)
    test_pos = load_test_pos(test_p)
    movies_path = os.path.join(data_dir, "movies.dat")
    movies = load_movies(movies_path)

    catalog: Set[int] = set()
    consumed: Dict[int, Set[int]] = defaultdict(set)
    train_items: Dict[int, Set[int]] = defaultdict(set)
    for u, i, k in train:
        catalog.add(i)
        consumed[u].add(i)
        train_items[u].add(i)
    for u, i, k in valid:
        catalog.add(i)
        consumed[u].add(i)
    for u, items in test_pos.items():
        for i in items:
            catalog.add(i)
            consumed[u].add(i)

    item_pop: Dict[int, int] = defaultdict(int)
    for u, i, k in train:
        item_pop[i] += 1

    degrees = [len(s) for s in train_items.values()]
    return {
        "train": train,
        "valid": valid,
        "test_pos": test_pos,
        "movies": movies,
        "catalog": catalog,
        "consumed": dict(consumed),
        "train_items": dict(train_items),
        "item_pop": dict(item_pop),
        "degrees": degrees,
        "n_users": max(consumed) + 1,
        "n_items": max(catalog) + 1,
        "paths": (train_p, valid_p, test_p),
    }
