#!/usr/bin/env python3
"""Build 60 watch-lists per target movie; n10–n50 are prefixes of n60.

Does not write train.rating. Host assignment for host-profile pollution is stored so
materialize can copy the same lists onto real users.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from data_io import load_original_bundle  # noqa: E402
from config import (  # noqa: E402
    BUILDERS,
    MIN_PROFILE_LEN,
    ORIG_DATA_DIR,
    PROFILES_PER_TARGET,
    SEED,
    SELECTED_K,
    fake_uid,
    hosts_path,
    patches_dir,
    profiles_path,
    targets_path,
)


def _load_targets(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    ids = [int(x) for x in payload["item_ids"]]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate targets")
    return ids


def _sample_length(rng: np.random.RandomState, degrees: List[int]) -> int:
    degs = [int(d) for d in degrees if int(d) >= MIN_PROFILE_LEN]
    if not degs:
        return MIN_PROFILE_LEN
    for _ in range(10000):
        L = int(rng.choice(degs))
        if L >= MIN_PROFILE_LEN:
            return L
    return max(MIN_PROFILE_LEN, int(np.median(degs)))


def _take_unique(cands: List[int], k: int, banned: Set[int]) -> List[int]:
    out = []
    for x in cands:
        if x in banned:
            continue
        out.append(int(x))
        if len(out) >= k:
            break
    return out


def _random_fill(
    rng: np.random.RandomState,
    catalog: List[int],
    n: int,
    banned: Set[int],
) -> List[int]:
    if n <= 0:
        return []
    pool = [x for x in catalog if x not in banned]
    if len(pool) < n:
        raise RuntimeError(f"not enough catalog items to sample {n} fillers")
    rng.shuffle(pool)
    return [int(x) for x in pool[:n]]


def build_profile(
    *,
    rng: np.random.RandomState,
    builder: str,
    target: int,
    length: int,
    catalog: List[int],
    popular_desc: List[int],
    genre_popular: Dict[int, List[int]],
) -> List[dict]:
    if length < MIN_PROFILE_LEN:
        raise RuntimeError("length < 2")
    items = [{"item": int(target), "role": "target"}]
    banned = {int(target)}
    n_sel = 0 if builder == "random" else min(SELECTED_K, length - 1)
    if builder == "bandwagon":
        selected = _take_unique(popular_desc, n_sel, banned)
    elif builder == "segment":
        selected = _take_unique(genre_popular[int(target)], n_sel, banned)
        if not selected and n_sel > 0:
            # no in-genre neighbors: fall back to popular (still not empty selected
            # only if n_sel>0). If still empty, leave selected empty and dump NaN.
            selected = _take_unique(popular_desc, n_sel, banned)
    else:
        selected = []
    for i in selected:
        items.append({"item": int(i), "role": "selected"})
        banned.add(int(i))
    n_fill = length - 1 - len(selected)
    if n_fill < 0:
        raise RuntimeError("selected longer than profile")
    for i in _random_fill(rng, catalog, n_fill, banned):
        items.append({"item": int(i), "role": "filler"})
        banned.add(int(i))
    if len(items) != length:
        raise RuntimeError(f"profile len {len(items)} != {length}")
    if len({x["item"] for x in items}) != length:
        raise RuntimeError("duplicate items in profile")
    return items


def assign_hosts(
    rng: np.random.RandomState,
    profiles: List[dict],
    consumed: Dict[int, Set[int]],
    catalog: List[int],
    n_real: int,
) -> List[dict]:
    used_hosts: Set[int] = set()
    hosts = []
    catalog_arr = list(catalog)
    for prof in profiles:
        target = int(prof["target_item"])
        eligible = [
            u
            for u in range(n_real)
            if u not in used_hosts and target not in consumed.get(u, set())
        ]
        if not eligible:
            raise RuntimeError(f"no host left for target {target}")
        host = int(eligible[rng.randint(0, len(eligible))])
        used_hosts.add(host)
        host_seen = set(consumed.get(host, set()))
        new_items = []
        used_in_profile = set()
        for rec in prof["items"]:
            item, role = int(rec["item"]), rec["role"]
            if role == "target":
                if item in host_seen:
                    raise RuntimeError("host already consumed target (leakage)")
                new_items.append({"item": item, "role": role, "replaced": False})
                used_in_profile.add(item)
                continue
            if item in host_seen or item in used_in_profile:
                banned = set(host_seen) | used_in_profile | {target}
                repl = _random_fill(rng, catalog_arr, 1, banned)
                if not repl:
                    raise RuntimeError("cannot replace colliding filler/selected")
                new_items.append(
                    {"item": int(repl[0]), "role": role, "replaced": True, "orig": item}
                )
                used_in_profile.add(int(repl[0]))
            else:
                new_items.append({"item": item, "role": role, "replaced": False})
                used_in_profile.add(item)
        hosts.append(
            {
                "uid": prof["uid"],
                "host_uid": host,
                "target_item": target,
                "target_slot": prof["target_slot"],
                "profile_index": prof["profile_index"],
                "items": new_items,
            }
        )
    return hosts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=ORIG_DATA_DIR)
    p.add_argument("--targets", default=targets_path())
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    targets = _load_targets(args.targets)
    bundle = load_original_bundle(args.data_dir)
    catalog = sorted(bundle["catalog"])
    degrees = bundle["degrees"]
    item_pop = bundle["item_pop"]
    movies = bundle["movies"]
    consumed = bundle["consumed"]
    n_real = bundle["n_users"]
    if n_real != 6040:
        print(f"[warn] n_users={n_real} (expected 6040)")

    popular_desc = sorted(catalog, key=lambda i: (-item_pop.get(i, 0), i))
    genre_of = {mid: set(movies[mid]["genres"]) for mid in catalog if mid in movies}

    genre_popular: Dict[int, List[int]] = {}
    for t in targets:
        tgen = genre_of.get(int(t), set())
        same = [
            i
            for i in popular_desc
            if i != t and genre_of.get(i, set()) & tgen
        ]
        genre_popular[int(t)] = same
        if not same:
            print(f"[warn] target {t} has no in-genre neighbors")

    os.makedirs(patches_dir(), exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # Draw lengths once, shared by all builders, so n10 prefix is nested
    # across methods on the same (target, profile_index).
    lengths = []
    for slot, t in enumerate(targets):
        row = []
        for pidx in range(PROFILES_PER_TARGET):
            L = _sample_length(rng, degrees)
            L = min(L, len(catalog) - 1)
            row.append(int(L))
        lengths.append(row)

    for builder in BUILDERS:
        profiles = []
        for slot, t in enumerate(targets):
            for pidx in range(PROFILES_PER_TARGET):
                items = build_profile(
                    rng=rng,
                    builder=builder,
                    target=int(t),
                    length=lengths[slot][pidx],
                    catalog=catalog,
                    popular_desc=popular_desc,
                    genre_popular=genre_popular,
                )
                profiles.append(
                    {
                        "builder": builder,
                        "target_item": int(t),
                        "target_slot": slot,
                        "profile_index": pidx,
                        "uid": fake_uid(slot, pidx),
                        "items": items,
                    }
                )
        hosts = assign_hosts(rng, profiles, consumed, catalog, n_real)

        # alignment: 做法一 vs host-profile pollution edge counts (after replacement, host-profile pollution
        # still one edge per list item)
        n_edges_a = sum(len(p["items"]) for p in profiles)
        n_edges_b = sum(len(h["items"]) for h in hosts)
        if n_edges_a != n_edges_b:
            raise RuntimeError(f"{builder} edge count mismatch {n_edges_a} vs {n_edges_b}")
        per_t_a = defaultdict(int)
        per_t_b = defaultdict(int)
        for p in profiles:
            per_t_a[p["target_item"]] += 1
        for h in hosts:
            per_t_b[h["target_item"]] += 1
        for t in targets:
            if per_t_a[t] != PROFILES_PER_TARGET or per_t_b[t] != PROFILES_PER_TARGET:
                raise RuntimeError(f"{builder} target {t} profile count")

        # leakage: host must not have target in original consumed
        for h in hosts:
            if h["target_item"] in consumed.get(h["host_uid"], set()):
                raise RuntimeError("leakage assert failed")
            if len({h2["host_uid"] for h2 in hosts}) != len(hosts):
                raise RuntimeError("duplicate hosts")

        with open(profiles_path(builder), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seed": args.seed,
                    "builder": builder,
                    "targets": targets,
                    "n_profiles": len(profiles),
                    "n_edges": n_edges_a,
                    "lengths_shared": lengths,
                    "profiles": profiles,
                },
                f,
            )
        with open(hosts_path(builder), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seed": args.seed,
                    "builder": builder,
                    "n_hosts": len(hosts),
                    "n_edges": n_edges_b,
                    "hosts": hosts,
                },
                f,
            )
        n_sel = sum(1 for p in profiles for x in p["items"] if x["role"] == "selected")
        print(
            f"{builder}: profiles={len(profiles)} edges={n_edges_a} "
            f"selected_edges={n_sel} hosts={len(hosts)} -> {profiles_path(builder)}"
        )


if __name__ == "__main__":
    main()
