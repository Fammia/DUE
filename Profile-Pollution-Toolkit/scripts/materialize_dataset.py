#!/usr/bin/env python3
"""Materialize polluted training splits without overwriting the clean inputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from data_io import split_paths  # noqa: E402
from config import (  # noqa: E402
    BUILDERS,
    DATASET_NAME,
    ORIG_DATA_DIR,
    budget_n,
    data_dir,
    extra_stems,
    hosts_path,
    parse_stem,
    profiles_path,
)


def _link_or_copy(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_train(path: str, rows) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for u, i, k in rows:
            f.write(f"{int(u)}\t{int(i)}\t{int(k)}\n")
    os.replace(tmp, path)


def _write_roles(path: str, rows) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("user\titem\trole\ttarget_item\n")
        for u, i, role, t in rows:
            f.write(f"{int(u)}\t{int(i)}\t{role}\t{int(t)}\n")
    os.replace(tmp, path)


def materialize_one(stem: str, orig_dir: str) -> None:
    parsed = parse_stem(stem)
    if parsed is None:
        raise ValueError(stem)
    builder, inject, budget = parsed
    with open(profiles_path(builder), "r", encoding="utf-8") as f:
        prof_payload = json.load(f)
    profiles = prof_payload["profiles"]
    keep_n = budget_n(budget)
    profiles = [p for p in profiles if int(p["profile_index"]) < keep_n]

    train_src, valid_src, test_src = split_paths(orig_dir)
    out_dir = data_dir(stem)
    os.makedirs(out_dir, exist_ok=True)

    orig_rows = []
    with open(train_src, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            u, i, k = line.strip().split("\t")[:3]
            orig_rows.append((int(u), int(i), int(k)))
    orig_pairs = {(u, i) for u, i, _ in orig_rows}

    fake_rows = []
    role_rows = []
    if inject == "fakeuser":
        for p in profiles:
            uid = int(p["uid"])
            t = int(p["target_item"])
            for rec in p["items"]:
                item, role = int(rec["item"]), rec["role"]
                fake_rows.append((uid, item, 2))
                role_rows.append((uid, item, role, t))
    else:
        with open(hosts_path(builder), "r", encoding="utf-8") as f:
            host_payload = json.load(f)
        hosts = host_payload["hosts"]
        hosts = [h for h in hosts if int(h["profile_index"]) < keep_n]
        for h in hosts:
            uid = int(h["host_uid"])
            t = int(h["target_item"])
            for rec in h["items"]:
                item, role = int(rec["item"]), rec["role"]
                if (uid, item) in orig_pairs:
                    raise RuntimeError(
                        f"collision not resolved: user={uid} item={item} stem={stem}"
                    )
                fake_rows.append((uid, item, 2))
                role_rows.append((uid, item, role, t))

    # per-target target-role count
    from collections import Counter

    tgt_counts = Counter(
        t for _, _, role, t in role_rows if role == "target"
    )
    expect = keep_n
    for t, c in tgt_counts.items():
        if c != expect:
            raise RuntimeError(f"{stem} target {t} inserted {c} times, expected {expect}")
    if inject == "pollute":
        n_a = sum(len(p["items"]) for p in profiles)
        if n_a != len(fake_rows):
            raise RuntimeError(f"{stem} fake edges {len(fake_rows)} != 做法一 {n_a}")

    out_train = os.path.join(out_dir, f"{DATASET_NAME}.train.rating")
    _write_train(out_train, orig_rows + fake_rows)
    _write_roles(os.path.join(out_dir, "fake_roles.csv"), role_rows)
    _link_or_copy(valid_src, os.path.join(out_dir, f"{DATASET_NAME}.valid.rating"))
    _link_or_copy(test_src, os.path.join(out_dir, f"{DATASET_NAME}.test.negative"))
    meta = {
        "stem": stem,
        "builder": builder,
        "inject": inject,
        "budget": budget,
        "n_orig_train": len(orig_rows),
        "n_fake": len(fake_rows),
        "n_target_inserts": dict(tgt_counts),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(
        f"{stem}: orig={len(orig_rows)} fake={len(fake_rows)} "
        f"users_max={max(u for u, _, _ in orig_rows + fake_rows)} -> {out_dir}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=ORIG_DATA_DIR)
    p.add_argument("--stems", nargs="*", default=None)
    args = p.parse_args()
    stems = args.stems or extra_stems()
    if os.path.abspath(args.data_dir) != os.path.abspath(ORIG_DATA_DIR):
        print("[warn] orig data_dir is not the canonical MovieLens path")
    for stem in stems:
        materialize_one(stem, args.data_dir)
    print(f"done {len(stems)} datasets")


if __name__ == "__main__":
    main()
