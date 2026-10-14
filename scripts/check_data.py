#!/usr/bin/env python3
"""Check expected input files and, optionally, the paper-split SHA-256 hashes."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("movielens", "yelp", "amazon_book")
MANIFEST = ROOT / "data" / "paper_split_sha256.txt"


def manifest_entries() -> list[tuple[str, Path]]:
    entries = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, relative = line.split("  ", 1)
        entries.append((digest, ROOT / relative))
    return entries


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_line_is_valid(path: Path) -> bool:
    if path.name == "movies.dat":
        return path.stat().st_size > 0
    required_columns = 2 if path.name.endswith(".test.negative") else 3
    with path.open("r", encoding="utf-8") as stream:
        line = next((row.strip() for row in stream if row.strip()), "")
    fields = line.split("\t")
    try:
        for value in fields[:required_columns]:
            int(value)
    except ValueError:
        return False
    return len(fields) >= required_columns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument(
        "--paper-hash",
        action="store_true",
        help="require exact SHA-256 matches to the paper's processed inputs",
    )
    parser.add_argument(
        "--profile-pollution",
        action="store_true",
        help="also require MovieLens movies.dat for attack construction",
    )
    args = parser.parse_args()
    chosen = DATASETS if args.dataset == "all" else (args.dataset,)
    errors = []
    for expected, path in manifest_entries():
        if path.parent.name not in chosen:
            continue
        if path.name == "movies.dat" and not args.profile_pollution:
            continue  # Needed by profile construction, not by main DUE training.
        if not path.is_file():
            errors.append(f"MISSING {path.relative_to(ROOT)}")
            continue
        if not first_line_is_valid(path):
            errors.append(f"FORMAT {path.relative_to(ROOT)}")
            continue
        if args.paper_hash and sha256_file(path) != expected:
            errors.append(f"HASH MISMATCH {path.relative_to(ROOT)}")
        else:
            print(f"OK {path.relative_to(ROOT)}")
    for error in errors:
        print(error)
    if errors:
        print("Input check failed; see data/README.md.")
        return 1
    print("Input check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
