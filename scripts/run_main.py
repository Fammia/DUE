#!/usr/bin/env python3
"""Print or execute the twelve configured DUE training commands."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ("GMF", "NeuMF", "CDAE", "LightGCN")
DATASETS = ("movielens", "yelp", "amazon_book")

# Default (floor_ratio, factor_upper) pairs, selected by validation Recall@50.
PARAMETERS = {
    ("GMF", "movielens"): (0.3, 0.25),
    ("GMF", "yelp"): (0.4, 0.1),
    ("GMF", "amazon_book"): (0.3, 0.06),
    ("NeuMF", "movielens"): (0.1, 2.25),
    ("NeuMF", "yelp"): (0.115, 0.088),
    ("NeuMF", "amazon_book"): (0.05, 0.065),
    ("CDAE", "movielens"): (0.1, 2.5),
    ("CDAE", "yelp"): (0.5, 0.6),
    ("CDAE", "amazon_book"): (0.5, 0.4),
    ("LightGCN", "movielens"): (0.1, 0.95),
    ("LightGCN", "yelp"): (0.4, 1.5),
    ("LightGCN", "amazon_book"): (0.1, 2.5),
}


def command(backbone: str, dataset: str, seed: int, gpu_id: int | None):
    ratio, upper = PARAMETERS[(backbone, dataset)]
    workdir = ROOT / ("NCF" if backbone in ("GMF", "NeuMF") else backbone)
    args = [
        sys.executable,
        "main_due.py",
        "--dataset",
        dataset,
        "--floor_ratio",
        str(ratio),
        "--factor_upper",
        str(upper),
        "--seed",
        str(seed),
    ]
    if backbone in ("GMF", "NeuMF"):
        args.extend(("--model", "GMF" if backbone == "GMF" else "NeuMF-end"))
    if gpu_id is not None:
        args.extend(("--gpu_id", str(gpu_id)))
    return workdir, args


def check_inputs(dataset: str) -> None:
    folder = ROOT / "data" / dataset
    missing = [
        name
        for name in (
            f"{dataset}.train.rating",
            f"{dataset}.valid.rating",
            f"{dataset}.test.negative",
        )
        if not (folder / name).is_file()
    ]
    if missing:
        raise SystemExit(
            f"Missing {dataset} inputs: {', '.join(missing)}. See data/README.md."
        )


def show_final_metrics(log_path: Path) -> None:
    log = log_path.read_text(encoding="utf-8", errors="replace")
    recall = re.findall(r"^Recall\s+([0-9.]+)-([0-9.]+)", log, re.MULTILINE)
    ndcg = re.findall(r"^NDCG\s+([0-9.]+)-([0-9.]+)", log, re.MULTILINE)
    if recall and ndcg:
        print(f"Test Recall@50/@100: {recall[-1]}; NDCG@50/@100: {ndcg[-1]}")
    else:
        print(f"No final test metrics found in {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=(*BACKBONES, "all"), default="all")
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--seed", type=int, default=123456789)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="run; otherwise print only")
    args = parser.parse_args()
    backbones = BACKBONES if args.backbone == "all" else (args.backbone,)
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    for backbone in backbones:
        for dataset in datasets:
            workdir, cmd = command(backbone, dataset, args.seed, args.gpu_id)
            print(f"[{backbone}/{dataset}] (cwd {workdir.relative_to(ROOT)}) {shlex.join(cmd)}", flush=True)
            if not args.execute:
                continue
            check_inputs(dataset)
            log_path = ROOT / "outputs" / "main" / dataset / f"{backbone}_seed{args.seed}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(cmd, cwd=workdir, stdout=stream, stderr=subprocess.STDOUT)
            print(f"Log: {log_path.relative_to(ROOT)}", flush=True)
            if completed.returncode:
                print(f"Training failed (exit {completed.returncode}); inspect the log.", flush=True)
                return completed.returncode
            show_final_metrics(log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
