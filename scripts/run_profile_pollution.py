#!/usr/bin/env python3
"""Print or execute matched Base/DUE profile-pollution evaluation jobs."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Profile-Pollution-Toolkit" / "scripts"))
from config import (  # noqa: E402
    MODELS,
    SETTINGS,
    all_stems,
    data_dir,
    gate_job,
    iter_jobs,
    score_command,
    scores_path,
    targets_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="GMF")
    parser.add_argument("--stem", choices=all_stems(), default="no_attack")
    parser.add_argument("--setting", choices=SETTINGS, default="Full")
    parser.add_argument("--gate", action="store_true", help="use the paper's gate cell")
    parser.add_argument("--all", action="store_true", help="select the full 148-job grid")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--execute", action="store_true", help="run; otherwise print only")
    args = parser.parse_args()
    if args.all and args.gate:
        parser.error("--all and --gate are mutually exclusive")
    if args.all:
        jobs = [(model, stem, setting) for model, stem, setting, _, _ in iter_jobs()]
    else:
        jobs = [gate_job() if args.gate else (args.model, args.stem, args.setting)]
    print(f"Selected {len(jobs)} profile-pollution job(s).", flush=True)
    if args.execute and not Path(targets_path()).is_file():
        raise SystemExit("Missing Profile-Pollution-Toolkit/meta/targets.json; see Profile-Pollution-Toolkit/README.md.")
    for model, stem, setting in jobs:
        if args.skip_completed and Path(scores_path(model, stem, setting)).is_file():
            print(f"SKIP {model}/{stem}/{setting}: metrics file exists", flush=True)
            continue
        cmd = shlex.split(score_command(model, stem, setting))
        cmd[0] = sys.executable
        if args.gpu_id is not None:
            cmd.extend(("--gpu_id", str(args.gpu_id)))
        workdir = ROOT / ("NCF" if model == "GMF" else "LightGCN")
        print(f"[{model}/{stem}/{setting}] (cwd {workdir.relative_to(ROOT)}) {shlex.join(cmd)}", flush=True)
        if not args.execute:
            continue
        folder = Path(data_dir(stem))
        for suffix in ("train.rating", "valid.rating", "test.negative"):
            if not (folder / f"movielens.{suffix}").is_file():
                raise SystemExit(f"Missing {folder / ('movielens.' + suffix)}; see Profile-Pollution-Toolkit/README.md.")
        log_path = ROOT / "outputs" / "profile_pollution" / model / stem / f"{setting}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(cmd, cwd=workdir, stdout=stream, stderr=subprocess.STDOUT)
        print(f"Log: {log_path.relative_to(ROOT)}", flush=True)
        if completed.returncode:
            print(f"Profile-pollution job failed (exit {completed.returncode}); inspect the log.", flush=True)
            return completed.returncode
        print(f"Metrics: {Path(scores_path(model, stem, setting)).relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
