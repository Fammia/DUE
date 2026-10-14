#!/usr/bin/env python3
"""Exit 0 iff the gate score job has some non-zero T-HR@50 among the 10 movies."""

from __future__ import annotations

import json
import os
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from config import gate_job, scores_path  # noqa: E402


def main() -> int:
    model, stem, setting = gate_job()
    path = scores_path(model, stem, setting)
    if not os.path.isfile(path):
        print(f"GATE FAIL: missing {path}")
        return 2
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("targets", {})
    hrs = [float(v.get("hr@50", 0.0)) for v in rows.values()]
    print(json.dumps({k: rows[k].get("hr@50") for k in rows}, indent=2))
    if not hrs:
        print("GATE FAIL: no target rows")
        return 1
    if all(h == 0.0 for h in hrs):
        print("GATE FAIL: all 10 movies have HR@50 = 0. Do not launch the 148-job grid.")
        return 1
    print("GATE PASS: at least one movie has HR@50 > 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
