#!/usr/bin/env python3
"""Save a task-specific arm + hand start from a recorded raw episode (no hardware).

Example: .venv/bin/python scripts/kinesthetic/capture_task_start.py --task screw \
    --from-episode data_trimorph_kinesthetic/raw/episode_000001 \
    --output config/task_starts/screw.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_collection.collection.task_start import capture_task_start, task_start_sha256, write_task_start


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="task name stored with this start")
    parser.add_argument("--from-episode", type=Path, required=True, help="raw/episode_XXXXXX directory")
    parser.add_argument("--at-s", type=float, default=0.0, help="seconds from episode start (default: 0)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="explicitly replace an existing profile")
    args = parser.parse_args(argv)
    try:
        profile = capture_task_start(args.task, args.from_episode, at_s=args.at_s)
        path = write_task_start(args.output, profile, overwrite=args.overwrite)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved task start {profile['task']!r}: {path}")
    print("Arm J1..J6 deg: " + " ".join(f"{q:+.3f}" for q in profile["arm"]["joints_deg"]))
    print(f"Measured hand motor IDs 0..12; arm/hand skew {profile['source']['sync_error_ms']:.3f} ms")
    print(f"SHA-256: {task_start_sha256(path)}")
    for warning in profile["warnings"]:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
