#!/usr/bin/env python3
"""Bench check of the receive-only gravity-comp collector (KINESTHETIC_PLAN.md 1.3).

With drag mode running (run_gravity_comp.sh --from drag, or gravity_comp.drag_mode
--force --allow-underpowered) and the arm resting in a pose it holds, record a
short episode with the collector in its receive-only mode:

    python3 data_collection/collect_hand_only.py --arm-mode gravity-comp --no-camera \\
        --episode-s 30 --dataset-root /tmp/kinesthetic_bench

then report on it:

    python3 scripts/grav_comp/listener_bench_report.py /tmp/kinesthetic_bench/raw/episode_000000

This reports the joint feedback rate and gaps, the drive and error state, and
whether the firmware end pose streamed while drag mode held the arm in MIT mode.
It cannot see drag mode itself: watch its display for freezes or overruns during
the recording, and note them with the report.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

JOINT_RATE_RANGE_HZ = (150.0, 250.0)  # the arm streams joint frames at about 200 Hz
MAX_GAP_S = 0.05
GAP_S = 0.02  # intervals longer than this are counted as gaps
EEF_STREAMING_FRACTION = 0.95


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or not path.read_text().strip():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(episode_dir: Path) -> dict:
    episode_dir = Path(episode_dir)
    rows = _read_csv(episode_dir / "piper.csv")
    metadata_path = episode_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    stamps = np.array([float(row["robot_timestamp_s"]) for row in rows])
    intervals = np.diff(stamps)
    span_s = float(stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0

    def fraction(predicate) -> float:
        return sum(1 for row in rows if predicate(row)) / len(rows) if rows else 0.0

    hand_rows = _read_csv(episode_dir / "leader_hand.csv")
    duration_s = float(metadata.get("duration_s") or 0.0)
    return {
        "piper_mode": (metadata.get("piper") or {}).get("mode"),
        "rows": len(rows),
        "span_s": span_s,
        "joint_rate_hz": (len(stamps) - 1) / span_s if span_s > 0 else 0.0,
        "duplicate_frames": int(len(stamps) - len(np.unique(stamps))),
        "median_interval_ms": float(np.median(intervals) * 1e3) if len(intervals) else None,
        "p99_interval_ms": float(np.percentile(intervals, 99) * 1e3) if len(intervals) else None,
        "max_interval_ms": float(intervals.max() * 1e3) if len(intervals) else None,
        "gaps_over_20ms": int((intervals > GAP_S).sum()),
        "motor_feedback_fraction": fraction(lambda row: row.get("vel_1", "") != ""),
        "all_drives_enabled_fraction": fraction(
            lambda row: all(row.get(f"enable_{joint}") == "1" for joint in range(1, 7))
        ),
        "arm_ok_fraction": fraction(lambda row: row.get("arm_ok") == "1"),
        "err_codes": sorted({row.get("err_code", "") for row in rows} - {""}),
        "ctrl_modes": sorted({row.get("ctrl_mode", "") for row in rows} - {""}),
        "mode_feedback": sorted({row.get("mode_feedback", "") for row in rows} - {""}),
        "arm_statuses": sorted({row.get("arm_status", "") for row in rows} - {""}),
        "eef_fraction": fraction(lambda row: row.get("x", "") != ""),
        "hand_rate_hz": len(hand_rows) / duration_s if duration_s > 0 else None,
        "flags": metadata.get("flags", []),
    }


def checks(summary: dict) -> list[tuple[bool, str]]:
    low, high = JOINT_RATE_RANGE_HZ
    max_gap = summary["max_interval_ms"]
    eef = summary["eef_fraction"]
    if eef >= EEF_STREAMING_FRACTION:
        eef_text = "firmware end pose streams in MIT mode"
    elif eef == 0:
        eef_text = "firmware end pose does NOT stream in MIT mode (FK poses do not need it)"
    else:
        eef_text = f"firmware end pose streams only part of the time ({eef:.0%} of rows)"
    return [
        (summary["piper_mode"] == "gravity_comp_kinesthetic",
         f"recorded in receive-only gravity-comp mode (piper.mode={summary['piper_mode']})"),
        (low <= summary["joint_rate_hz"] <= high,
         f"joint frames at {summary['joint_rate_hz']:.1f} Hz (expected about 200)"),
        (summary["duplicate_frames"] == 0, f"{summary['duplicate_frames']} duplicate frames"),
        (max_gap is not None and max_gap <= MAX_GAP_S * 1e3,
         f"longest gap {max_gap if max_gap is not None else float('nan'):.1f} ms "
         f"({summary['gaps_over_20ms']} over 20 ms)"),
        (summary["arm_ok_fraction"] == 1.0,
         f"drives enabled with no error code in {summary['arm_ok_fraction']:.1%} of rows"),
        # Informational: either answer is a result for 1.3.
        (True, eef_text),
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episode_dir", type=Path, help="raw/episode_NNNNNN folder of the bench recording")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON instead")
    args = parser.parse_args(argv)
    summary = summarize(args.episode_dir)
    if not summary["rows"]:
        print(f"No Piper rows in {args.episode_dir / 'piper.csv'}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for key, value in summary.items():
            print(f"{key:28s} {value}")
        print()
    results = checks(summary)
    for ok, text in results:
        print(f"{'OK   ' if ok else 'CHECK'} {text}")
    print("\nAlso confirm by eye: drag mode showed no freeze or overrun during the recording.")
    return 0 if all(ok for ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
