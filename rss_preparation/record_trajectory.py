#!/usr/bin/env python3
"""Record a hand trajectory with torque OFF while a person wears the hand.

The recording becomes the desired trajectory for replay_trajectory.py. It is
saved as <data>/<session>/trajectory/leader_hand.csv using the same columns as
collect_hand_only.py, so data_collection/replay_episode.py can also read it.

Example:
  python record_trajectory.py --session grasp01 --duration 15
  python record_trajectory.py --session grasp01        # Enter stops recording
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rss_common as rc  # noqa: E402


DEFAULT_RATE_HZ = 100.0
DEFAULT_COUNTDOWN_S = 3
PROFILE_VELOCITY_RAD_S = rc.REPLAY_PROFILE_VELOCITY_RAD_S  # replay_trajectory.py default
SPEED_CHECK_SMOOTH_HZ = 5.0  # same low-pass the replay applies by default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", default=None, help="Session name or directory (default: timestamp)")
    parser.add_argument("--data-root", type=Path, default=rc.DEFAULT_DATA_ROOT, help="Where sessions live (default: %(default)s)")
    rc.add_hand_arguments(parser)
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ, help="Sample rate (default: %(default)s)")
    parser.add_argument("--duration", type=float, default=None, help="Stop after this many seconds (Enter also stops)")
    parser.add_argument("--countdown", type=int, default=DEFAULT_COUNTDOWN_S, help="Seconds of countdown before recording (default: %(default)s)")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing trajectory in this session")
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    rc.resolve_hand_args(parser, args)
    return args


def record(hand, rate_hz: float, duration_s: float | None, stop_event) -> tuple[list[dict], int]:
    """Sample positions at a fixed rate until duration, Enter, or Ctrl-C.

    Returns the CSV rows and the number of loop overruns.
    """

    motor_ids = list(hand.motor_ids)
    rows: list[dict] = []
    clock = rc.FixedRateClock(rate_hz)
    clock.start()
    first_ns = None
    next_status_s = 1.0
    try:
        while True:
            elapsed = clock.wait()
            stamp_ns = time.monotonic_ns()
            wall_s = time.time()
            positions, _velocities, _currents = hand.read_pos_vel_cur()
            if first_ns is None:
                first_ns = stamp_ns
            row = {
                "sample_idx": len(rows),
                "monotonic_ns": stamp_ns,
                "wall_time_s": f"{wall_s:.6f}",
                "elapsed_s": f"{(stamp_ns - first_ns) / 1e9:.6f}",
            }
            row.update({f"pos_{m}": f"{float(q):.6f}" for m, q in zip(motor_ids, positions)})
            rows.append(row)
            if elapsed >= next_status_s:
                print(f"\r  recording... {elapsed:5.1f} s, {len(rows)} samples", end="", flush=True)
                next_status_s += 1.0
            if stop_event.is_set() or (duration_s is not None and elapsed >= duration_s):
                break
    except KeyboardInterrupt:
        print("\n  Ctrl-C: stopping and saving.")
    print()
    return rows, clock.overruns


def main() -> int:
    args = parse_args()
    paths = rc.session_paths(args.session or rc.default_session_name(), args.data_root)
    if paths.trajectory_csv.exists() and not args.force:
        print(f"error: {paths.trajectory_csv} exists; pass --force to overwrite", file=sys.stderr)
        return 1

    config = rc.load_config(args.config, args.port, operating_mode=rc.REPLAY_OPERATING_MODE)
    hand = rc.open_hand(config, args.config)
    try:
        rc.warn_if_slow_latency(args.port)
        hand.disable_torque(verify=True)
        # Put the motors in the replay's operating mode now (torque stays off), so
        # no mode change happens between recording and replay. A swapped or reset
        # motor comes up in factory mode 3.
        hand.configure(enable_torque=False)
        motor_models = rc.model_names(hand.ping())
        print(f"Connected to {args.config.stem} on {hand.port}. Torque is OFF on all {len(hand.motor_ids)} motors.")
        print(f"Session: {paths.root}")
        rc.prompt_enter("Put the hand on. Press Enter to start recording (Enter again stops)...")
        for remaining in range(args.countdown, 0, -1):
            print(f"  {remaining}...", flush=True)
            time.sleep(1.0)
        print("  GO" + (f" ({args.duration:g} s)" if args.duration else " (press Enter to stop)"))

        watcher = rc.EnterWatcher().start()
        with rc.ReadErrorCounter() as read_errors:
            rows, overruns = record(hand, args.rate_hz, args.duration, watcher.pressed)
    finally:
        hand.shutdown()

    if len(rows) < 2:
        print("error: fewer than two samples recorded; nothing saved", file=sys.stderr)
        return 1

    motor_ids = list(config.motor_ids)
    rc.write_csv(paths.trajectory_csv, rc.trajectory_fieldnames(motor_ids), rows)
    stats = rc.rate_stats([row["monotonic_ns"] for row in rows])
    positions = np.asarray([[float(row[f"pos_{m}"]) for m in motor_ids] for row in rows])
    lower = config.joint_lower_limits_array
    upper = config.joint_upper_limits_array
    outside = {
        int(m): float(np.mean((positions[:, i] < lower[i]) | (positions[:, i] > upper[i])))
        for i, m in enumerate(motor_ids)
    }
    rc.write_json(
        paths.trajectory_metadata,
        {
            "kind": "trajectory",
            "recorded_at": rc.now_iso(),
            "config_path": str(Path(args.config).resolve()),
            "port": args.port,
            "motor_models": motor_models,
            "requested_rate_hz": args.rate_hz,
            "timing": stats,
            "loop_overruns": overruns,
            "read_errors": read_errors.count,
            "fraction_outside_joint_limits": outside,
            "position_range_rad": {
                int(m): [float(positions[:, i].min()), float(positions[:, i].max())]
                for i, m in enumerate(motor_ids)
            },
            "config": rc.config_snapshot(config),
            "git": rc.git_state(),
        },
    )

    print(f"Saved {len(rows)} samples to {paths.trajectory_csv}")
    print(
        f"  {stats['duration_s']:.2f} s at {stats['achieved_rate_hz']:.1f} Hz "
        f"(dt median {stats['dt_median_ms']:.2f} ms, p95 {stats['dt_p95_ms']:.2f} ms, "
        f"max {stats['dt_max_ms']:.2f} ms), overruns {overruns}, read errors {read_errors.count}"
    )
    times = np.asarray([float(row["elapsed_s"]) for row in rows])
    smooth_times, smoothed = rc.smooth_trajectory(times - times[0], positions, SPEED_CHECK_SMOOTH_HZ)
    speeds = np.abs(np.gradient(smoothed, smooth_times, axis=0))
    peak_speed = speeds.max(axis=0)
    print(f"  Range of motion and peak speed per joint (speed check vs {PROFILE_VELOCITY_RAD_S:g} rad/s replay profile):")
    for i, m in enumerate(motor_ids):
        span = positions[:, i].max() - positions[:, i].min()
        note = f"  ({outside[m]:.0%} outside config limits)" if outside[m] else ""
        fast = "  TOO FAST" if peak_speed[i] > PROFILE_VELOCITY_RAD_S else ""
        print(
            f"    ID {m:>2} {rc.motor_label(m):<17} {positions[:, i].min():+7.3f} .. {positions[:, i].max():+7.3f}"
            f"  (span {span:.3f})  peak {peak_speed[i]:5.1f} rad/s{fast}{note}"
        )
    over = float(np.mean((speeds > PROFILE_VELOCITY_RAD_S).any(axis=1)))
    if over > 0:
        print(
            f"  note: some joint exceeds {PROFILE_VELOCITY_RAD_S:g} rad/s for {over:.0%} of the recording. "
            "The replay's motion profile caps motors at that speed, "
            "so those parts will lag. Re-record more slowly, or replay with --profile-velocity 0."
        )
    if any(outside.values()):
        print(
            "  note: some samples are outside the YAML joint limits. Replay does not clip by "
            "default (the recording is physically reachable), so this only matters with --clip-to-limits."
        )
    if read_errors.count:
        print(f"  WARNING: {read_errors.count} sync-read errors; those samples repeat the previous value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
