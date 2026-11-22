#!/usr/bin/env python3
"""Replay a recorded hand trajectory and log current, command, and position.

Run once per condition. Each run writes <session>/replays/<condition>.csv in
replay_episode.py's telemetry schema (readable by gen_graph.sh and
gen_graphs_from_csvs.py), plus a JSON file with settings and diagnostics.

Sequence: connect (torque off) -> prompt -> configure mode/current cap ->
remove whole-turn offsets -> torque on holding the current pose -> cosine ramp
to the first sample -> settle -> replay (command + log every tick) -> hold ->
torque off. Ctrl-C, an exception, or the current abort all disable torque.

Example:
  python replay_trajectory.py grasp01 --condition idle
  python replay_trajectory.py grasp01 --condition tpu
  python replay_trajectory.py grasp01 --condition human --current-limit-ma 800
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rss_common as rc  # noqa: E402


DEFAULT_RATE_HZ = 100.0
DEFAULT_OPERATING_MODE = 5
DEFAULT_ABORT_FRACTION = 0.95
DEFAULT_ABORT_SAMPLES = 10
DEFAULT_PREPOSITION_S = 3.0
DEFAULT_SETTLE_S = 1.0
DEFAULT_HOLD_S = 1.0
DEFAULT_MAX_START_DELTA_RAD = 1.5
DEFAULT_PROFILE_VELOCITY_RAD_S = rc.REPLAY_PROFILE_VELOCITY_RAD_S
DEFAULT_PROFILE_ACCELERATION_RAD_S2 = rc.REPLAY_PROFILE_ACCELERATION_RAD_S2
# I = 0 avoids integral windup while the motor is current-saturated.
DEFAULT_I_GAIN = 0
DEFAULT_SMOOTH_HZ = 5.0
TWO_PI = 2.0 * math.pi

CONDITION_INSTRUCTIONS = {
    "idle": "Make sure the hand is EMPTY.",
    "tpu": "Insert the TPU pseudo-hand.",
    "human": "Put your hand in and RELAX it. Ctrl-C stops and releases the hand.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", help="Session name (under --data-root) or directory")
    parser.add_argument("--condition", required=True, help="idle, tpu, human (or another label)")
    parser.add_argument("--data-root", type=Path, default=rc.DEFAULT_DATA_ROOT, help="Where sessions live (default: %(default)s)")
    parser.add_argument("--config", type=Path, default=rc.DEFAULT_CONFIG, help="Hand YAML (default: %(default)s)")
    parser.add_argument("--port", default=rc.DEFAULT_PORT, help="Serial port (default: %(default)s)")
    parser.add_argument("--operating-mode", type=int, choices=(3, 5), default=DEFAULT_OPERATING_MODE, help="5 = current-based position (cap enforced), 3 = pure position (default: %(default)s)")
    parser.add_argument("--current-limit-ma", type=int, default=None, help="Goal Current cap in mode 5 (default: the YAML's goal_current_limit_ma)")
    parser.add_argument("--abort-current-ma", type=float, default=None, help=f"Torque off if any |current| stays at/above this (default: {DEFAULT_ABORT_FRACTION} x cap)")
    parser.add_argument("--abort-samples", type=int, default=DEFAULT_ABORT_SAMPLES, help="Consecutive samples above the abort threshold (default: %(default)s)")
    parser.add_argument("--profile-velocity", type=float, default=DEFAULT_PROFILE_VELOCITY_RAD_S, help="Dynamixel profile velocity limit in rad/s (policy rollout uses 3); 0 disables (default: %(default)s)")
    parser.add_argument("--profile-acceleration", type=float, default=DEFAULT_PROFILE_ACCELERATION_RAD_S2, help="Dynamixel profile acceleration in rad/s^2; 0 disables (default: %(default)s)")
    parser.add_argument("--p-gain", type=int, default=None, help="Position P gain (default: the YAML's, 2000 for Trimi1)")
    parser.add_argument("--i-gain", type=int, default=DEFAULT_I_GAIN, help="Position I gain (default: %(default)s; the YAML/rollout use 100)")
    parser.add_argument("--d-gain", type=int, default=None, help="Position D gain (default: the YAML's, 400 for Trimi1)")
    parser.add_argument("--smooth-hz", type=float, default=DEFAULT_SMOOTH_HZ, help="Zero-phase low-pass cutoff applied to the recording before replay; 0 disables (default: %(default)s)")
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ, help="Command + logging rate (default: %(default)s)")
    parser.add_argument("--playback-speed", type=float, default=1.0, help="Time scale: 1 = recorded speed, 0.5 = half speed (default: %(default)s)")
    parser.add_argument("--preposition-s", type=float, default=DEFAULT_PREPOSITION_S, help="Ramp time to the first sample (default: %(default)s)")
    parser.add_argument("--settle-s", type=float, default=DEFAULT_SETTLE_S, help="Hold at the first sample before replay (default: %(default)s)")
    parser.add_argument("--hold-s", type=float, default=DEFAULT_HOLD_S, help="Hold at the last sample before torque off (default: %(default)s)")
    parser.add_argument("--max-start-delta-rad", type=float, default=DEFAULT_MAX_START_DELTA_RAD, help="Refuse to start if any joint is farther than this from the first sample (default: %(default)s)")
    parser.add_argument("--clip-to-limits", action="store_true", help="Clip commands to the YAML joint limits (off by default; see README)")
    parser.add_argument("--yes", action="store_true", help="Do not wait for Enter before enabling torque")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing replay for this condition")
    args = parser.parse_args()
    for name in ("rate_hz", "playback_speed", "max_start_delta_rad"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("preposition_s", "settle_s", "hold_s"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.abort_samples < 1:
        parser.error("--abort-samples must be at least 1")
    for name in ("profile_velocity", "profile_acceleration", "smooth_hz"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


# --------------------------------------------------------------------------
# Pure helpers (unit-tested without hardware)
# --------------------------------------------------------------------------


def interpolate(times: np.ndarray, positions: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation of every joint at time t, clamped to the ends."""

    t = min(max(float(t), float(times[0])), float(times[-1]))
    index = int(np.searchsorted(times, t, side="right"))
    if index <= 0:
        return positions[0].copy()
    if index >= len(times):
        return positions[-1].copy()
    t0, t1 = times[index - 1], times[index]
    blend = (t - t0) / (t1 - t0)
    return positions[index - 1] + blend * (positions[index] - positions[index - 1])


def unwrap_turns(trajectory_start: np.ndarray, measured: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Whole motor turns between the measured pose and the trajectory start.

    In mode 5 positions are multi-turn, but a power cycle resets each motor to
    a single turn. A motor near the 0/4096 boundary then reads exactly one turn
    away from its recorded value. A full output-shaft turn is the same physical
    pose, so shifting the trajectory by whole turns is exact and avoids
    commanding a 360 degree spin. Returns (turns, residual_rad).
    """

    delta = np.asarray(measured, dtype=float) - np.asarray(trajectory_start, dtype=float)
    turns = np.round(delta / TWO_PI).astype(int)
    return turns, delta - turns * TWO_PI


class CurrentAbortMonitor:
    """Trip when any motor's |current| stays >= threshold for N samples."""

    def __init__(self, motor_ids, threshold_ma: float, samples: int) -> None:
        self.motor_ids = list(motor_ids)
        self.threshold_ma = float(threshold_ma)
        self.samples = int(samples)
        self._counts = np.zeros(len(self.motor_ids), dtype=int)

    def update(self, currents_ma) -> dict | None:
        over = np.abs(np.asarray(currents_ma, dtype=float)) >= self.threshold_ma
        self._counts = np.where(over, self._counts + 1, 0)
        tripped = np.flatnonzero(self._counts >= self.samples)
        if tripped.size == 0:
            return None
        index = int(tripped[0])
        return {
            "motor_id": int(self.motor_ids[index]),
            "joint": rc.motor_label(self.motor_ids[index]),
            "current_ma": float(currents_ma[index]),
            "threshold_ma": self.threshold_ma,
            "consecutive_samples": self.samples,
        }


@dataclass
class ReplayResult:
    rows: list[dict] = field(default_factory=list)
    aborted: dict | None = None
    interrupted: bool = False
    overruns: int = 0


def _telemetry_row(trial, elapsed_s, stamp_ns, motor_ids, current, target, actual, target_velocity, actual_velocity) -> dict:
    row = {"trial": trial, "elapsed_s": f"{elapsed_s:.6f}", "monotonic_ns": stamp_ns}
    for i, motor_id in enumerate(motor_ids):
        row[f"motor_{motor_id}_current_ma"] = f"{float(current[i]):.1f}"
        row[f"motor_{motor_id}_target_position_rad"] = f"{float(target[i]):.6f}"
        row[f"motor_{motor_id}_actual_position_rad"] = f"{float(actual[i]):.6f}"
        row[f"motor_{motor_id}_target_velocity_rad_s"] = f"{float(target_velocity[i]):.6f}"
        row[f"motor_{motor_id}_actual_velocity_rad_s"] = f"{float(actual_velocity[i]):.6f}"
    return row


def ramp_to(hand, start, goal, duration_s: float, rate_hz: float, monitor: CurrentAbortMonitor) -> dict | None:
    """Cosine ramp from start to goal, then return; checks the current abort."""

    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    clock = rc.FixedRateClock(rate_hz)
    clock.start()
    while True:
        elapsed = clock.wait()
        alpha = 1.0 if duration_s <= 0 else min(elapsed / duration_s, 1.0)
        blend = 0.5 - 0.5 * math.cos(math.pi * alpha)
        hand.set_positions(start + blend * (goal - start), clip=False)
        _q, _v, current = hand.read_pos_vel_cur()
        tripped = monitor.update(current)
        if tripped is not None:
            return tripped
        if alpha >= 1.0:
            return None


def hold(hand, pose, duration_s: float, rate_hz: float, monitor: CurrentAbortMonitor) -> dict | None:
    if duration_s <= 0:
        return None
    clock = rc.FixedRateClock(rate_hz)
    clock.start()
    while True:
        elapsed = clock.wait()
        hand.set_positions(pose, clip=False)
        _q, _v, current = hand.read_pos_vel_cur()
        tripped = monitor.update(current)
        if tripped is not None:
            return tripped
        if elapsed >= duration_s:
            return None


def run_replay(
    hand,
    times: np.ndarray,
    positions: np.ndarray,
    *,
    rate_hz: float,
    playback_speed: float,
    monitor: CurrentAbortMonitor,
    clip: bool = False,
    trial: int = 1,
) -> ReplayResult:
    """Stream the trajectory at rate_hz, logging one telemetry row per tick."""

    motor_ids = list(hand.motor_ids)
    duration_s = float(times[-1] - times[0])
    result = ReplayResult()
    clock = rc.FixedRateClock(rate_hz)
    previous_target = None
    previous_t = None
    clock.start()
    try:
        while True:
            wall_elapsed = clock.wait()
            t = min(wall_elapsed * playback_speed, duration_s)
            target = interpolate(times, positions, times[0] + t)
            if clip:
                target = hand.clip_positions(target)
            hand.set_positions(target, clip=False)
            stamp_ns = time.monotonic_ns()
            actual, actual_velocity, current = hand.read_pos_vel_cur()
            if previous_target is None or t <= previous_t:
                target_velocity = np.zeros_like(target)
            else:
                target_velocity = (target - previous_target) / (t - previous_t)
            result.rows.append(
                _telemetry_row(trial, t, stamp_ns, motor_ids, current, target, actual, target_velocity, actual_velocity)
            )
            previous_target, previous_t = target, t
            tripped = monitor.update(current)
            if tripped is not None:
                hand.disable_torque(verify=True)
                tripped["elapsed_s"] = t
                result.aborted = tripped
                break
            if t >= duration_s:
                break
    except KeyboardInterrupt:
        hand.disable_torque(verify=False)
        result.interrupted = True
    result.overruns = clock.overruns
    return result


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    paths = rc.session_paths(args.session, args.data_root)
    if not paths.trajectory_csv.is_file():
        print(f"error: no trajectory at {paths.trajectory_csv}; run record_trajectory.py first", file=sys.stderr)
        return 1
    output_csv = paths.replay_csv(args.condition)
    output_json = paths.replay_metadata(args.condition)
    if output_csv.exists() and not args.force:
        print(f"error: {output_csv} exists; pass --force to overwrite", file=sys.stderr)
        return 1

    trajectory_ids, raw_times, raw_positions = rc.load_trajectory(paths.trajectory_csv)
    times, positions = rc.smooth_trajectory(raw_times, raw_positions, args.smooth_hz)
    config = rc.load_config(
        args.config,
        args.port,
        operating_mode=args.operating_mode,
        current_limit_ma=args.current_limit_ma,
        p_gain=args.p_gain,
        i_gain=args.i_gain,
        d_gain=args.d_gain,
    )
    if tuple(trajectory_ids) != tuple(config.motor_ids):
        print(f"error: trajectory motors {trajectory_ids} != config motors {tuple(config.motor_ids)}", file=sys.stderr)
        return 1
    cap_ma = int(config.goal_current_limit)
    abort_ma = args.abort_current_ma if args.abort_current_ma is not None else DEFAULT_ABORT_FRACTION * cap_ma
    duration_s = float(times[-1])

    print(f"Session {paths.name}: trajectory {duration_s:.2f} s, {len(times)} samples")
    print(
        f"Condition: {args.condition} | mode {args.operating_mode} | current cap {cap_ma} mA"
        f"{'' if args.operating_mode == 5 else ' (NOT enforced in mode 3)'} | abort at {abort_ma:.0f} mA "
        f"x{args.abort_samples} | {args.rate_hz:g} Hz | speed {args.playback_speed:g}x"
        f"{' | clipped to joint limits' if args.clip_to_limits else ''}"
    )
    print(
        f"Gains P/I/D {config.position_p_gain}/{config.position_i_gain}/{config.position_d_gain} | "
        f"profile {args.profile_velocity:g} rad/s, {args.profile_acceleration:g} rad/s^2"
        f"{' (disabled)' if args.profile_velocity == 0 else ''} | "
        f"smoothing {f'{args.smooth_hz:g} Hz low-pass' if args.smooth_hz > 0 else 'off'}"
    )
    if args.smooth_hz > 0:
        change = np.abs(np.column_stack(
            [np.interp(times, raw_times, raw_positions[:, j]) for j in range(raw_positions.shape[1])]
        ) - positions).max(axis=0)
        print(f"  smoothing changed the trajectory by at most {change.max():.3f} rad (joint ID {trajectory_ids[int(change.argmax())]})")

    monitor = CurrentAbortMonitor(config.motor_ids, abort_ma, args.abort_samples)
    result = ReplayResult()
    turns = np.zeros(len(trajectory_ids), dtype=int)
    start_residual = np.zeros(len(trajectory_ids))
    started_at = rc.now_iso()
    with rc.ReadErrorCounter() as read_errors:
        hand = rc.open_hand(config)
        try:
            rc.warn_if_slow_latency(args.port)
            hand.disable_torque(verify=True)
            instruction = CONDITION_INSTRUCTIONS.get(args.condition, "")
            if not args.yes:
                rc.prompt_enter(f"[{args.condition}] {instruction} Press Enter to enable torque and move to the start pose...")
            hand.configure(enable_torque=False)
            hand.set_motion_profile(
                profile_velocity_rad_s=args.profile_velocity or None,
                profile_acceleration_rad_s2=args.profile_acceleration or None,
            )
            measured = hand.read_pos()

            turns, start_residual = unwrap_turns(positions[0], measured)
            if np.any(turns):
                shifted = {int(m): int(k) for m, k in zip(trajectory_ids, turns) if k}
                print(f"Note: removing whole-turn offsets after a power cycle (motor: turns) {shifted}")
                positions = positions + turns * TWO_PI
            far = {
                int(m): round(float(r), 3)
                for m, r in zip(trajectory_ids, start_residual)
                if abs(r) > args.max_start_delta_rad
            }
            if far:
                raise RuntimeError(
                    f"joints too far from the trajectory start (rad): {far}; limit "
                    f"{args.max_start_delta_rad} rad. Move the hand closer to the start pose or "
                    "raise --max-start-delta-rad."
                )

            hand.set_positions(measured, clip=False)
            hand.enable_torque()
            print(f"Torque ON. Moving to the start pose over {args.preposition_s:g} s...")
            start_pose = hand.clip_positions(positions[0]) if args.clip_to_limits else positions[0]
            tripped = ramp_to(hand, measured, start_pose, args.preposition_s, args.rate_hz, monitor)
            tripped = tripped or hold(hand, start_pose, args.settle_s, args.rate_hz, monitor)
            if tripped is not None:
                hand.disable_torque(verify=True)
                tripped["phase"] = "preposition"
                result.aborted = tripped
            else:
                print(f"Replaying {duration_s / args.playback_speed:.1f} s...")
                result = run_replay(
                    hand,
                    times,
                    positions,
                    rate_hz=args.rate_hz,
                    playback_speed=args.playback_speed,
                    monitor=monitor,
                    clip=args.clip_to_limits,
                )
                if result.aborted is None and not result.interrupted:
                    final_pose = hand.clip_positions(positions[-1]) if args.clip_to_limits else positions[-1]
                    tripped = hold(hand, final_pose, args.hold_s, args.rate_hz, monitor)
                    if tripped is not None:
                        tripped["phase"] = "final hold"
                        result.aborted = tripped
        except KeyboardInterrupt:
            result.interrupted = True
            print("\nInterrupted.")
        finally:
            hand.shutdown()
            print("Torque OFF.")

    if result.aborted:
        print(f"ABORTED: {result.aborted}")
    if not result.rows:
        print("No replay samples recorded; nothing saved.")
        return 1

    motor_ids = list(config.motor_ids)
    rc.write_csv(output_csv, rc.telemetry_fieldnames(motor_ids), result.rows)
    stats = rc.rate_stats([row["monotonic_ns"] for row in result.rows])
    complete = result.aborted is None and not result.interrupted
    rc.write_json(
        output_json,
        {
            "kind": "replay",
            "condition": args.condition,
            "condition_label": rc.CONDITION_LABELS.get(args.condition, args.condition),
            "started_at": started_at,
            "complete": complete,
            "aborted": result.aborted,
            "interrupted": result.interrupted,
            "trajectory_csv": str(paths.trajectory_csv),
            "trajectory_sha256": rc.sha256_file(paths.trajectory_csv),
            "trajectory_duration_s": duration_s,
            "whole_turn_offsets": {int(m): int(k) for m, k in zip(trajectory_ids, turns)},
            "start_residual_rad": {int(m): float(r) for m, r in zip(trajectory_ids, start_residual)},
            "settings": {
                "operating_mode": args.operating_mode,
                "current_limit_ma": cap_ma,
                "abort_current_ma": abort_ma,
                "abort_samples": args.abort_samples,
                "rate_hz": args.rate_hz,
                "playback_speed": args.playback_speed,
                "preposition_s": args.preposition_s,
                "settle_s": args.settle_s,
                "hold_s": args.hold_s,
                "clip_to_limits": args.clip_to_limits,
                "position_pid_gains": [
                    int(config.position_p_gain),
                    int(config.position_i_gain),
                    int(config.position_d_gain),
                ],
                "profile_velocity_rad_s": args.profile_velocity,
                "profile_acceleration_rad_s2": args.profile_acceleration,
                "smooth_hz": args.smooth_hz,
            },
            "timing": stats,
            "loop_overruns": result.overruns,
            "read_errors": read_errors.count,
            "config_path": str(Path(args.config).resolve()),
            "config": rc.config_snapshot(config),
            "git": rc.git_state(),
        },
    )
    print(f"Saved {len(result.rows)} samples to {output_csv}")
    print(
        f"  {stats.get('achieved_rate_hz', float('nan')):.1f} Hz achieved, overruns {result.overruns}, "
        f"read errors {read_errors.count}{'' if complete else '  (PARTIAL RUN)'}"
    )
    metrics = rc.tracking_metrics(rc.load_telemetry(output_csv))
    print(f"  {'ID':>2}  {'joint':<17} {'RMS err (deg)':>13} {'max err (deg)':>13} {'mean |I| mA':>11} {'peak |I| mA':>11}")
    for motor_id, m in metrics.items():
        print(
            f"  {motor_id:>2}  {rc.motor_label(motor_id):<17} {math.degrees(m['rms_error_rad']):>13.2f} "
            f"{math.degrees(m['max_abs_error_rad']):>13.2f} {m['mean_abs_current_ma']:>11.0f} {m['peak_abs_current_ma']:>11.0f}"
        )
    return 0 if complete else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
