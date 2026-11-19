#!/usr/bin/env python3
"""One-to-one joint teleoperation between two identical TriMI hands.

    python3 scripts/teleop/trimi_teleop.py \\
        --leader-port /dev/serial/by-id/<leader adapter> \\
        --follower-port /dev/serial/by-id/<follower adapter> [--force-feedback] [--log run.npz]

The leader is moved by hand. Every cycle its calibrated joint angles are sent,
unchanged, as the follower's goal positions. That is one to one only when both
hands were homed with the same homing table, so the same pose reads the same
angles on both. Home each hand into its own file, the second from the first so
they share the table:

    trimi-live home --hand-port <leader adapter> --out ~/.config/trimi-copilot/hand_leader.yaml
    trimi-live home --hand-config ~/.config/trimi-copilot/hand_leader.yaml \\
        --hand-port <follower adapter> --out ~/.config/trimi-copilot/hand_follower.yaml

The script refuses calibrations whose motor IDs, joint signs or homing tables
differ, a follower file that still has the leader's offsets (not homed), and
any joint reading far outside its limits (a power cycle can shift a joint by a
turn until it is homed again).

On start the follower eases from where it is to the leader's pose over
--ramp-s, then follows it. Its goals are clipped to its joint limits, its
current is capped at --current-limit-ma and its speed at --max-speed-rad-s.
A leader reading that jumps more than --max-jump-rad in one cycle stops the
teleoperation. On exit (Ctrl-C or any error) both hands' torque is switched
off.

Force feedback (--force-feedback) uses the contact detector and leader current
mapping of data_collection's leader/follower pipeline. Each follower joint
watches its position error: once it has reached its goal after the ramp, an
error beyond 0.05 rad while still (or a current above 250 mA while moving,
which needs --current-limit-ma above that) is contact, and the error beyond
0.03 rad is fed back as a current on the same leader joint, opposing the push:
--feedback-max-current-ma at --feedback-full-scale-rad beyond the deadband.
Without force feedback the leader is unpowered; with it the leader's joints
are in current mode at 0 mA except while their follower joint is in contact.

--log saves every cycle (leader angles, follower goals, angles, velocities and
currents, the feedback) for scripts/teleop/analyze_teleop_log.py.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

# Support both direct execution and package imports.
REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from data_collection.collection.contact_detection import (
    CONTACT_EXIT_THRESHOLD,
    CONTACT_THRESHOLD,
    ContactStateMachine,
    contact_feedback,
)

CALIBRATION_DIR = Path.home() / ".config/trimi-copilot"
DEFAULT_LEADER_CONFIG = CALIBRATION_DIR / "hand_leader.yaml"
DEFAULT_FOLLOWER_CONFIG = CALIBRATION_DIR / "hand_follower.yaml"
HAND_API_CHECKOUT = REPO_ROOT.parent / "trimorph_hand_api"  # the monorepo's copy
SERIAL_BY_ID = Path("/dev/serial/by-id")
DEFAULT_RATE_HZ = 100.0
DEFAULT_RAMP_S = 2.0
DEFAULT_CURRENT_LIMIT_MA = 400.0
DEFAULT_MAX_SPEED_RAD_S = 6.0
DEFAULT_MAX_ACCELERATION_RAD_S2 = 80.0
DEFAULT_MAX_JUMP_RAD = 1.0
# As data_collection.collection.config's FORCE_FEEDBACK_* defaults.
DEFAULT_FEEDBACK_MAX_CURRENT_MA = 100.0
DEFAULT_FEEDBACK_FULL_SCALE_RAD = 0.01
LIMIT_MARGIN_RAD = math.radians(20.0)
LOG_FIELDS = ("t", "leader", "goal", "position", "velocity", "current", "feedback")


def _by_motor(value, motor_ids: Sequence[int]) -> dict[int, float]:
    """A per-motor YAML field (mapping or list in motor order) as {motor: value}."""

    if isinstance(value, Mapping):
        return {int(k): float(v) for k, v in value.items()}
    return {int(m): float(v) for m, v in zip(motor_ids, value)}


def check_pair(leader: Mapping, follower: Mapping) -> None:
    """Refuse two hand calibrations that would not read the same angle in the same pose."""

    leader_ids = [int(m) for m in leader["motor_ids"]]
    follower_ids = [int(m) for m in follower["motor_ids"]]
    if leader_ids != follower_ids:
        raise ValueError(f"Motor IDs differ: leader {leader_ids}, follower {follower_ids}")
    problems = []
    if _by_motor(leader["joint_signs"], leader_ids) != _by_motor(follower["joint_signs"], follower_ids):
        problems.append("joint signs")
    if leader.get("homing") != follower.get("homing"):
        problems.append("homing tables")
    if problems:
        raise ValueError(
            f"The two calibrations have different {' and '.join(problems)}, so the hands do not read "
            "the same angle in the same pose. Home the follower from the leader's file "
            "(trimi-live home --hand-config <leader file> --out <follower file>)."
        )
    if _by_motor(leader["home_offsets"], leader_ids) == _by_motor(follower["home_offsets"], follower_ids):
        raise ValueError(
            "Both calibrations have the same home offsets: the follower file is a copy of the leader's "
            "that was never homed on the follower hand."
        )


def out_of_limits(positions, lower, upper, margin: float = LIMIT_MARGIN_RAD) -> list[int]:
    """Indices of joints reading farther than `margin` outside their limits."""

    positions = np.asarray(positions, dtype=np.float64)
    bad = (positions < np.asarray(lower) - margin) | (positions > np.asarray(upper) + margin)
    return [int(i) for i in np.where(bad | ~np.isfinite(positions))[0]]


class ContactFeedback:
    """Per-joint follower contact from its position error, as in the collection pipeline."""

    def __init__(self, motor_ids: Sequence[int], feedback_motor_ids: Sequence[int]):
        motor_ids = [int(m) for m in motor_ids]
        unknown = sorted(set(int(m) for m in feedback_motor_ids) - set(motor_ids))
        if unknown:
            raise ValueError(f"Force-feedback motors {unknown} are not among the active motors {motor_ids}")
        self.detectors = {motor_ids.index(int(m)): ContactStateMachine() for m in feedback_motor_ids}
        self.armed = np.zeros(len(motor_ids), dtype=bool)

    def update(self, goal, positions, velocities, currents, commanded_velocities, now_s: float) -> np.ndarray:
        """Feedback (rad beyond the deadband, signed like the error) per joint; 0 without contact.

        A joint is watched only once it has come within the contact threshold
        of its goal, so the start-up ramp does not count as contact.
        """

        error = np.asarray(goal, dtype=np.float64) - np.asarray(positions, dtype=np.float64)
        feedback = np.zeros_like(error)
        for index, detector in self.detectors.items():
            if not self.armed[index]:
                self.armed[index] = abs(error[index]) <= CONTACT_THRESHOLD
                if not self.armed[index]:
                    continue
            feedback[index] = detector.update(
                commanded_velocity_rad_s=float(commanded_velocities[index]),
                measured_velocity_rad_s=float(velocities[index]),
                measured_current_ma=float(currents[index]),
                candidate_feedback=float(contact_feedback([error[index]], CONTACT_EXIT_THRESHOLD)[0]),
                now_s=now_s,
            )
        return feedback


def teleop_loop(
    leader,
    follower,
    start: Sequence[float],
    *,
    rate_hz: float = DEFAULT_RATE_HZ,
    ramp_s: float = DEFAULT_RAMP_S,
    max_jump_rad: float = DEFAULT_MAX_JUMP_RAD,
    print_period_s: float | None = None,
    contact: ContactFeedback | None = None,
    send_feedback: Callable[[np.ndarray, np.ndarray], None] | None = None,
    current_signs: Sequence[float] | None = None,
    log: dict[str, list] | None = None,
    keep_running: Callable[[], bool] = lambda: True,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Send the leader's angles to the follower until keep_running() is False; return cycles run.

    Over the first ramp_s seconds the goal blends from `start` (the follower's
    pose) to the leader's pose. With `contact`, the follower is read every
    cycle and `send_feedback(leader_angles, feedback)` gets its contact
    feedback; `log` collects LOG_FIELDS. Follower currents are multiplied by
    `current_signs` (its joint signs) so they share the angles' direction.
    """

    start = np.asarray(start, dtype=np.float64)
    signs = np.ones_like(start) if current_signs is None else np.asarray(current_signs, dtype=np.float64)
    read_follower = contact is not None or log is not None or print_period_s is not None
    period_s = 1.0 / rate_hz
    t0 = clock()
    next_cycle = next_print = t0
    previous = None
    last_goal = start.copy()
    last_time = t0
    feedback = np.zeros_like(start)
    cycles = 0
    while keep_running():
        positions = np.asarray(leader.read_pos(), dtype=np.float64)
        if positions.shape != start.shape or not np.isfinite(positions).all():
            raise RuntimeError(f"Leader returned an invalid reading: {positions}")
        if previous is not None:
            jump = np.abs(positions - previous)
            if jump.max() > max_jump_rad:
                index = int(jump.argmax())
                raise RuntimeError(
                    f"Leader joint {index} jumped {math.degrees(jump[index]):.0f} deg in one cycle; stopping"
                )
        previous = positions
        now = clock()
        alpha = 1.0 if ramp_s <= 0 else min(1.0, (now - t0) / ramp_s)
        goal = follower.clip_positions(start + alpha * (positions - start))
        follower.set_positions(goal, clip=True)
        if read_follower:
            measured, velocity, current = (np.asarray(v, dtype=np.float64) for v in follower.read_pos_vel_cur())
            current = current * signs
            if contact is not None and alpha >= 1.0:
                commanded_velocity = (goal - last_goal) / max(now - last_time, 1e-6)
                feedback = contact.update(goal, measured, velocity, current, commanded_velocity, now)
                if send_feedback is not None:
                    send_feedback(positions, feedback)
            if log is not None:
                for name, value in zip(LOG_FIELDS, (now - t0, positions, goal, measured, velocity, current, feedback)):
                    log.setdefault(name, []).append(np.array(value, dtype=np.float64))
            if print_period_s is not None and now >= next_print:
                error = np.degrees(measured - goal)
                worst = int(np.abs(error).argmax())
                in_contact = [i for i in np.nonzero(feedback)[0]]
                print("leader deg " + " ".join(f"{v:+5.0f}" for v in np.degrees(positions))
                      + f" | worst error {error[worst]:+.1f} deg (index {worst})"
                      + (f" | contact {in_contact}" if contact is not None else ""), flush=True)
                next_print = now + print_period_s
        last_goal, last_time = goal, now
        cycles += 1
        next_cycle += period_s
        if next_cycle < now:
            next_cycle = now
        sleep(max(0.0, next_cycle - now))
    return cycles


def save_log(path: Path, log: Mapping[str, list], motor_ids: Sequence[int], **meta) -> None:
    arrays = {name: np.asarray(log.get(name, []), dtype=np.float64) for name in LOG_FIELDS}
    np.savez(path, motor_ids=np.asarray(motor_ids), **arrays,
             **{key: np.asarray(value) for key, value in meta.items()})


def _resolve_port(given: str | None, config_port: str | None, role: str) -> str:
    port = given or config_port
    if port:
        return port
    found = sorted(p.name for p in SERIAL_BY_ID.iterdir()) if SERIAL_BY_ID.is_dir() else []
    listing = "\n  ".join(str(SERIAL_BY_ID / name) for name in found) or "none"
    raise RuntimeError(f"Pass --{role}-port. Adapters in {SERIAL_BY_ID}:\n  {listing}")


def _import_hand_api():
    try:
        import midas_hand_api  # noqa: F401
    except ImportError:
        if not HAND_API_CHECKOUT.is_dir():
            raise
        sys.path.insert(0, str(HAND_API_CHECKOUT))
    from midas_hand_api import HandConfig, MidasHand

    return HandConfig, MidasHand


def run(args: argparse.Namespace) -> int:
    import yaml

    HandConfig, MidasHand = _import_hand_api()
    from data_collection.collection.hand_common import load_hand_config, verify_connected_hand

    for path in (args.leader_config, args.follower_config):
        if not path.is_file():
            raise FileNotFoundError(f"Calibration not found: {path}. Home that hand first (see --help).")
    raw = {}
    for role, path in (("leader", args.leader_config), ("follower", args.follower_config)):
        with path.open("r", encoding="utf-8") as stream:
            raw[role] = yaml.safe_load(stream)
    check_pair(raw["leader"], raw["follower"])
    leader_port = _resolve_port(args.leader_port, HandConfig.load(args.leader_config).port, "leader")
    follower_port = _resolve_port(args.follower_port, HandConfig.load(args.follower_config).port, "follower")
    if Path(leader_port).resolve() == Path(follower_port).resolve():
        raise ValueError(f"Leader and follower are on the same port {leader_port}")

    leader_config = load_hand_config(args.leader_config, leader_port, args.motors)
    follower_config = load_hand_config(args.follower_config, follower_port, args.motors)
    motor_ids = [int(m) for m in follower_config.motor_ids]
    feedback_motors = motor_ids if args.feedback_motors is None else [int(m) for m in args.feedback_motors]
    contact = ContactFeedback(motor_ids, feedback_motors) if args.force_feedback else None
    current_ma = min(args.current_limit_ma, follower_config.goal_current_limit)
    print(f"Leader   {leader_port}\n         {args.leader_config}\n"
          f"Follower {follower_port}\n         {args.follower_config}\n"
          f"Motors {motor_ids}; follower current cap {current_ma:.0f} mA, "
          f"speed cap {args.max_speed_rad_s:g} rad/s, {args.rate_hz:g} Hz")
    if contact is not None:
        print(f"Force feedback on motors {feedback_motors}: up to {args.feedback_max_current_ma:g} mA "
              f"at {args.feedback_full_scale_rad:g} rad beyond the deadband")

    leader = follower = feedback_controller = None
    log: dict[str, list] | None = {} if args.log else None
    try:
        leader = MidasHand(config=leader_config, autoconnect=True)
        verify_connected_hand(leader, role="leader")
        follower = MidasHand(config=follower_config, autoconnect=True)
        verify_connected_hand(follower, role="follower")
        errors = {m: e for m, e in follower.read_hardware_error_status().items() if e}
        if errors:
            raise RuntimeError(f"Follower motors report hardware errors {errors}; power-cycle the follower")
        leader.disable_torque()
        follower.configure(enable_torque=False)
        for role, hand in (("leader", leader), ("follower", follower)):
            positions = hand.read_pos()
            bad = out_of_limits(positions, hand.config.joint_lower_limits_array, hand.config.joint_upper_limits_array)
            if bad:
                details = ", ".join(f"motor {hand.motor_ids[i]} reads {math.degrees(positions[i]):+.0f} deg"
                                    for i in bad)
                raise RuntimeError(f"The {role} is far outside its joint limits ({details}); home it again.")
        send_feedback = None
        if contact is not None:
            from data_collection.collection.leader_hand_worker import TrimorphForceFeedbackController

            feedback_controller = TrimorphForceFeedbackController(
                leader,
                feedback_motor_ids=feedback_motors,
                max_current_ma=args.feedback_max_current_ma,
                full_scale_error_rad=args.feedback_full_scale_rad,
                tactile_current_ma=0.0,
                tactile_motor_ids=((), (), (), ()),
            )
            feedback_controller.start(leader.read_pos())

            def send_feedback(leader_angles, feedback):
                feedback_controller.update(dict(zip(motor_ids, feedback)), leader_angles)

        follower.set_current_limit(int(round(current_ma)))
        follower.set_motion_profile(
            profile_velocity_rad_s=args.max_speed_rad_s,
            profile_acceleration_rad_s2=args.max_acceleration_rad_s2,
        )
        start = follower.clip_positions(follower.read_pos())
        follower.set_positions(start, clip=True)
        follower.enable_torque()
        print(f"Follower easing to the leader's pose over {args.ramp_s:g} s, then following. Ctrl-C stops.")
        cycles = teleop_loop(
            leader, follower, start,
            rate_hz=args.rate_hz, ramp_s=args.ramp_s, max_jump_rad=args.max_jump_rad,
            print_period_s=0.5 if args.print else None,
            contact=contact, send_feedback=send_feedback,
            current_signs=follower.config.joint_signs_array, log=log,
        )
        print(f"Stopped after {cycles} cycles.")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if follower is not None:
            follower.shutdown()
        if feedback_controller is not None:
            try:
                feedback_controller.shutdown()
            except Exception as exc:  # still switch the leader off below
                print(f"Warning: {exc}", file=sys.stderr)
        if leader is not None:
            leader.shutdown()
        print("Both hands' torque is off.")
        if log:
            save_log(args.log, log, motor_ids, rate_hz=args.rate_hz, current_limit_ma=current_ma,
                     force_feedback=args.force_feedback, ramp_s=args.ramp_s)
            print(f"Saved {len(log['t'])} cycles to {args.log}")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     epilog=__doc__.split("\n\n", 1)[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--leader-port", help="leader's serial device (default: the calibration's port)")
    parser.add_argument("--follower-port", help="follower's serial device (default: the calibration's port)")
    parser.add_argument("--leader-config", type=Path, default=DEFAULT_LEADER_CONFIG,
                        help="leader calibration from 'trimi-live home' (default: %(default)s)")
    parser.add_argument("--follower-config", type=Path, default=DEFAULT_FOLLOWER_CONFIG,
                        help="follower calibration from 'trimi-live home' (default: %(default)s)")
    parser.add_argument("--motors", type=int, nargs="+", help="only these motor IDs (default: all)")
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ, help="(default: %(default)g)")
    parser.add_argument("--ramp-s", type=float, default=DEFAULT_RAMP_S,
                        help="time to ease the follower onto the leader's pose at start (default: %(default)g)")
    parser.add_argument("--current-limit-ma", type=float, default=DEFAULT_CURRENT_LIMIT_MA,
                        help="follower current cap, at most the calibration's (default: %(default)g)")
    parser.add_argument("--max-speed-rad-s", type=float, default=DEFAULT_MAX_SPEED_RAD_S,
                        help="follower profile velocity (default: %(default)g)")
    parser.add_argument("--max-acceleration-rad-s2", type=float, default=DEFAULT_MAX_ACCELERATION_RAD_S2,
                        help="follower profile acceleration (default: %(default)g)")
    parser.add_argument("--max-jump-rad", type=float, default=DEFAULT_MAX_JUMP_RAD,
                        help="stop if a leader reading changes more than this in one cycle (default: %(default)g)")
    parser.add_argument("--force-feedback", action="store_true",
                        help="push back on leader joints whose follower joint is in contact")
    parser.add_argument("--feedback-motors", type=int, nargs="+",
                        help="motor IDs that get force feedback (default: all active)")
    parser.add_argument("--feedback-max-current-ma", type=float, default=DEFAULT_FEEDBACK_MAX_CURRENT_MA,
                        help="largest leader feedback current (default: %(default)g)")
    parser.add_argument("--feedback-full-scale-rad", type=float, default=DEFAULT_FEEDBACK_FULL_SCALE_RAD,
                        help="follower error beyond the 0.03 rad deadband that gives the largest current "
                             "(default: %(default)g; larger is a softer spring)")
    parser.add_argument("--log", type=Path, help="save every cycle to this .npz")
    parser.add_argument("--print", action="store_true",
                        help="print leader angles, follower error and contacts twice a second")
    args = parser.parse_args(argv)
    for name in ("rate_hz", "current_limit_ma", "max_speed_rad_s", "max_acceleration_rad_s2", "max_jump_rad",
                 "feedback_max_current_ma", "feedback_full_scale_rad"):
        value = getattr(args, name)
        if not (math.isfinite(value) and value > 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not (math.isfinite(args.ramp_s) and args.ramp_s >= 0):
        parser.error("--ramp-s must be finite and not negative")
    if args.feedback_motors is not None and not args.force_feedback:
        parser.error("--feedback-motors needs --force-feedback")
    args.leader_config = args.leader_config.expanduser()
    args.follower_config = args.follower_config.expanduser()
    if args.log is not None:
        args.log = args.log.expanduser()
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
