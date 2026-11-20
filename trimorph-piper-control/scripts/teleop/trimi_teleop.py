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

Force feedback (--force-feedback): a follower joint is in contact when, for
20 ms, it is still (under --contact-still-rad-s) and the leader is more than
--contact-error-rad past it and still pulling away from it (a gap that stays
put is the follower falling short of its goal, not an object; a gap past twice
the contact error counts anyway). The motor current does not count by default: it
mixes gear ratio, moment arm, acceleration and friction, differs from finger
to finger, and cannot tell a fast move from a light touch.
--contact-motors limits contact detection to some joints (the others are still
walled with their finger). --contact-motor-overrides (e.g. 0:error_rad=0.06,hold_ms=60) sets the error
and hold time of single joints, for one that lags behind more than the rest.
--contact-current-ma N adds the condition that the follower also pushes
towards its goal with more than N mA. --grasp-only makes the finger PIP/MCP
(flexion reads negative), and any MOTOR=+1/-1 given, count only errors in the
grasping direction; the other joints count both directions.

At contact the leader is stopped where the follower stopped (a wall): the
joint's own motor holds the leader at the follower's position at contact, in
current-based position mode with up to --wall-current-ma, its internal loop
far stiffer than anything this 100 Hz loop could render. The wall stays there
(with --wall-follow it moves with the follower instead, so a soft object
gives, and so do the follower's own flex and gear play). The follower keeps
following the leader, so pressing harder bends the leader a little past the
wall and the follower squeezes harder. For the first --wall-settle-ms the wall
holds whatever the hand does, so the leader is brought back to where the
follower stopped (at no more than --wall-speed-rad-s) before it can be let go;
only the follower moving on ends it sooner. After that the wall lets go the
moment the hand opens: the leader comes out past it by --wall-release-rad, or
its motor current turns to pull the finger in by more than --wall-release-ma.
At any time it lets go when the follower moves on past it by --wall-pass-rad
or faster than --contact-still-rad-s: nothing was there. --wall-p-gain/-i-gain/
-d-gain set its stiffness (default: the leader calibration's), and
--wall-motor-gains (e.g. 0:p=600,i=0) a single joint's, for a softer wall on a
joint geared too high to give under the hand.

Walls go up a whole finger at a time (thumb 0-3, index 4-6, middle 7-9, ring
10-12): while any of its joints is in contact, the others are held too, each
at its follower's position, and the hand opening at any of them (judged in the
grasping direction for a joint not in contact itself) takes them all down.
--no-wall-whole-finger walls each joint on its own. Out of contact the
leader's joints are unpowered (torque off): a powered Dynamixel drags when
turned by hand, most on the geared PIP and MCP joints.

--log saves every cycle (leader angles, follower goals, angles, velocities and
currents, the joints in contact and the leader joints held) for
scripts/teleop/analyze_teleop_log.py. --plot shows each joint's follower-minus-leader angle (or with --plot current, the
follower's current) live in a separate window, red while in contact, orange
while held only because the rest of its finger is.

The two hands are read at the same time, each on its own bus, so a cycle takes
about as long as the slower read instead of both.

Settings come from built-in defaults, then a tuning YAML (config/
trimi_teleop_tuning.yaml unless --tuning names another, skipped with
--no-tuning), then the command line. Its keys are the long option names
(wall_current_ma: 1200). --save-tuning FILE writes the settings a
command line amounts to and exits.
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

from data_collection.collection.contact_detection import CONTACT_THRESHOLD

CALIBRATION_DIR = Path.home() / ".config/trimi-copilot"
DEFAULT_LEADER_CONFIG = CALIBRATION_DIR / "hand_leader.yaml"
DEFAULT_FOLLOWER_CONFIG = CALIBRATION_DIR / "hand_follower.yaml"
HAND_API_CHECKOUT = REPO_ROOT.parent / "trimorph_hand_api"  # the monorepo's copy
DEFAULT_TUNING = REPO_ROOT / "config/trimi_teleop_tuning.yaml"
# The settings --save-tuning writes, in this order.
TUNING_KEYS = (
    "current_limit_ma", "max_speed_rad_s", "max_acceleration_rad_s2",
    "follower_p_gain", "follower_i_gain", "follower_d_gain", "rate_hz", "ramp_s", "max_jump_rad",
    "contact_error_rad", "contact_current_ma", "contact_still_rad_s", "contact_hold_ms", "stop_margin_rad",
    "contact_motors", "contact_motor_overrides",
    "wall_current_ma", "wall_whole_finger", "wall_follow", "wall_settle_ms", "wall_release_rad", "wall_release_ma", "wall_pass_rad",
    "wall_speed_rad_s",
    "wall_p_gain", "wall_i_gain", "wall_d_gain", "wall_motor_gains",
    "force_feedback", "feedback_motors", "grasp_only", "motors",
    "leader_port", "follower_port", "leader_config", "follower_config", "plot", "plot_window_s", "print",
)
SERIAL_BY_ID = Path("/dev/serial/by-id")
DEFAULT_RATE_HZ = 100.0
DEFAULT_RAMP_S = 2.0
DEFAULT_CURRENT_LIMIT_MA = 400.0
DEFAULT_MAX_SPEED_RAD_S = 6.0
DEFAULT_MAX_ACCELERATION_RAD_S2 = 80.0
DEFAULT_MAX_JUMP_RAD = 1.0
DEFAULT_WALL_CURRENT_MA = 700.0  # the wall's holding current; the XL330 allows 1750
STOP_MARGIN_RAD = 0.1  # a follower this close to the stop homing found is at that stop, not on an object
DEFAULT_WALL_RELEASE_RAD = 0.01  # the leader this far out of the wall: the hand is opening
DEFAULT_WALL_RELEASE_MA = 40.0  # the wall's motor pulling the finger in this hard: the hand is opening
DEFAULT_WALL_SPEED_RAD_S = 3.0  # how fast the leader is brought back to the wall it went past
DEFAULT_WALL_SETTLE_MS = 120.0  # after contact, the wall holds this long so the leader gets back to it
DEFAULT_WALL_PASS_RAD = 0.03  # the follower this far past the wall: there is nothing there
CONTACT_STILL_RAD_S = 0.25  # held back below this; a slow follower still tracking moves 0.2-0.6 rad/s
CONTACT_GROWTH_RAD = 0.004  # the gap must grow this much over the hold time (about 3 encoder steps)
CONTACT_PERSISTENCE_S = 0.02  # longer than the follower takes to speed up past CONTACT_STILL_RAD_S
DEFAULT_CONTACT_ERROR_RAD = 0.04  # leader past the follower by this much: contact
DEFAULT_CONTACT_CURRENT_MA = None  # off: contact from positions only
# Flexion reads negative on the finger PIP and MCP joints (limits about -2.9 .. 0.02).
FINGER_FLEXION_DIRECTIONS = {4: -1, 5: -1, 7: -1, 8: -1, 10: -1, 11: -1}
LIMIT_MARGIN_RAD = math.radians(20.0)
# Motors of each finger: thumb, index, middle, ring (walled together by default).
FINGER_GROUPS = ((0, 1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12))
LOG_FIELDS = ("t", "leader", "goal", "position", "velocity", "current", "contact",
              "leader_velocity", "leader_current", "wall")
# Starts the line printed once the follower is set up and starts following; quest_trimi_teleop.py waits for it.
FOLLOWING_MESSAGE = "Follower easing to the leader's pose"


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


def homing_stop_readings(calibration: Mapping) -> dict[int, tuple[float, int]]:
    """Per motor, the reading at the stop homing drives into and the reading's direction towards it.

    home_offset = raw_stop + cad_offset, so the stop reads -joint_sign * cad_offset.
    """

    motor_ids = [int(m) for m in calibration["motor_ids"]]
    signs = _by_motor(calibration["joint_signs"], motor_ids)
    homing = calibration.get("homing") or {}
    stops = {}
    for motor, _name, cad, direction in list(homing.get("thumb", [])) + list(homing.get("fingers", [])):
        sign = signs[int(motor)]
        stops[int(motor)] = (-sign * float(cad), 1 if sign * float(direction) > 0 else -1)
    return stops


def out_of_limits(positions, lower, upper, margin: float = LIMIT_MARGIN_RAD) -> list[int]:
    """Indices of joints reading farther than `margin` outside their limits."""

    positions = np.asarray(positions, dtype=np.float64)
    bad = (positions < np.asarray(lower) - margin) | (positions > np.asarray(upper) + margin)
    return [int(i) for i in np.where(bad | ~np.isfinite(positions))[0]]


class BlockedContact:
    """Contact: the follower held still short of its goal (and, optionally, pushing towards it).

    In `direction` (+1/-1) only, or either way (0). Entering needs the
    position error beyond enter_rad (and, unless enter_current_ma is None,
    the follower's joint-signed current beyond it, both towards the goal) for
    persistence_s, with the error grown by growth_rad over that time, so the
    leader is still going on while the follower stays (an error that stays
    put is the follower falling short of its goal); an error past twice
    enter_rad needs no growth. The detector resets once the error is back
    inside exit_rad (default: half of enter_rad).
    """

    def __init__(self, direction: int = 0, *, enter_rad: float = DEFAULT_CONTACT_ERROR_RAD,
                 enter_current_ma: float | None = DEFAULT_CONTACT_CURRENT_MA, exit_rad: float | None = None,
                 still_rad_s: float = CONTACT_STILL_RAD_S, persistence_s: float = CONTACT_PERSISTENCE_S,
                 growth_rad: float = CONTACT_GROWTH_RAD):
        exit_rad = enter_rad / 2 if exit_rad is None else exit_rad
        if enter_rad <= exit_rad:
            raise ValueError(f"The contact error {enter_rad:g} rad must exceed the release error {exit_rad:g} rad")
        self.direction = float(direction)
        self.enter_rad, self.exit_rad = enter_rad, exit_rad
        self.enter_current_ma = enter_current_ma
        self.still_rad_s, self.persistence_s = still_rad_s, persistence_s
        self.growth_rad = growth_rad
        self.engaged = False
        self.side = self.direction
        self._since: float | None = None
        self._depth_at_since = 0.0

    def update(self, error: float, follower_velocity: float, follower_current: float, now_s: float) -> float:
        """The side pushed into (+1/-1) while in contact, 0 otherwise."""

        if self.engaged:
            if self.side * error < self.exit_rad:
                self.engaged, self._since = False, None
                return 0.0
            return self.side
        side = self.direction or (1.0 if error >= 0 else -1.0)
        depth = side * error
        pushing = self.enter_current_ma is None or side * follower_current > self.enter_current_ma
        if depth > self.enter_rad and pushing and abs(follower_velocity) < self.still_rad_s:
            if self._since is None or side != self.side:
                self._since, self.side, self._depth_at_since = now_s, side, depth
            grown = depth - self._depth_at_since >= self.growth_rad or depth >= 2 * self.enter_rad
            if now_s - self._since >= self.persistence_s and grown:
                self.engaged = True
                return side
            return 0.0
        self._since = None
        return 0.0


class ContactDetector:
    """BlockedContact for each feedback joint.

    Joints in `directions` (motor -> +1 or -1) count only errors of that sign,
    the grasping direction; the others count both. `stops` (motor ->
    (reading, direction), from homing_stop_readings) mark each joint's own hard
    stop: a follower held there is at the stop, not on an object (the leader
    meets its own stop there too).
    """

    def __init__(self, motor_ids: Sequence[int], feedback_motor_ids: Sequence[int],
                 directions: Mapping[int, int] | None = None, *,
                 stops: Mapping[int, tuple[float, int]] | None = None, stop_margin_rad: float = STOP_MARGIN_RAD,
                 enter_rad: float = DEFAULT_CONTACT_ERROR_RAD,
                 enter_current_ma: float | None = DEFAULT_CONTACT_CURRENT_MA,
                 still_rad_s: float = CONTACT_STILL_RAD_S, persistence_s: float = CONTACT_PERSISTENCE_S,
                 motor_overrides: Mapping[int, Mapping[str, float]] | None = None):
        """`motor_overrides` (motor -> error_rad / hold_ms) replace enter_rad / persistence_s for single joints."""

        motor_ids = [int(m) for m in motor_ids]
        directions = {int(m): int(d) for m, d in (directions or {}).items()}
        unknown = sorted((set(int(m) for m in feedback_motor_ids) | set(directions)) - set(motor_ids))
        if unknown:
            raise ValueError(f"Force-feedback motors {unknown} are not among the active motors {motor_ids}")
        if any(d not in (-1, 1) for d in directions.values()):
            raise ValueError(f"Grasp directions must be +1 or -1: {directions}")
        overrides = {int(m): dict(o) for m, o in (motor_overrides or {}).items()}
        unknown = sorted(set(overrides) - set(int(m) for m in feedback_motor_ids))
        if unknown:
            raise ValueError(f"Contact settings given for motors {unknown}, which detect no contact")
        self.detectors = {
            motor_ids.index(int(m)): BlockedContact(
                directions.get(int(m), 0),
                enter_rad=float(overrides.get(int(m), {}).get("error_rad", enter_rad)),
                enter_current_ma=enter_current_ma, still_rad_s=still_rad_s,
                persistence_s=float(overrides.get(int(m), {}).get("hold_ms", 1000 * persistence_s)) / 1000)
            for m in feedback_motor_ids
        }
        self.armed = np.zeros(len(motor_ids), dtype=bool)
        stops = {int(m): v for m, v in (stops or {}).items() if int(m) in motor_ids}
        self.stop_reading = np.full(len(motor_ids), np.nan)
        self.stop_direction = np.zeros(len(motor_ids))
        for m, (reading, direction) in stops.items():
            self.stop_reading[motor_ids.index(m)] = reading
            self.stop_direction[motor_ids.index(m)] = direction
        self.stop_margin_rad = stop_margin_rad

    def update(self, goal, positions, velocities, currents, now_s: float) -> np.ndarray:
        """The side pushed into (+1/-1) per joint in contact, 0 elsewhere.

        A joint is watched only once it has come within the contact threshold
        of its goal, so the start-up ramp does not count as contact.
        """

        positions = np.asarray(positions, dtype=np.float64)
        error = np.asarray(goal, dtype=np.float64) - positions
        to_stop = self.stop_direction * (self.stop_reading - positions)  # nan where unknown
        at_stop = (to_stop < self.stop_margin_rad) & (self.stop_direction * error > 0)
        error = np.where(at_stop, 0.0, error)
        found = np.zeros_like(error)
        for index, detector in self.detectors.items():
            if not self.armed[index]:
                self.armed[index] = abs(error[index]) <= CONTACT_THRESHOLD
                if not self.armed[index]:
                    continue
            found[index] = detector.update(float(error[index]), float(velocities[index]),
                                           float(currents[index]), now_s)
        return found


class WallContact:
    """Contact for the wall: entered like ContactDetector, left when the hand opens.

    Once a joint is in contact the leader is held at a wall, so the
    follower's error no longer says anything. A joint's contact ends when its
    follower moves on into the free space faster than still_rad_s or past the
    wall by pass_rad (nothing was there). The walls come down when the hand
    opens: a leader joint comes out of its wall by release_rad, or its own
    current turns to pull the finger in by release_ma, judged on the side
    pushed into (a held joint not in contact itself: its grasp direction, if
    it has one). For settle_s after the walls go up only the follower moving
    on counts, so the leader is first brought back to them.

    `groups` (tuples of motor IDs; default: each joint alone) are walled
    together: while any joint of a group is in contact, all its joints are
    held, and the hand opening at any of them takes all its walls down.
    `walls` holds where the held leader joints are held: the follower's
    position when they were first held, or with `follow` its position now;
    `engaged` marks them.
    """

    def __init__(self, motor_ids: Sequence[int], feedback_motor_ids: Sequence[int],
                 directions: Mapping[int, int] | None = None, *, release_rad: float = DEFAULT_WALL_RELEASE_RAD,
                 release_ma: float = DEFAULT_WALL_RELEASE_MA, still_rad_s: float = CONTACT_STILL_RAD_S,
                 follow: bool = False, settle_s: float = DEFAULT_WALL_SETTLE_MS / 1000.0,
                 pass_rad: float = DEFAULT_WALL_PASS_RAD, groups: Sequence[Sequence[int]] | None = None,
                 contact_motor_ids: Sequence[int] | None = None, **detect):
        contact_motor_ids = feedback_motor_ids if contact_motor_ids is None else contact_motor_ids
        outside = sorted(set(int(m) for m in contact_motor_ids) - set(int(m) for m in feedback_motor_ids))
        if outside:
            raise ValueError(f"Contact motors {outside} are not among the wall motors {list(feedback_motor_ids)}")
        self.detect = ContactDetector(motor_ids, contact_motor_ids, directions, still_rad_s=still_rad_s, **detect)
        motor_ids = [int(m) for m in motor_ids]
        index = {m: i for i, m in enumerate(motor_ids)}
        self.indices = sorted(index[int(m)] for m in feedback_motor_ids)
        walled = set(self.indices)
        self.groups: list[list[int]] = []  # active wall joints walled together, as indices
        for group in groups or ():
            members = sorted(index[int(m)] for m in group if int(m) in index and index[int(m)] in walled)
            if members and not walled.isdisjoint(members):
                self.groups.append(members)
                walled -= set(members)
        self.groups += [[i] for i in sorted(walled)]
        self.grasp = np.zeros(len(motor_ids))
        for m, d in (directions or {}).items():
            if int(m) in index:
                self.grasp[index[int(m)]] = float(d)
        self.release_rad, self.release_ma, self.still_rad_s = release_rad, release_ma, still_rad_s
        self.follow = follow
        self.settle_s = float(settle_s)
        self.pass_rad = float(pass_rad)
        self.since = np.full(len(motor_ids), np.nan)  # when each group's walls went up
        self.engaged = np.zeros(len(motor_ids), dtype=bool)  # leader held at walls[i]
        self.side = np.zeros(len(motor_ids))  # the side pushed into, per joint in contact
        self.walls = np.full(len(motor_ids), np.nan)

    def _release(self, members) -> None:
        self.engaged[members], self.side[members] = False, 0.0
        self.walls[members], self.since[members] = np.nan, np.nan

    def update(self, goal, positions, velocities, currents, now_s: float, *, leader_positions,
               leader_currents) -> np.ndarray:
        """+-1 (the side pushed into) per joint in contact, 0 elsewhere; `engaged` marks the walls."""

        positions = np.asarray(positions, dtype=np.float64)
        found = self.detect.update(goal, positions, velocities, currents, now_s)
        for members in self.groups:
            for i in members:
                side = self.side[i]
                if side == 0.0:
                    if found[i] != 0.0:
                        self.side[i] = found[i]
                        if not self.engaged[i]:
                            self.walls[i] = positions[i]
                elif (side * velocities[i] > self.still_rad_s
                      or side * (positions[i] - self.walls[i]) > self.pass_rad):
                    self.side[i] = 0.0  # the follower moved on: nothing there
                    self.walls[i] = positions[i]  # still held while the group is, from here
            if not self.side[members].any():
                self._release(members)
                continue
            if not self.engaged[members].any():
                self.since[members] = now_s  # the walls go up
            for i in members:
                if not self.engaged[i]:
                    self.engaged[i] = True
                    if np.isnan(self.walls[i]):
                        self.walls[i] = positions[i]
                elif self.follow:
                    self.walls[i] = positions[i]
            if now_s - self.since[members[0]] < self.settle_s:
                continue
            for i in members:
                side = self.side[i] or self.grasp[i]
                if side and (side * (leader_positions[i] - self.walls[i]) < -self.release_rad
                             or side * leader_currents[i] > self.release_ma):
                    self._release(members)  # the hand opens
                    break
        return self.side.copy()


class LeaderWall:
    """Leader joints held at their wall by their own position control, unpowered otherwise."""

    def __init__(self, hand, motor_ids: Sequence[int], feedback_motor_ids: Sequence[int], *,
                 current_ma: float, speed_rad_s: float = DEFAULT_WALL_SPEED_RAD_S,
                 gains: Mapping[str, int] | None = None,
                 motor_gains: Mapping[int, Mapping[str, int]] | None = None):
        """`gains` (p/i/d) for every wall motor, then `motor_gains` (motor -> p/i/d) over them."""

        from midas_hand_api.actuators import control_table

        self.ct = control_table
        self.hand = hand
        self.client = hand.dxl_client
        if self.client is None:
            raise RuntimeError("The leader is not connected")
        if current_ma > hand.config.max_current_limit_ma:
            raise ValueError(f"Wall current {current_ma:g} mA exceeds the motors' {hand.config.max_current_limit_ma} mA")
        index = {int(m): i for i, m in enumerate(motor_ids)}
        self.ids = [int(m) for m in feedback_motor_ids]
        self.indices = np.asarray([index[m] for m in self.ids], dtype=np.int64)
        self.signs = np.asarray(hand.config.joint_signs_array, dtype=np.float64)[self.indices]
        self.offsets = np.asarray(hand.config.home_offsets_array, dtype=np.float64)[self.indices]
        self.unit_ma = float(hand.config.current_unit_ma)
        self.current_raw = int(current_ma / self.unit_ma)
        self.restore_raw = int(hand.config.goal_current_limit / self.unit_ma)
        self.speed_rad_s = float(speed_rad_s)
        self.gains = {k: int(v) for k, v in (gains or {}).items() if v is not None}
        self.motor_gains = {int(m): {k: int(v) for k, v in g.items() if v is not None}
                            for m, g in (motor_gains or {}).items()}
        unknown = sorted(set(self.motor_gains) - set(self.ids))
        if unknown:
            raise ValueError(f"Wall gains given for motors {unknown}, which get no wall (wall motors {self.ids})")
        self.powered: set[int] = set()
        self.started = False

    def _write(self, ids, values, address, size) -> None:
        if ids:
            self.client.sync_write(list(ids), [int(v) for v in values], address, size)

    def start(self) -> None:
        ct = self.ct
        self.client.set_torque_enabled(self.ids, False, retries=3, verify=True)
        self._write(self.ids, [ct.OPERATING_MODE_CURRENT_BASED_POSITION] * len(self.ids), ct.ADDR_OPERATING_MODE,
                    ct.LEN_OPERATING_MODE)
        self._write(self.ids, [self.current_raw] * len(self.ids), ct.ADDR_GOAL_CURRENT, ct.LEN_GOAL_CURRENT)
        if self.gains:
            self.hand.set_gains(self.ids, **self.gains)
        for motor, gains in self.motor_gains.items():
            if gains:
                self.hand.set_gains([motor], **gains)
        self.hand.set_motion_profile(profile_velocity_rad_s=self.speed_rad_s, motor_ids=self.ids)
        self.started = True

    def update(self, engaged, walls) -> None:
        """Hold the engaged joints at `walls` (calibrated, per active motor); release the rest."""

        ct = self.ct
        engaged = np.asarray(engaged, dtype=bool)[self.indices]
        walls = np.asarray(walls, dtype=np.float64)[self.indices]
        on = [m for m, e in zip(self.ids, engaged) if e]
        if on:
            raw = (walls * self.signs + self.offsets)[engaged]
            self.client.write_desired_pos(on, raw)  # the goal first, so the motor never jumps to an old one
        start = [m for m in on if m not in self.powered]
        stop = [m for m in self.ids if m in self.powered and m not in on]
        self._write(start, [1] * len(start), ct.ADDR_TORQUE_ENABLE, ct.LEN_TORQUE_ENABLE)
        self._write(stop, [0] * len(stop), ct.ADDR_TORQUE_ENABLE, ct.LEN_TORQUE_ENABLE)
        self.powered = set(on)

    def shutdown(self) -> None:
        """Torque off, with the calibration's current cap, gains and no profile limit."""

        if not self.started:
            return
        ct = self.ct
        self.client.set_torque_enabled(self.ids, False, retries=3, verify=True)
        self._write(self.ids, [self.restore_raw] * len(self.ids), ct.ADDR_GOAL_CURRENT, ct.LEN_GOAL_CURRENT)
        config = self.hand.config
        self.hand.set_gains(self.ids, p=int(config.position_p_gain), i=int(config.position_i_gain),
                            d=int(config.position_d_gain))
        self.hand.set_motion_profile(motor_ids=self.ids)
        self.powered, self.started = set(), False


def teleop_loop(
    leader,
    follower,
    start: Sequence[float],
    *,
    rate_hz: float = DEFAULT_RATE_HZ,
    ramp_s: float = DEFAULT_RAMP_S,
    max_jump_rad: float = DEFAULT_MAX_JUMP_RAD,
    print_period_s: float | None = None,
    contact: WallContact | None = None,
    send_feedback: Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray | None] | None = None,
    current_signs: Sequence[float] | None = None,
    leader_current_signs: Sequence[float] | None = None,
    log: dict[str, list] | None = None,
    monitor: Callable[[float, np.ndarray, np.ndarray, np.ndarray], None] | None = None,
    keep_running: Callable[[], bool] = lambda: True,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Send the leader's angles to the follower until keep_running() is False; return cycles run.

    Over the first ramp_s seconds the goal blends from `start` (the follower's
    pose) to the leader's pose. When the follower is read (for `contact`,
    `log`, `monitor` or printing), the two hands are read at the same time on
    their own buses, and the follower's state is compared with the goal it was
    following, the one sent the cycle before. `send_feedback(leader_angles,
    leader_velocities, feedback, dt)` gets the contact feedback (and may return
    the leader currents it sent); `log` collects LOG_FIELDS ("contact": the
    side each joint in contact pushes into, "wall": 1 where the leader is
    held) and `monitor(t, current, error, feedback)` sees every cycle. Follower currents
    are multiplied by `current_signs` (its joint signs) so they share the
    angles' direction.
    """

    from concurrent.futures import ThreadPoolExecutor

    start = np.asarray(start, dtype=np.float64)
    signs = np.ones_like(start) if current_signs is None else np.asarray(current_signs, dtype=np.float64)
    leader_signs = (np.ones_like(start) if leader_current_signs is None
                    else np.asarray(leader_current_signs, dtype=np.float64))
    leader_measured_current = np.zeros_like(start)
    read_follower = any(x is not None for x in (contact, log, monitor, print_period_s))
    period_s = 1.0 / rate_hz
    t0 = clock()
    next_cycle = next_print = t0
    previous = None
    last_time = t0
    last_goal = start.copy()  # what the follower is following
    feedback = np.zeros_like(start)
    held = np.zeros_like(start)  # leader joints held at a wall
    leader_velocity = np.zeros_like(start)
    leader_current = np.zeros_like(start)
    cycles = 0
    busy: list[float] = []  # bus and compute time per cycle since the last print

    def read_leader():
        if contact is not None:
            return tuple(np.asarray(v, dtype=np.float64) for v in leader.read_pos_vel_cur())
        return np.asarray(leader.read_pos(), dtype=np.float64), np.zeros_like(start), np.zeros_like(start)

    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bus") if read_follower else None
    try:
        while keep_running():
            cycle_start = clock()
            if pool is not None:
                leader_reading = pool.submit(read_leader)
                follower_reading = pool.submit(follower.read_pos_vel_cur)
                positions, leader_velocity, leader_measured_current = leader_reading.result()
                leader_measured_current = leader_measured_current * leader_signs
                measured, velocity, current = (np.asarray(v, dtype=np.float64) for v in follower_reading.result())
                current = current * signs
            else:
                positions, _, _ = read_leader()
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
            if read_follower:
                if contact is not None and alpha >= 1.0:
                    feedback = contact.update(last_goal, measured, velocity, current, now,
                                              leader_positions=positions, leader_currents=leader_measured_current)
                    held = np.asarray(contact.engaged, dtype=np.float64)
                    leader_current = leader_measured_current
                    if send_feedback is not None:
                        sent = send_feedback(positions, leader_velocity, feedback, now - last_time)
                        if sent is not None:
                            leader_current = np.asarray(sent, dtype=np.float64)
                error = measured - last_goal
                if log is not None:
                    values = (now - t0, positions, last_goal, measured, velocity, current, feedback,
                              leader_velocity, leader_current, held)
                    for name, value in zip(LOG_FIELDS, values):
                        log.setdefault(name, []).append(np.array(value, dtype=np.float64))
                if monitor is not None:
                    monitor(now - t0, current, error, feedback)
                if print_period_s is not None and now >= next_print:
                    degrees = np.degrees(error)
                    worst = int(np.abs(degrees).argmax())
                    in_contact = [int(i) for i in np.nonzero(feedback)[0]]
                    walled = [int(i) for i in np.nonzero(held)[0]]
                    timing = (f" | loop {len(busy) / print_period_s:.0f} Hz, busy {1000 * np.mean(busy):.1f}"
                              f"/{1000 * np.max(busy):.1f} ms mean/max" if busy and cycles >= len(busy) else "")
                    print("leader deg " + " ".join(f"{v:+5.0f}" for v in np.degrees(positions))
                          + f" | worst error {degrees[worst]:+.1f} deg (index {worst})"
                          + (f" | contact {in_contact} wall {walled}" if contact is not None else "") + timing,
                          flush=True)
                    busy.clear()
                    next_print = now + print_period_s
            follower.set_positions(goal, clip=True)
            last_goal, last_time = goal, now
            busy.append(clock() - cycle_start)
            cycles += 1
            next_cycle += period_s
            if next_cycle < now:
                next_cycle = now
            sleep(max(0.0, next_cycle - now))
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
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
    """midas_hand_api from the environment, else from the monorepo's trimorph_hand_api checkout."""

    import importlib.util

    if importlib.util.find_spec("midas_hand_api") is None and HAND_API_CHECKOUT.is_dir():
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
    contact_motors = feedback_motors if args.contact_motors is None else [int(m) for m in args.contact_motors]
    current_ma = min(args.current_limit_ma, follower_config.goal_current_limit)
    directions = None
    if args.grasp_only is not None:
        directions = {m: d for m, d in FINGER_FLEXION_DIRECTIONS.items() if m in feedback_motors}
        directions.update(args.grasp_only)
    if args.force_feedback and args.contact_current_ma is not None and args.contact_current_ma >= current_ma:
        raise ValueError(f"--contact-current-ma {args.contact_current_ma:g} must be below the follower's "
                         f"current cap {current_ma:g} mA, or no contact is ever seen")
    contact = None
    if args.force_feedback:
        contact = WallContact(
            motor_ids, feedback_motors, directions, release_rad=args.wall_release_rad,
            release_ma=args.wall_release_ma, still_rad_s=args.contact_still_rad_s, follow=args.wall_follow,
            settle_s=args.wall_settle_ms / 1000.0, pass_rad=args.wall_pass_rad,
            groups=FINGER_GROUPS if args.wall_whole_finger else None,
            stops=homing_stop_readings(raw["follower"]), stop_margin_rad=args.stop_margin_rad,
            enter_rad=args.contact_error_rad, enter_current_ma=args.contact_current_ma,
            persistence_s=args.contact_hold_ms / 1000.0, motor_overrides=args.contact_motor_overrides,
            contact_motor_ids=contact_motors)
    print(f"Leader   {leader_port}\n         {args.leader_config}\n"
          f"Follower {follower_port}\n         {args.follower_config}\n"
          f"Motors {motor_ids}; follower current cap {current_ma:.0f} mA, "
          f"speed cap {args.max_speed_rad_s:g} rad/s, {args.rate_hz:g} Hz")
    if contact is not None:
        pushing = ("" if args.contact_current_ma is None
                   else f" and pushing with over {args.contact_current_ma:g} mA")
        print(f"Contact on motors {contact_motors}: follower under {args.contact_still_rad_s:g} rad/s and over {args.contact_error_rad:g} rad "
              f"short of its goal{pushing} for {args.contact_hold_ms:g} ms" + "".join(
                  f"; motor {m}: over {o.get('error_rad', args.contact_error_rad):g} rad for "
                  f"{o.get('hold_ms', args.contact_hold_ms):g} ms"
                  for m, o in sorted((args.contact_motor_overrides or {}).items())))
        print(f"Wall on motors {feedback_motors}"
              + (", a whole finger at a time" if args.wall_whole_finger else ", joint by joint")
              + ": the leader held at the follower's "
              f"{'current position' if args.wall_follow else 'position at contact'} with up to "
              f"{args.wall_current_ma:g} mA, back to it at {args.wall_speed_rad_s:g} rad/s and held "
              f"{args.wall_settle_ms:g} ms; then let go when it comes {args.wall_release_rad:g} rad out or pulls in "
              f"with {args.wall_release_ma:g} mA")
        if directions:
            print(f"Grasp direction only on {directions}")

    leader = follower = feedback_controller = plot = None
    log: dict[str, list] | None = {} if args.log else None
    if log is not None:
        print(f"Logging every cycle to {args.log.resolve()} (written when it stops)")
    _stop_on_signals()  # closing the terminal or kill: stop like Ctrl-C, so the hands go off and the log is kept
    try:
        if args.plot:
            from scripts.teleop.live_plot import LivePlot

            labels = {int(row[0]): str(row[1]) for rows in (raw["follower"].get("homing") or {}).values()
                      if isinstance(rows, list) for row in rows}
            if args.plot == "error":
                plot = LivePlot(motor_ids, labels, limit=30.0, unit="deg", window_s=args.plot_window_s,
                                title="follower - leader (deg)")
            else:
                plot = LivePlot(motor_ids, labels, limit=current_ma, unit="mA", window_s=args.plot_window_s,
                                title="follower current (mA)")
        leader = MidasHand(config=leader_config, autoconnect=True)
        verify_connected_hand(leader, role="leader")
        follower = MidasHand(config=follower_config, autoconnect=True)
        verify_connected_hand(follower, role="follower")
        errors = {m: e for m, e in follower.read_hardware_error_status().items() if e}
        if errors:
            raise RuntimeError(f"Follower motors report hardware errors {errors}; power-cycle the follower")
        leader.disable_torque()
        follower.configure(enable_torque=False)
        gains = {k: v for k, v in (("p", args.follower_p_gain), ("i", args.follower_i_gain),
                                   ("d", args.follower_d_gain)) if v is not None}
        if gains:
            follower.set_gains(**{k: int(v) for k, v in gains.items()})
        print("Follower gains P {} I {} D {}".format(
            *(gains.get(k, default) for k, default in (("p", follower_config.position_p_gain),
                                                        ("i", follower_config.position_i_gain),
                                                        ("d", follower_config.position_d_gain)))))
        for role, hand in (("leader", leader), ("follower", follower)):
            positions = hand.read_pos()
            bad = out_of_limits(positions, hand.config.joint_lower_limits_array, hand.config.joint_upper_limits_array)
            if bad:
                details = ", ".join(f"motor {hand.motor_ids[i]} reads {math.degrees(positions[i]):+.0f} deg"
                                    for i in bad)
                raise RuntimeError(f"The {role} is far outside its joint limits ({details}); home it again.")
        send_feedback = None
        if contact is not None:
            feedback_controller = LeaderWall(
                leader, motor_ids, feedback_motors, current_ma=args.wall_current_ma,
                speed_rad_s=args.wall_speed_rad_s,
                gains={"p": args.wall_p_gain, "i": args.wall_i_gain, "d": args.wall_d_gain},
                motor_gains=args.wall_motor_gains,
            )
            feedback_controller.start()
            base = {k: v if v is not None else default for k, v, default in (
                ("p", args.wall_p_gain, leader_config.position_p_gain),
                ("i", args.wall_i_gain, leader_config.position_i_gain),
                ("d", args.wall_d_gain, leader_config.position_d_gain))}
            print("Wall gains P {p} I {i} D {d}".format(**base) + "".join(
                "; motor {} P {p} I {i} D {d}".format(m, **{**base, **g})
                for m, g in sorted((args.wall_motor_gains or {}).items())))

            def send_feedback(leader_angles, leader_velocities, found, dt):
                feedback_controller.update(contact.engaged, contact.walls)

        follower.set_current_limit(int(round(current_ma)))
        follower.set_motion_profile(
            profile_velocity_rad_s=args.max_speed_rad_s,
            profile_acceleration_rad_s2=args.max_acceleration_rad_s2,
        )
        start = follower.clip_positions(follower.read_pos())
        follower.set_positions(start, clip=True)
        follower.enable_torque()
        print(f"{FOLLOWING_MESSAGE} over {args.ramp_s:g} s, then following. Ctrl-C stops.", flush=True)
        cycles = teleop_loop(
            leader, follower, start,
            rate_hz=args.rate_hz, ramp_s=args.ramp_s, max_jump_rad=args.max_jump_rad,
            print_period_s=0.5 if args.print else None,
            contact=contact, send_feedback=send_feedback,
            current_signs=follower.config.joint_signs_array, leader_current_signs=leader.config.joint_signs_array,
            log=log,
            monitor=None if plot is None else (
                lambda t, current, error, feedback: plot.push(
                    t, np.degrees(error) if args.plot == "error" else current, feedback,
                    None if contact is None else contact.engaged)),
        )
        print(f"Stopped after {cycles} cycles.")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        # Each step runs whatever the one before did (an error, a second Ctrl-C), and the log is saved last.
        failed = []
        for name, part, method in (("follower", follower, "shutdown"), ("leader wall", feedback_controller, "shutdown"),
                                   ("leader", leader, "shutdown"), ("plot", plot, "close")):
            if part is None:
                continue
            try:
                getattr(part, method)()
            except BaseException as exc:
                failed.append(name)
                _say(f"Warning: {name} shutdown: {exc!r}", file=sys.stderr)
        if leader is not None or follower is not None:
            _say("Both hands' torque is off." if not {"follower", "leader"} & set(failed)
                 else f"Check the hands: {', '.join(failed)} shutdown failed.")
        if log:
            for attempt in range(3):  # a Ctrl-C while saving tries again
                try:
                    save_log(args.log, log, motor_ids, rate_hz=args.rate_hz, current_limit_ma=current_ma,
                             force_feedback=args.force_feedback, ramp_s=args.ramp_s)
                    _say(f"Saved {len(log['t'])} cycles to {args.log.resolve()}")
                    break
                except KeyboardInterrupt:
                    _say("Still saving the log...", file=sys.stderr)
    return 0


def _say(*args, **kwargs) -> None:
    """print, but never fail on a terminal that has gone away."""

    try:
        print(*args, **kwargs)
    except OSError:
        pass


def _stop_on_signals() -> None:
    """SIGTERM and SIGHUP (the terminal closed) raise KeyboardInterrupt, as Ctrl-C does."""

    import signal
    import threading

    if threading.current_thread() is not threading.main_thread():
        return

    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop)


def _current_or_off(text: str) -> float | None:
    if str(text).lower() in ("off", "none", "null", "0"):
        return None
    return float(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     epilog=__doc__.split("\n\n", 1)[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tuning", type=Path,
                        help=f"tuning YAML with defaults for the options below (default: {DEFAULT_TUNING})")
    parser.add_argument("--no-tuning", action="store_true", help="ignore the tuning YAML")
    parser.add_argument("--save-tuning", type=Path, metavar="FILE",
                        help="write the settings this command line amounts to as a tuning YAML and exit")
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
    for gain in ("p", "i", "d"):
        parser.add_argument(f"--follower-{gain}-gain", type=int,
                            help=f"follower position {gain.upper()} gain (default: the follower calibration's)")
    parser.add_argument("--max-jump-rad", type=float, default=DEFAULT_MAX_JUMP_RAD,
                        help="stop if a leader reading changes more than this in one cycle (default: %(default)g)")
    parser.add_argument("--force-feedback", action="store_true",
                        help="push back on leader joints whose follower joint is in contact")
    parser.add_argument("--feedback-motors", type=int, nargs="+",
                        help="motor IDs whose leader joint gets a wall (default: all active)")
    parser.add_argument("--contact-motors", type=int, nargs="+",
                        help="motor IDs whose follower joint is watched for contact, among the wall motors; "
                             "with --wall-whole-finger a contact still walls the whole finger "
                             "(default: all wall motors)")
    parser.add_argument("--grasp-only", nargs="*", metavar="MOTOR=SIGN",
                        help="count only the grasping direction: the finger PIP/MCP (flexion negative), "
                             "plus any MOTOR=+1/-1 given (e.g. thumb joints)")
    parser.add_argument("--contact-error-rad", type=float, default=DEFAULT_CONTACT_ERROR_RAD,
                        help="leader past the follower by more than this (motor rad) is contact "
                             "(default: %(default)g)")
    parser.add_argument("--contact-current-ma", type=_current_or_off, default=DEFAULT_CONTACT_CURRENT_MA,
                        help="also require the follower to push towards its goal with more than this (mA, below "
                             "--current-limit-ma); 'off' or 0 for positions only (default: off)")
    parser.add_argument("--contact-still-rad-s", type=float, default=CONTACT_STILL_RAD_S,
                        help="the follower counts as held back below this speed (default: %(default)g)")
    parser.add_argument("--contact-hold-ms", type=float, default=1000 * CONTACT_PERSISTENCE_S,
                        help="how long the contact conditions must hold (default: %(default)g)")
    parser.add_argument("--contact-motor-overrides", nargs="+", metavar="MOTOR:error_rad=E,hold_ms=H",
                        help="contact error and hold time for single motors, e.g. 0:error_rad=0.06 for a joint "
                             "that lags behind (default: none)")
    parser.add_argument("--stop-margin-rad", type=float, default=STOP_MARGIN_RAD,
                        help="no feedback within this of the joint's homing stop (default: %(default)g)")
    parser.add_argument("--wall-current-ma", type=float, default=DEFAULT_WALL_CURRENT_MA,
                        help="the wall's holding current on the leader (default: %(default)g; XL330 limit 1750)")
    parser.add_argument("--wall-whole-finger", action=argparse.BooleanOptionalAction, default=True,
                        help="a wall on any joint of a finger holds all that finger's joints, and the hand "
                             "opening at any of them lets them all go; --no-wall-whole-finger walls each "
                             "joint on its own")
    parser.add_argument("--wall-follow", action="store_true",
                        help="wall: move with the follower while in contact instead of staying where contact began")
    parser.add_argument("--wall-settle-ms", type=float, default=DEFAULT_WALL_SETTLE_MS,
                        help="after contact the wall holds this long, whatever the hand does, so the leader is "
                             "brought back to the follower (default: %(default)g)")
    parser.add_argument("--wall-release-rad", type=float, default=DEFAULT_WALL_RELEASE_RAD,
                        help="wall: let go once the leader is this far out of it (default: %(default)g)")
    parser.add_argument("--wall-release-ma", type=float, default=DEFAULT_WALL_RELEASE_MA,
                        help="wall: let go once its motor pulls the finger in this hard (default: %(default)g)")
    parser.add_argument("--wall-pass-rad", type=float, default=DEFAULT_WALL_PASS_RAD,
                        help="wall: let go once the follower moves on this far past it, nothing being there "
                             "(default: %(default)g)")
    parser.add_argument("--wall-speed-rad-s", type=float, default=DEFAULT_WALL_SPEED_RAD_S,
                        help="wall: how fast the leader is brought back to it (default: %(default)g)")
    for gain in ("p", "i", "d"):
        parser.add_argument(f"--wall-{gain}-gain", type=int,
                            help=f"wall: leader position {gain.upper()} gain (default: the leader calibration's)")
    parser.add_argument("--wall-motor-gains", nargs="+", metavar="MOTOR:p=P,i=I,d=D",
                        help="wall gains for single motors over --wall-p/i/d-gain, e.g. 0:p=600,i=0 for a "
                             "softer wall on a high-ratio joint (default: none)")
    parser.add_argument("--log", type=Path, help="save every cycle to this .npz")
    parser.add_argument("--plot", nargs="?", const="error", choices=("error", "current"),
                        help="live plot per joint: 'error' (follower minus leader, deg; the default) or "
                             "'current' (follower mA)")
    parser.add_argument("--plot-window-s", type=float, default=10.0,
                        help="seconds of history in the plot (default: %(default)g)")
    parser.add_argument("--print", action="store_true",
                        help="print leader angles, follower error and contacts twice a second")
    return parser


def load_tuning(path: Path, parser: argparse.ArgumentParser) -> dict:
    """Option defaults from a tuning YAML (keys: long option names, '-' or '_')."""

    import yaml

    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected a mapping of option names to values")
    known = {action.dest for action in parser._actions} - {"help", "tuning", "no_tuning", "save_tuning"}
    values, unknown = {}, []
    for key, value in data.items():
        name = str(key).replace("-", "_")
        if name not in known:
            unknown.append(str(key))
            continue
        if name in ("wall_motor_gains", "contact_motor_overrides") and isinstance(value, Mapping):  # {0: {p: 600}}
            value = [f"{m}:" + ",".join(f"{k}={v}" for k, v in (gains or {}).items())
                     for m, gains in value.items()]
        if name == "grasp_only":  # true / {} = the finger defaults, a mapping or MOTOR=SIGN list adds joints
            if value is True:
                value = []
            elif value is False or value is None:
                value = None
            elif isinstance(value, Mapping):
                value = [f"{int(m)}={int(d)}" for m, d in value.items()]
            else:
                value = [str(item) for item in value]
        values[name] = value
    if unknown:
        raise ValueError(f"{path}: unknown settings {unknown}; known: {sorted(known)}")
    return values


def save_tuning(path: Path, args: argparse.Namespace) -> None:
    import yaml

    values = {}
    for key in TUNING_KEYS:
        value = getattr(args, key)
        if isinstance(value, Path):
            value = str(value)
        if key == "grasp_only" and value is not None:
            value = {int(m): int(d) for m, d in value.items()}
        if key in ("wall_motor_gains", "contact_motor_overrides") and value is not None:
            value = {int(m): dict(g) for m, g in value.items()}
        values[key] = value
    header = ("# Tuning for scripts/teleop/trimi_teleop.py (--tuning FILE). Keys are the long option names;\n"
              "# command-line flags override them. Angles are motor radians.\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    early, _ = parser.parse_known_args(argv)
    if not early.no_tuning:
        path = (early.tuning or DEFAULT_TUNING).expanduser()
        if path.is_file():
            try:
                parser.set_defaults(**load_tuning(path, parser))
            except (ValueError, TypeError) as exc:
                parser.error(str(exc))
            parser.set_defaults(tuning=path)
        elif early.tuning is not None:
            parser.error(f"Tuning file not found: {path}")
    args = parser.parse_args(argv)
    if args.no_tuning:
        args.tuning = None
    if args.plot not in (None, "error", "current"):
        parser.error(f"plot must be error or current, not {args.plot!r}")
    for name in ("rate_hz", "current_limit_ma", "max_speed_rad_s", "max_acceleration_rad_s2", "max_jump_rad",
                 "wall_current_ma", "plot_window_s", "contact_still_rad_s", "contact_hold_ms", "wall_release_rad",
                 "wall_pass_rad",
                 "wall_release_ma", "wall_speed_rad_s"):
        value = getattr(args, name)
        if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive, not {value!r}")
    if args.contact_current_ma == 0:
        args.contact_current_ma = None
    if args.contact_current_ma is not None and not (
            isinstance(args.contact_current_ma, (int, float)) and math.isfinite(args.contact_current_ma)
            and args.contact_current_ma > 0):
        parser.error(f"--contact-current-ma must be positive or off, not {args.contact_current_ma!r}")
    for name in ("follower_p_gain", "follower_i_gain", "follower_d_gain", "wall_p_gain", "wall_i_gain",
                 "wall_d_gain"):
        value = getattr(args, name)
        if value is not None and not (isinstance(value, int) and 0 <= value <= 16383):
            parser.error(f"--{name.replace('_', '-')} must be an integer from 0 to 16383, not {value!r}")
    if not (isinstance(args.contact_error_rad, (int, float)) and math.isfinite(args.contact_error_rad)
            and args.contact_error_rad > 0):
        parser.error("--contact-error-rad must be finite and positive")
    if not (isinstance(args.wall_settle_ms, (int, float)) and math.isfinite(args.wall_settle_ms)
            and args.wall_settle_ms >= 0):
        parser.error("--wall-settle-ms must be finite and not negative")
    if not (math.isfinite(args.stop_margin_rad) and args.stop_margin_rad >= 0):
        parser.error("--stop-margin-rad must be finite and not negative")
    if not (math.isfinite(args.ramp_s) and args.ramp_s >= 0):
        parser.error("--ramp-s must be finite and not negative")
    if args.grasp_only is not None:
        pairs = {}
        for item in args.grasp_only:
            try:
                motor, sign = str(item).split("=")
                pairs[int(motor)] = int(float(sign))
            except ValueError:
                parser.error(f"--grasp-only takes MOTOR=+1 or MOTOR=-1, not {item!r}")
            if pairs[int(motor)] not in (-1, 1):
                parser.error(f"--grasp-only takes MOTOR=+1 or MOTOR=-1, not {item!r}")
        args.grasp_only = pairs
    per_motor = (
        ("wall_motor_gains", "MOTOR:p=P,i=I,d=D (gains 0-16383)",
         {k: (int, lambda v: 0 <= v <= 16383) for k in ("p", "i", "d")}),
        ("contact_motor_overrides", "MOTOR:error_rad=E,hold_ms=H (E positive, H not negative)",
         {"error_rad": (float, lambda v: math.isfinite(v) and v > 0),
          "hold_ms": (float, lambda v: math.isfinite(v) and v >= 0)}),
    )
    for name, usage, keys in per_motor:
        items = getattr(args, name)
        if items is None:
            continue
        settings: dict[int, dict] = {}
        for item in items:
            try:
                motor, _, rest = str(item).partition(":")
                entry = settings.setdefault(int(motor), {})
                for pair in filter(None, rest.split(",")):
                    key, text = pair.split("=")
                    cast, valid = keys[key.strip().lower()]
                    value = cast(text)
                    if not valid(value):
                        raise ValueError
                    entry[key.strip().lower()] = value
            except (ValueError, KeyError):
                parser.error(f"--{name.replace('_', '-')} takes {usage}, not {item!r}")
        setattr(args, name, settings)
    args.leader_config = Path(args.leader_config).expanduser()
    args.follower_config = Path(args.follower_config).expanduser()
    if args.log is not None:
        args.log = Path(args.log).expanduser()
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.save_tuning is not None:
        save_tuning(args.save_tuning.expanduser(), args)
        print(f"Saved the settings to {args.save_tuning}")
        return 0
    if args.tuning is not None:
        print(f"Tuning: {args.tuning}")
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
