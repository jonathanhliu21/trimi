#!/usr/bin/env python3
"""How clearly the follower's motor current shows contact, from trimi_teleop.py --log runs.

    python3 scripts/teleop/analyze_teleop_log.py --free free.npz --contact contact.npz

free.npz: move the leader through each joint's range, slow and fast, with
nothing touching the follower. contact.npz: close follower fingers on a rigid
object and keep pushing the leader in (the follower is blocked), lightly and
firmly, letting go in between.

The free run gives the current each follower joint needs just to move
(gearbox and linkage friction, gear meshing, the fingers' own weight): raw,
and what is left after a fitted motion model

    I = c0 + c1 v + c2 tanh(v / 0.05) + c3 a + c4 cos q + c5 sin q

The 99th percentile of that residual is the floor below which a contact can't
be told from motion. In the contact run, samples where the follower is still
(|v| < 0.1 rad/s) but more than 0.05 rad short of its goal are contacts; the
script reports their current above the motion model, how it grows with the
error (mA per rad of the leader pushing past the contact), and the error at
which it clears the floor. Readings are in motor units (mA, motor radians).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

V_SMOOTH_RAD_S = 0.05
STILL_RAD_S = 0.1
CONTACT_ERROR_RAD = 0.05
SETTLE_S = 0.5  # after the start-up ramp
FLOOR_PERCENTILE = 99.0


@dataclass(frozen=True)
class Run:
    t: np.ndarray
    goal: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    current: np.ndarray
    motor_ids: tuple[int, ...]


@dataclass(frozen=True)
class JointReport:
    motor_id: int
    free_raw_p99: float
    floor: float  # residual percentile in free motion, mA
    contacts: int = 0
    contact_residual: float = math.nan  # median, signed along the error
    ma_per_rad: float = math.nan
    clear_error_rad: float = math.nan


def load_run(path: Path) -> Run:
    data = np.load(path)
    t = np.asarray(data["t"], dtype=np.float64)
    keep = t >= float(data["ramp_s"]) + SETTLE_S if "ramp_s" in data else np.ones(len(t), bool)
    if keep.sum() < 10:
        raise ValueError(f"{path}: fewer than 10 cycles after the start-up ramp")
    return Run(t[keep], *(np.asarray(data[k], dtype=np.float64)[keep]
                          for k in ("goal", "position", "velocity", "current")),
               tuple(int(m) for m in data["motor_ids"]))


def motion_features(t: np.ndarray, q: np.ndarray, v: np.ndarray) -> np.ndarray:
    a = np.gradient(v, t) if len(t) > 2 else np.zeros_like(v)
    return np.column_stack([np.ones_like(v), v, np.tanh(v / V_SMOOTH_RAD_S), a, np.cos(q), np.sin(q)])


def analyze(free: Run, contact: Run | None = None) -> list[JointReport]:
    if contact is not None and contact.motor_ids != free.motor_ids:
        raise ValueError("The two runs have different motors")
    reports = []
    for j, motor_id in enumerate(free.motor_ids):
        x = motion_features(free.t, free.position[:, j], free.velocity[:, j])
        y = free.current[:, j]
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        floor = float(np.percentile(np.abs(y - x @ beta), FLOOR_PERCENTILE))
        report = dict(motor_id=motor_id, free_raw_p99=float(np.percentile(np.abs(y), FLOOR_PERCENTILE)),
                      floor=floor)
        if contact is not None:
            error = contact.goal[:, j] - contact.position[:, j]
            residual = contact.current[:, j] - motion_features(
                contact.t, contact.position[:, j], contact.velocity[:, j]) @ beta
            mask = (np.abs(contact.velocity[:, j]) < STILL_RAD_S) & (np.abs(error) > CONTACT_ERROR_RAD)
            report["contacts"] = int(mask.sum())
            if mask.sum() >= 5:
                along = residual[mask] * np.sign(error[mask])  # current pushing towards the goal
                depth = np.abs(error[mask])
                report["contact_residual"] = float(np.median(along))
                if np.ptp(depth) > 1e-3:
                    slope, intercept = np.polyfit(depth, along, 1)
                    report["ma_per_rad"] = float(slope)
                    if slope > 0:
                        report["clear_error_rad"] = float(max((floor - intercept) / slope, 0.0))
        reports.append(JointReport(**report))
    return reports


def format_reports(reports: list[JointReport]) -> str:
    rows = [f"{'motor':>5} {'free |I| p99':>12} {'floor':>7} {'contacts':>8} {'contact I':>9} "
            f"{'mA/rad':>8} {'clears floor at':>15}"]
    for r in reports:
        clear = "" if math.isnan(r.clear_error_rad) else f"{math.degrees(r.clear_error_rad):6.1f} deg"
        contact = "" if math.isnan(r.contact_residual) else f"{r.contact_residual:7.0f}mA"
        slope = "" if math.isnan(r.ma_per_rad) else f"{r.ma_per_rad:8.0f}"
        rows.append(f"{r.motor_id:>5} {r.free_raw_p99:10.0f}mA {r.floor:5.0f}mA {r.contacts:>8} {contact:>9} "
                    f"{slope:>8} {clear:>15}")
    return "\n".join(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     epilog=__doc__.split("\n\n", 1)[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--free", type=Path, required=True, help="log with free motion only")
    parser.add_argument("--contact", type=Path, help="log with the follower pushing on an object")
    args = parser.parse_args(argv)
    free = load_run(args.free)
    contact = load_run(args.contact) if args.contact else None
    print(f"Free run: {len(free.t)} cycles" + (f", contact run: {len(contact.t)} cycles" if contact else ""))
    print(format_reports(analyze(free, contact)))
    print(f"\nfloor = {FLOOR_PERCENTILE:g}th percentile of the free current left after the motion model. "
          "'clears floor at' is how far past first contact the leader must push before the current alone "
          f"shows it; compare with the position detector's {math.degrees(CONTACT_ERROR_RAD):.1f} deg.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
