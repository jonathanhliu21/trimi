#!/usr/bin/env python3
"""Clearance of the torque-stage probe poses with the spring-held joints drifted to the abort limit.

In verify_delivery (S3, S4b, S4c) and measure_friction (S5) only the probe joint is free; the
other joints hold on a soft spring (kp 3 N*m/rad) plus gravity feed-forward, and a trial aborts
once one drifts past HELD_DQ_ABORT_RAD (0.08 rad). grav_comp_clearance.py replays these poses
with the held joints exact. This checks every corner of the +-drift box on the held joints
(clipped to the joint limits) with the probe joint at -catch, 0 and +catch radius, and prints
the worst clearances (mm) next to the undrifted ones. Braking past the abort is not included.

    python3 scripts/grav_comp/held_joint_drift.py --midas-repo ~/workspace/midas-piper-control
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grav_comp import grav_comp_clearance as C  # noqa: E402

KEYS = ("desk", "other", "self", "face", "person")


def values(c: C.Clear) -> dict[str, float]:
    return dict(desk=c.env["desk"], other=c.other()[0], self=c.self_, face=c.face, person=c.person)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--midas-repo", type=Path, default=Path("~/workspace/midas-piper-control"),
                   help="midas-piper-control checkout with gravity_comp (norman/piper-midas-gravity-comp)")
    p.add_argument("--base-above-desk", type=float, default=0.49, help="base origin above the desk top (m)")
    p.add_argument("--desk-height", type=float, default=C.DESK_HEIGHT, help="desk top above the floor (m)")
    p.add_argument("--hand-turn-deg", type=float, default=0.0,
                   help="extra turn of the hand about the flange axis vs the sim mount (CAD's own: +57.5)")
    p.add_argument("--drift", type=float, default=None,
                   help="held-joint drift (rad, default: each script's HELD_DQ_ABORT_RAD)")
    p.add_argument("--viscous", action="store_true", help="S5 as measure_friction --viscous runs it")
    args = p.parse_args(argv)

    gp, vd, mf, gw, gm = C.load_gravity_comp(args.midas_repo.expanduser())
    scene = C.build_scene(args.desk_height, args.base_above_desk, args.hand_turn_deg)
    check = C.Checker(scene)
    lo, hi = np.asarray(gp.JOINT_LIMITS).T
    stages = C.stage_items(gp, vd, mf, np.zeros(6), vd.parse_args(["--j6"]).catch_rad, mf.CATCH_RAD, args.viscous)

    print("worst clearance (mm) over the probe poses, held joints exact -> drifted to every corner of "
          "+-drift, probe joint at -catch/0/+catch")
    print(f"{'stage':20s} {'drift':>5s} " + " ".join(f"{k:>14s}" for k in KEYS) + "   worst drifted pose")
    for name, items in stages.items():
        probes = [it for it in items if it.kind == "probe"]
        if not probes:
            continue
        drift = args.drift if args.drift is not None else (
            mf.HELD_DQ_ABORT_RAD if name.startswith("S5") else vd.HELD_DQ_ABORT_RAD)
        exact = {k: (np.inf, "") for k in KEYS}
        drifted = {k: (np.inf, "") for k in KEYS}
        for it in probes:
            held = [j for j in range(6) if j != it.probe]
            for swing in (-it.sag, 0.0, it.sag):
                base = it.q.copy()
                base[it.probe] += swing
                for k, v in values(check(base)).items():
                    exact[k] = min(exact[k], (v, it.label))
                for signs in itertools.product((-1.0, 1.0), repeat=len(held)):
                    q = base.copy()
                    q[held] += drift * np.asarray(signs)
                    for k, v in values(check(np.clip(q, lo, hi))).items():
                        drifted[k] = min(drifted[k], (v, it.label))
        print(f"{name:20s} {drift:5.2f} "
              + " ".join(f"{C.mm(exact[k][0]):>6s} -> {C.mm(drifted[k][0]):>4s}" for k in KEYS)
              + "   " + ", ".join(f"{k} @ {drifted[k][1]}" for k in ("desk", "other", "face")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
