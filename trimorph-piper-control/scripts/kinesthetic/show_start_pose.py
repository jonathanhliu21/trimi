#!/usr/bin/env python3
"""Show live Piper joints against the named start, using receive-only CAN.

Works with the arm disabled. This viewer never enables drives, selects modes,
or sends motion; closing its window only stops the viewer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_collection.collection.config import DEFAULT_GRAV_COMP_CALIB, PIPER_START_JOINTS_DEG
from data_collection.collection.piper_worker import arm_health_problem
from data_collection.collection.shared_state import ArmHealth, SharedState
from data_collection.collection.task_start import load_task_start
from data_collection.collection.tracking_monitor import TrackingMonitor
from scripts.piper_api.piper_api import ReceiveOnlyPiperAPI


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--firmware", default="v183")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--start-home", choices=sorted(PIPER_START_JOINTS_DEG))
    selection.add_argument("--start-profile", type=Path, help="this task's recorded start")
    selection.add_argument("--define-start", action="store_true", help="show the current arm without a fixed target (default)")
    parser.add_argument("--calib", type=Path, default=DEFAULT_GRAV_COMP_CALIB)
    parser.add_argument("--render", type=Path, help="save one rendered snapshot and exit")
    args = parser.parse_args(argv)
    calibration = json.loads(args.calib.read_text())
    gravity = calibration.get("gravity", {}).get("vector_m_s2")
    profile = None if args.start_profile is None else load_task_start(args.start_profile)
    start_name = profile["task"] if profile is not None else args.start_home
    definition = start_name is None
    shared = SharedState()
    monitor = TrackingMonitor(
        shared, show_quest=False, show_vicon=False, show_arm=True,
        start_name=start_name,
        start_joints_deg=None if profile is None else profile["arm"]["joints_deg"],
        definition_mode=definition, passive_j6=True, gravity_vector=gravity,
        window_name="Piper current pose + start (READ ONLY)", size=(1120, 820),
    )
    cv2 = monitor.cv2
    wall_to_monotonic_ns = time.monotonic_ns() - round(time.time() * 1e9)
    print("READ ONLY: current arm is solid; a loaded task start is dashed. Drag to rotate, scroll to zoom.", flush=True)
    print("No drive or motion commands. q/Escape closes only this viewer.", flush=True)
    try:
        with ReceiveOnlyPiperAPI(firmware=args.firmware, channel=args.channel) as api:
            if not api.is_ok():
                raise RuntimeError(f"No Piper feedback on {args.channel}")
            while True:
                joints, status = api.get_joint_state(), api.get_arm_status()
                enabled = api.get_driver_enable_status()
                problem = arm_health_problem(enabled, None if status is None else status.err_code)
                if joints is not None:
                    stamp_ns = round(joints.timestamp * 1e9) + wall_to_monotonic_ns
                    shared.set_arm_health(ArmHealth(
                        stamp_ns, not problem, problem, joints.hz, tuple(joints.joints)))
                if args.render is not None:
                    args.render.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(args.render), monitor.render()):
                        raise RuntimeError(f"Could not save {args.render}")
                    print(f"Saved {args.render}")
                    return 0
                monitor.update()
                if cv2.waitKey(33) & 0xFF in (ord("q"), 27):
                    return 0
                if cv2.getWindowProperty(monitor.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    return 0
    finally:
        monitor.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
