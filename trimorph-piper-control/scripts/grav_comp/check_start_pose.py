#!/usr/bin/env python3
"""Offline gravity demand at a named start; run with the midas gravity-comp Python."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

CONTROL_ROOT = Path(__file__).resolve().parents[2]
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))
from data_collection.collection.config import DEFAULT_GRAV_COMP_CALIB, PIPER_START_JOINTS_DEG


def find_midas_repo():
    for path in (Path.home() / "midas_hand/midas-piper-control", Path.home() / "workspace/midas-piper-control"):
        if (path / "gravity_comp").is_dir():
            return path
    raise FileNotFoundError("No midas-piper-control checkout; pass --midas-repo")


def check_start(calib, name, midas_repo, max_torque_nm=6.5, *, joints_deg=None):
    if not np.isfinite(max_torque_nm) or max_torque_nm <= 0:
        raise ValueError("maximum torque must be finite and positive")
    sys.path.insert(0, str(Path(midas_repo).resolve()))
    from gravity_comp.calib import CalibrationArtifact
    from gravity_comp.model import ArmGravityModel

    artifact = CalibrationArtifact.load(calib)
    artifact.require("firmware", "gravity", "delivery", "payload")
    mass, com = artifact.payload_mass_com()
    model = ArmGravityModel(gravity=artifact.gravity_vector(), payload_mass=mass, payload_com=com)
    target = np.asarray(PIPER_START_JOINTS_DEG[name] if joints_deg is None else joints_deg, dtype=float)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError("start joints must contain six finite degree values")
    q = np.radians(target)
    gravity = np.asarray(model.gravity_torques(q))
    alpha, scales = np.asarray(artifact.alpha()), np.asarray(artifact.output_scales())
    if not np.isfinite(alpha).all() or np.any(alpha <= 0) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Invalid calibrated delivery/output scales")
    command = gravity / alpha  # same convention as gravity_comp.drag_mode
    safe_cap = .9 * np.asarray(artifact.torque_limits()) / scales
    limits = np.asarray(model.joint_limits())
    within_limits = bool(np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1]))
    ok = bool(within_limits and np.isfinite(command).all()
              and np.all(np.abs(command[:2]) <= max_torque_nm)
              and np.all(np.abs(command) <= safe_cap))
    return dict(ok=ok, start_home=name, joints_deg=target.tolist(),
                gravity_nm=gravity.tolist(), command_nm=command.tolist(), wire_nm=(command * scales).tolist(),
                safe_cap_nm=safe_cap.tolist(), max_j1_j2_nm=max_torque_nm, within_joint_limits=within_limits,
                calibration_path=str(Path(calib).resolve()),
                calibration_sha256=hashlib.sha256(Path(calib).read_bytes()).hexdigest())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calib", type=Path, default=DEFAULT_GRAV_COMP_CALIB)
    p.add_argument("--midas-repo", type=Path)
    start = p.add_mutually_exclusive_group(required=True)
    start.add_argument("--start-home", choices=sorted(PIPER_START_JOINTS_DEG), help="explicit legacy named start")
    start.add_argument("--start-profile", type=Path, help="saved per-task start JSON")
    p.add_argument("--max-j1-j2-nm", type=float, default=6.5)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    name, target = args.start_home, None
    if args.start_profile is not None:
        from data_collection.collection.task_start import load_task_start, task_start_sha256
        profile = load_task_start(args.start_profile)
        name, target = profile["task"], profile["arm"]["joints_deg"]
    result = check_start(args.calib, name, args.midas_repo or find_midas_repo(), args.max_j1_j2_nm,
                         joints_deg=target)
    if args.start_profile is not None:
        result.update(task=name, start_profile_path=str(args.start_profile.resolve()),
                      start_profile_sha256=task_start_sha256(args.start_profile))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{'PASS' if result['ok'] else 'FAIL'} {name}: commanded gravity torque J1..J6 (N*m): "
              + " ".join(f"{x:+.3f}" for x in result["command_nm"]))
        print(f"J1/J2 limit: {args.max_j1_j2_nm:g} N*m; checks this pose only, not the approach or drag workspace.")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
