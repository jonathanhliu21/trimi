#!/usr/bin/env python3
"""Compare command and feedback responses in two saved drag-mode NPZ logs.

This is an offline diagnostic. Current-versus-position fits describe an
association in the recorded samples; they do not establish the control mode,
prove physical position holding, or replace a supported hardware check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REQUIRED = ("t", "q", "tau_cmd", "tau_fb", "state")


def _number(value: float) -> float | None:
    return round(float(value), 6) if np.isfinite(value) else None


def _means(values: np.ndarray) -> list[float | None]:
    return [_number(column[np.isfinite(column)].mean())
            if np.isfinite(column).any() else None for column in values.T]


def _span(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return _number(np.ptp(finite)) if finite.size else None


def summarize(path: Path) -> dict:
    """Read one log without hardware dependencies or pickle-enabled arrays."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = set(REQUIRED) - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing arrays: {', '.join(sorted(missing))}")
        arrays = {key: np.asarray(data[key]) for key in REQUIRED}
    t = arrays["t"]
    if t.ndim != 1 or not t.size:
        raise ValueError(f"{path}: t must be a nonempty one-dimensional array")
    if not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError(f"{path}: timestamps must be finite and strictly increasing")
    for key in ("q", "tau_cmd", "tau_fb"):
        if arrays[key].shape != (len(t), 6):
            raise ValueError(f"{path}: {key} must have shape ({len(t)}, 6)")
    if arrays["state"].shape != t.shape:
        raise ValueError(f"{path}: state must have shape ({len(t)},)")
    if arrays["state"].dtype.kind not in "US":
        raise ValueError(f"{path}: state must contain string labels")
    states = np.char.upper(arrays["state"].astype(str))
    elapsed = t - t[0]
    q, command, feedback = (arrays[key].astype(float)
                            for key in ("q", "tau_cmd", "tau_fb"))
    windows = {}
    for label, start, end in (("first_0_to_0_5_s", 0.0, 0.5),
                              ("later_2_to_3_s", 2.0, 3.0)):
        selected = (elapsed >= start) & (elapsed < end) & (states == "RAMP_IN")
        windows[label] = {
            "ramp_samples": int(selected.sum()),
            "mean_command_nm": _means(command[selected]),
            "mean_feedback_nm": _means(feedback[selected]),
        }
    active = states == "ACTIVE"
    joints = []
    for j in range(6):
        valid = active & np.isfinite(q[:, j]) & np.isfinite(feedback[:, j])
        angle, torque = q[valid, j], feedback[valid, j]
        slope = correlation = None
        if len(angle) >= 3 and np.ptp(angle) > 1e-9:
            centered_q = angle - angle.mean()
            centered_torque = torque - torque.mean()
            slope = _number(np.dot(centered_q, centered_torque)
                            / np.dot(centered_q, centered_q))
            if np.ptp(torque) > 1e-9:
                correlation = _number(np.corrcoef(angle, torque)[0, 1])
        joints.append({
            "joint": j + 1,
            "active_command_range_nm": _span(command[active, j]),
            "active_feedback_range_nm": _span(feedback[active, j]),
            "active_angle_range_deg": _span(np.rad2deg(q[active, j])),
            "fit_samples": int(valid.sum()),
            "feedback_vs_angle_slope_nm_per_rad": slope,
            "feedback_vs_angle_correlation": correlation,
        })
    unique, counts = np.unique(states, return_counts=True)
    return {
        "path": str(path.resolve()),
        "samples": len(t),
        "duration_s": _number(elapsed[-1]),
        "state_samples": dict(zip(unique.tolist(), counts.tolist())),
        "nonfinite_values": {
            key: int((~np.isfinite(arrays[key])).sum())
            for key in ("q", "tau_cmd", "tau_fb")
        },
        "ramp_response": windows,
        "active_joint_diagnostics": joints,
    }


def compare(reference: Path, retry: Path) -> dict:
    return {
        "interpretation": (
            "Offline observations only. A current-versus-position association "
            "does not prove physical holding or its cause. Logged commands "
            "are software values; these logs do not verify transmitted CAN "
            "bytes, accepted gains, or drive enables. Slopes use all finite "
            "ACTIVE angle/feedback pairs, including changing commands."
        ),
        "reference": summarize(reference),
        "retry": summarize(retry),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path, help="original drag-mode NPZ log")
    parser.add_argument("retry", type=Path, help="retry drag-mode NPZ log")
    parser.add_argument("--output", type=Path, help="also save the JSON report here")
    args = parser.parse_args(argv)
    try:
        report = compare(args.reference, args.retry)
        rendered = json.dumps(report, allow_nan=False, separators=(",", ":"))
        if args.output:
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, TypeError) as exc:
        print(f"Cannot compare drag response: {exc}", file=sys.stderr)
        return 1
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
