"""Validated, portable task-specific arm and hand start snapshots.

This module only reads saved data. It never connects to either device and never
clips measured positions to configured limits.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


MAX_SYNC_NS = 50_000_000
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


def _number(value, name):
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite number")
    return value


def _positive_ns(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer timestamp")
    return value


def _sha256(value, name):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
    return value


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def validate_task_start(profile) -> dict:
    """Return an independent JSON snapshot after validating the version-1 schema.

    Hand positions follow ``hand.motor_ids``; callers must map by motor ID rather
    than assuming an input file's list order. Captures write IDs in order 0..12.
    Warnings do not change the saved pose or authorize replay of it.
    """
    _object(profile, "task start")
    if type(profile.get("schema_version")) is not int or profile["schema_version"] != 1:
        raise ValueError("task start schema_version must be 1")
    task = profile.get("task")
    if not isinstance(task, str) or not task.strip() or task != task.strip():
        raise ValueError("task must be a nonblank name without surrounding whitespace")
    try:
        captured = datetime.fromisoformat(profile["captured_utc"].replace("Z", "+00:00"))
        if captured.utcoffset() is None or captured.utcoffset().total_seconds() != 0:
            raise ValueError("not UTC")
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError("captured_utc must be an ISO-8601 UTC timestamp") from exc
    arm = _object(profile.get("arm"), "arm")
    joints = arm.get("joints_deg")
    if not isinstance(joints, list) or len(joints) != 6:
        raise ValueError("arm.joints_deg must contain six joint angles")
    for i, value in enumerate(joints, 1):
        _number(value, f"arm.joints_deg J{i}")
    hand = _object(profile.get("hand"), "hand")
    ids, positions = hand.get("motor_ids"), hand.get("positions_rad")
    if (not isinstance(ids, list) or len(ids) != 13
            or any(type(i) is not int for i in ids) or set(ids) != set(range(13))):
        raise ValueError("hand.motor_ids must contain each motor ID 0..12 exactly once")
    if not isinstance(positions, list) or len(positions) != 13:
        raise ValueError("hand.positions_rad must contain 13 positions aligned with motor_ids")
    for motor, value in zip(ids, positions):
        _number(value, f"hand.positions_rad motor {motor}")
    _sha256(hand.get("config_sha256"), "hand.config_sha256")
    source = _object(profile.get("source"), "source")
    if not isinstance(source.get("episode_path"), str) or not source["episode_path"].strip():
        raise ValueError("source.episode_path must name the source episode")
    arm_ns = _positive_ns(source.get("arm_monotonic_ns"), "source.arm_monotonic_ns")
    hand_ns = _positive_ns(source.get("hand_monotonic_ns"), "source.hand_monotonic_ns")
    if abs(arm_ns - hand_ns) > MAX_SYNC_NS:
        raise ValueError("arm/hand synchronization exceeds 50 ms")
    if _number(source.get("at_s"), "source.at_s") < 0:
        raise ValueError("source.at_s must be nonnegative")
    if "calibration_sha256" in source:
        _sha256(source["calibration_sha256"], "source.calibration_sha256")
    warnings = profile.get("warnings")
    if not isinstance(warnings, list) or any(not isinstance(w, str) for w in warnings):
        raise ValueError("warnings must be a list of strings")
    try:
        return json.loads(json.dumps(profile, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("task start must contain finite JSON values") from exc


def load_task_start(path) -> dict:
    """Load and validate a fresh snapshot; changing it cannot mutate the file."""
    return validate_task_start(json.loads(Path(path).expanduser().read_text()))


def task_start_sha256(path) -> str:
    return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()


def write_task_start(path, profile, *, overwrite=False) -> Path:
    """Write a validated snapshot, refusing to replace any existing file by default."""
    snapshot = validate_task_start(profile)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also rejects a concurrent creator, not just files seen
    # during an earlier existence check.
    with path.open("w" if overwrite else "x") as stream:
        stream.write(json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def _read_csv(path, required):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if (reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or not set(required).issubset(reader.fieldnames)):
            raise ValueError(f"{path.name}: missing or duplicate required CSV columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name}: no samples")
    return rows


def _csv_number(row, name, file_name):
    try:
        result = float(row[name])
        if not math.isfinite(result):
            raise ValueError("nonfinite")
        return result
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{file_name}: missing or invalid {name}") from exc


def _csv_times(rows, field, file_name):
    try:
        stamps = [int(row[field]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{file_name}: missing or invalid {field}") from exc
    if any(ns <= 0 for ns in stamps) or any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError(f"{file_name}: {field} must be positive and strictly increasing")
    return stamps


def _arm_limit_warnings(joints_deg):
    try:
        # Constants only: importing this module does not open CAN or query limits.
        from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD
    except ImportError:
        return ["Arm SDK limit check unavailable; measured angles are preserved."]
    warnings = []
    for i, value in enumerate(joints_deg, 1):
        low, high = map(math.degrees, ROBOT_JOINT_LIMIT_PRESET_RAD["piper"][f"joint{i}"])
        margin = min(value - low, high - value)
        if margin < 0:
            warnings.append(f"Arm J{i} {value:.6f} deg is outside SDK limits {low:.6f}..{high:.6f} deg.")
        elif margin < 3:
            warnings.append(f"Arm J{i} is within {margin:.6f} deg of an SDK joint limit.")
    return warnings


def _hand_limit_warnings(positions, hand_meta):
    lower, upper = hand_meta.get("joint_lower_limits"), hand_meta.get("joint_upper_limits")
    if not isinstance(lower, dict) or not isinstance(upper, dict):
        return ["Hand YAML limit check unavailable in source metadata; measured positions are preserved."]
    warnings = []
    for motor, value in enumerate(positions):
        try:
            low, high = float(lower[str(motor)]), float(upper[str(motor)])
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError("invalid limits")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"source metadata has missing or invalid hand limits for motor {motor}") from exc
        if value < low or value > high:
            warnings.append(f"Hand motor {motor} {value:.9f} rad is outside recorded YAML limits {low:g}..{high:g} rad.")
    return warnings


def capture_task_start(task, episode_path, *, at_s=0.0) -> dict:
    """Capture the nearest actual CAN joint sample and nearest measured hand row.

    ``at_s`` is relative to the episode's ``started_monotonic_ns``. Require the
    actual CAN receipt clock, measured motor-ID columns, and at most 50 ms of
    arm/hand skew. No interpolation, angle wrapping or limit clipping is used.
    """
    _number(at_s, "at_s")
    if at_s < 0:
        raise ValueError("at_s must be nonnegative")
    episode = Path(episode_path).expanduser().resolve()
    metadata = _object(json.loads((episode / "metadata.json").read_text()), "episode metadata")
    start_ns = _positive_ns(metadata.get("started_monotonic_ns"), "started_monotonic_ns")
    piper = _object(metadata.get("piper"), "source piper metadata")
    trimorph = _object(metadata.get("trimorph"), "source trimorph metadata")
    hand_meta = _object(trimorph.get("hand"), "source hand metadata")
    _sha256(hand_meta.get("config_sha256"), "source hand config_sha256")
    ids = hand_meta.get("motor_ids", trimorph.get("motor_ids"))
    if (not isinstance(ids, list) or len(ids) != 13
            or any(type(i) is not int for i in ids) or set(ids) != set(range(13))):
        raise ValueError("source metadata must identify all 13 hand motor IDs 0..12")
    arm_fields = [f"joint_{j}" for j in range(1, 7)]
    hand_fields = [f"pos_{motor}" for motor in range(13)]
    arm_rows = _read_csv(episode / "piper.csv", ["joint_rx_monotonic_ns", *arm_fields])
    hand_rows = _read_csv(episode / "leader_hand.csv", ["monotonic_ns", *hand_fields])
    arm_times = _csv_times(arm_rows, "joint_rx_monotonic_ns", "piper.csv")
    hand_times = _csv_times(hand_rows, "monotonic_ns", "leader_hand.csv")
    if at_s > (arm_times[-1] - start_ns + MAX_SYNC_NS) / 1e9:
        raise ValueError("requested time is more than 50 ms from the nearest arm sample")
    target_ns = start_ns + round(at_s * 1e9)
    arm_index = min(range(len(arm_times)), key=lambda i: abs(arm_times[i] - target_ns))
    arm_ns = arm_times[arm_index]
    if abs(arm_ns - target_ns) > MAX_SYNC_NS:
        raise ValueError("requested time is more than 50 ms from the nearest arm sample")
    hand_index = min(range(len(hand_times)), key=lambda i: abs(hand_times[i] - arm_ns))
    hand_ns = hand_times[hand_index]
    if abs(hand_ns - arm_ns) > MAX_SYNC_NS:
        raise ValueError("nearest arm/hand samples differ by more than 50 ms")
    joints = [math.degrees(_csv_number(arm_rows[arm_index], name, "piper.csv")) for name in arm_fields]
    positions = [_csv_number(hand_rows[hand_index], name, "leader_hand.csv") for name in hand_fields]
    warnings = _arm_limit_warnings(joints) + _hand_limit_warnings(positions, hand_meta)
    source = dict(episode_path=str(episode), arm_monotonic_ns=arm_ns,
                  hand_monotonic_ns=hand_ns, at_s=float(at_s),
                  selected_at_s=(arm_ns - start_ns) / 1e9,
                  sync_error_ms=abs(arm_ns - hand_ns) / 1e6)
    gravity = _object(piper.get("gravity_comp") or {}, "source gravity-comp metadata")
    if gravity.get("calibration_sha256") is not None:
        source["calibration_sha256"] = _sha256(gravity["calibration_sha256"], "source calibration_sha256")
    else:
        warnings.append("Gravity-compensation calibration hash unavailable in source metadata.")
    if gravity.get("calibration_changed_during_episode"):
        warnings.append("Source calibration changed during the episode; its initial hash may not describe this sample.")
    profile = dict(schema_version=1, task=task,
                   captured_utc=datetime.now(timezone.utc).isoformat(),
                   arm=dict(joints_deg=joints),
                   hand=dict(motor_ids=list(range(13)), positions_rad=positions,
                             config_sha256=hand_meta["config_sha256"]),
                   source=source, warnings=warnings)
    return validate_task_start(profile)
