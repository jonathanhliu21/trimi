#!/usr/bin/env python3
"""Quality report for raw collection episodes, run before postprocessing.

    python3 scripts/dataset/check_episodes.py data_collection_0

For every raw/episode_XXXXXX it reports each stream's rate, gaps, and coverage
of the episode, then stream-specific checks: identical hand rows (a still hand
or a failed bus read; the API repeats the last good values for all 13 motors),
camera drops and capture-to-host latency, Quest frozen or jumping positions,
Vicon lost-object samples and occluded markers, and how far each camera frame
is from the nearest hand/Quest/Vicon sample. Episodes without metadata.json
(collection stopped before saving) are listed and skipped. Read-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

QUEST_JUMP_M = 0.05
QUEST_FROZEN_SPAN_M = 1e-4


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def stream_line(name: str, stamps: np.ndarray, start_ns: int, stop_ns: int) -> str:
    if len(stamps) < 2:
        return f"  {name:7s} {len(stamps)} samples"
    gaps = np.diff(stamps) / 1e6
    rate = 1e9 * (len(stamps) - 1) / max(int(stamps[-1]) - int(stamps[0]), 1)
    return (f"  {name:7s} n={len(stamps):6d}  {rate:6.1f} Hz  "
            f"gap ms median {np.median(gaps):5.1f} p95 {np.percentile(gaps, 95):5.1f} max {gaps.max():6.1f}  "
            f"covers {(stamps[0] - start_ns) / 1e6:+.0f} .. {(stamps[-1] - stop_ns) / 1e6:+.0f} ms of the episode"
            + ("  NON-INCREASING TIMESTAMPS" if np.any(gaps <= 0) else ""))


def nearest_ms(source: np.ndarray, query: np.ndarray) -> np.ndarray:
    index = np.clip(np.searchsorted(source, query), 1, len(source) - 1)
    return np.minimum(np.abs(query - source[index - 1]), np.abs(source[index] - query)) / 1e6


def longest_run(flags: np.ndarray) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def start_checks(joints_rad, guide, warnings):
    """Use the recorded criterion and target; leave joint-limit checks separate."""
    if not guide.get("guided_start") and guide.get("guided_start_joints_deg") is None:
        return
    name = guide.get("guided_start") or guide.get("task") or "saved task start"
    criterion = guide.get("start_criterion")
    try:
        explicit_target = guide.get("guided_start_joints_deg")
        if explicit_target is not None:
            target = np.asarray(explicit_target, dtype=float)
        elif criterion == "end_effector_pose":
            raise ValueError("end-effector criterion needs recorded target joints")
        else:
            from data_collection.collection.config import PIPER_START_JOINTS_DEG
            target = np.asarray(PIPER_START_JOINTS_DEG[name], dtype=float)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("recorded start target must contain six finite degree values")
        if criterion == "end_effector_pose":
            from data_collection.collection.arm_geometry import flange_pose_error
            position_tolerance = float(guide.get("start_position_tolerance_m", .10))
            orientation_tolerance = float(guide.get("start_orientation_tolerance_deg", 45.))
            if (not np.isfinite([position_tolerance, orientation_tolerance]).all()
                    or position_tolerance <= 0 or orientation_tolerance <= 0):
                raise ValueError("end-effector start tolerances must be finite and positive")
            distance, rotation = flange_pose_error(joints_rad, np.radians(target))
            print(f"  start   flange position error {distance * 100:.2f} cm "
                  f"(tolerance {position_tolerance * 100:g} cm), total orientation error "
                  f"{rotation:.2f} deg (tolerance {orientation_tolerance:g} deg)")
            if distance > position_tolerance + 1e-12:
                warnings.append(f"Piper start position differs from {name} by {distance * 100:.2f} cm "
                                f"(tolerance {position_tolerance * 100:g} cm)")
            if rotation > orientation_tolerance + 1e-9:
                warnings.append(f"Piper start total orientation differs from {name} by {rotation:.2f} deg "
                                f"(tolerance {orientation_tolerance:g} deg)")
        elif criterion is None:
            # Historical episodes keep their recorded joint-space criterion.
            error = np.degrees(joints_rad) - target
            tolerance = float(guide.get("start_tolerance_deg", 5))
            print("  start   joint errors deg: " + " ".join(f"{x:+.2f}" for x in error))
            if not np.isfinite(tolerance) or tolerance <= 0:
                warnings.append("Piper start-guide tolerance missing/invalid")
            elif np.any(np.abs(error) > tolerance):
                warnings.append(f"Piper start differs from {name} by up to "
                                f"{np.abs(error).max():.2f} deg (tolerance {tolerance:g} deg)")
        else:
            warnings.append(f"unsupported Piper start criterion: {criterion}")
    except (KeyError, TypeError, ValueError, ImportError) as exc:
        warnings.append(f"Piper start check unavailable: {exc}")


def piper_checks(rows, metadata, warnings):
    """Read-only checks; returns the stream clock for cross-stream sync checks."""
    kinesthetic = (metadata.get("piper") or {}).get("mode") == "gravity_comp_kinesthetic"
    if not rows:
        if kinesthetic:
            warnings.append("no Piper samples")
        return None
    clock = "joint_rx_monotonic_ns" if kinesthetic or all(row.get("joint_rx_monotonic_ns") for row in rows) else "monotonic_ns"
    try:
        stamps = np.array([int(row[clock]) for row in rows], dtype=np.int64)
    except (KeyError, ValueError, OverflowError):
        warnings.append(f"Piper missing or invalid {clock}")
        return None
    print(stream_line("piper", stamps, metadata["started_monotonic_ns"], metadata["stopped_monotonic_ns"]))
    print(f"  piper   clock={clock}")
    valid_clock = len(stamps) >= 2 and np.all(stamps > 0) and np.all(np.diff(stamps) > 0)
    if not valid_clock:
        warnings.append("Piper needs at least two positive, strictly increasing timestamps (duplicates/backwards invalid)")
    if valid_clock and kinesthetic:
        rate = 1e9 * (len(stamps) - 1) / (stamps[-1] - stamps[0])
        gap = np.diff(stamps) / 1e6
        if not 150 <= rate <= 250:
            warnings.append(f"Piper joint rate {rate:.1f} Hz outside 150..250 Hz")
        if gap.max() > 50:
            warnings.append(f"Piper maximum gap {gap.max():.1f} ms exceeds 50 ms")
    if kinesthetic:
        disabled = ((metadata.get("piper") or {}).get("gravity_comp") or {}).get("disabled_joints", [])
        if disabled not in ([], [6]):
            warnings.append(f"unsupported gravity-comp disabled joints: {disabled}")
        required = ("arm_ok", "err_code", "arm_status", "ctrl_mode", "mode_feedback", *[f"enable_{j}" for j in range(1, 7)])
        missing = sum(any(row.get(k, "") == "" for k in required) for row in rows)
        unhealthy = sum(row.get("arm_ok") != "1" or row.get("err_code") != "0"
                        or any(row.get(f"enable_{j}") != ("0" if j in disabled else "1") for j in range(1, 7))
                        or not row.get("mode_feedback", "").startswith(("MOVE_MIT", "MIT"))
                        or not row.get("arm_status", "").startswith("NORMAL") for row in rows)
        if missing:
            warnings.append(f"Piper status missing in {missing} rows")
        if unhealthy:
            warnings.append(f"Piper unhealthy or not in MIT mode in {unhealthy}/{len(rows)} rows")
        fields = ("ctrl_mode", "mode_feedback", "arm_status", "err_code", *[f"enable_{j}" for j in range(1, 7)])
        transitions = sum(any(a.get(k) != b.get(k) for k in fields) for a, b in zip(rows, rows[1:]))
        print(f"  piper   unhealthy={unhealthy}/{len(rows)}, status transitions={transitions}")
        if transitions:
            warnings.append(f"Piper arm status changed {transitions} times")
    joint_fields = [f"joint_{j}" for j in range(1, 7)]
    if kinesthetic or all(rows[0].get(k) for k in joint_fields):
        try:
            joints = np.array([[float(row[k]) for k in joint_fields] for row in rows])
            if not np.isfinite(joints).all():
                raise ValueError("nonfinite")
            guide = (metadata.get("piper") or {}).get("gravity_comp") or {}
            if kinesthetic:
                start_checks(joints[0], guide, warnings)
            from data_collection.replay_actions import piper_model
            _, bounds = piper_model()  # pure model; no CAN connection or limit queries
            margins = np.degrees(np.minimum(joints - bounds[:, 0], bounds[:, 1] - joints))
            print("  piper   minimum joint-limit margins deg: " + " ".join(f"{x:.2f}" for x in margins.min(axis=0)))
            for j in range(6):
                outside, near = int((margins[:, j] < 0).sum()), int(((margins[:, j] >= 0) & (margins[:, j] < 3)).sum())
                if outside or near:
                    warnings.append(f"Piper J{j+1}: {outside} outside SDK limits, {near} within 3 deg")
        except (KeyError, ValueError, ImportError) as exc:
            warnings.append(f"Piper joint-limit check unavailable: {exc}")
    if kinesthetic and valid_clock:
        dt = np.diff(stamps) / 1e9
        observed = (dt > 0) & (dt <= .05)  # never extend a sample across an outage
        for j in (1, 2):
            try:
                torque = np.array([float(row[f"torque_{j}"]) for row in rows])
                if not np.isfinite(torque).all():
                    raise ValueError("nonfinite torque")
            except (KeyError, ValueError):
                warnings.append(f"Piper J{j} torque unavailable/invalid")
                continue
            near_cap = np.abs(torque[:-1]) >= .95 * 7.2
            seconds = float(dt[observed & near_cap].sum())
            print(f"  piper   J{j} current-derived torque >= 6.84 Nm: {seconds:.3f}/{dt[observed].sum():.3f} observed s")
            if seconds:
                warnings.append(f"Piper J{j} torque near 7.2 Nm cap for {seconds:.3f}s (possible saturation/support, not proof)")
    return stamps if valid_clock else None


def hand_limit_checks(hand_rows, metadata, warnings):
    hand_meta = (metadata.get("trimorph") or {}).get("hand") or {}
    lower, upper = hand_meta.get("joint_lower_limits"), hand_meta.get("joint_upper_limits")
    kinesthetic = (metadata.get("piper") or {}).get("mode") == "gravity_comp_kinesthetic"
    if not lower or not upper:
        if kinesthetic:
            warnings.append("hand limit check unavailable: no recorded YAML limits")
        return
    motors = hand_meta.get("motor_ids", [])
    if kinesthetic and (len(motors) != 13 or set(motors) != set(range(13))):
        warnings.append("hand limit check incomplete: expected recorded motor IDs 0..12")
    for motor in motors:
        key = str(motor)
        try:
            values = np.array([float(row[f"pos_{motor}"]) for row in hand_rows])
            lo, hi = float(lower[key]), float(upper[key])
            if not np.isfinite([lo, hi]).all() or lo > hi:
                raise ValueError("invalid limits")
        except (KeyError, ValueError):
            warnings.append(f"hand motor {motor}: missing/invalid positions or recorded limits")
            continue
        below, above, invalid = int((values < lo).sum()), int((values > hi).sum()), int((~np.isfinite(values)).sum())
        if below or above or invalid:
            warnings.append(f"hand motor {motor}: {below} below/{above} above YAML limits, {invalid} invalid samples")


def check_episode(episode: Path, max_sync_error_ms: float) -> list[str]:
    """Print the report for one episode and return its warnings."""

    metadata = json.loads((episode / "metadata.json").read_text())
    start_ns, stop_ns = metadata["started_monotonic_ns"], metadata["stopped_monotonic_ns"]
    warnings: list[str] = []
    role = metadata.get("episode_role", ((metadata.get("piper") or {}).get("gravity_comp") or {}).get("episode_role"))
    if role == "start_definition":
        warnings.append("start definition recording: reference pose only, not a training episode")
    guide = ((metadata.get("piper") or {}).get("gravity_comp") or {})
    if role != "start_definition" and guide.get("start_guide_enabled") is False:
        warnings.append("no recorded task-start target: collection had no start guide")
    print(f"\n=== {episode.name}  status={metadata['status']}  {metadata['duration_s']:.1f} s  "
          f"stop={metadata['stop_reason']}")
    if metadata["status"] != "complete":
        warnings.append(f"status {metadata['status']}: postprocessing skips it")
    for error in metadata.get("errors", []):
        warnings.append(f"{error['worker']} worker error: {error['message']}")
    warnings.extend(f"episode flag: {flag}" for flag in metadata.get("flags", []))

    hand_rows = read_rows(episode / "leader_hand.csv")
    camera_rows = read_rows(episode / "camera_timestamps.csv")
    quest_rows = read_rows(episode / "quest.csv")
    pose_rows = [row for row in read_rows(episode / "mocap.csv") if row["message_type"].endswith("/Position")]
    marker_rows = [row for row in read_rows(episode / "mocap.csv") if row["message_type"].endswith("/MarkerPositions")]
    piper_ns = piper_checks(read_rows(episode / "piper.csv"), metadata, warnings)
    stamps = {name: np.array([int(row["monotonic_ns"]) for row in rows], dtype=np.int64)
              for name, rows in (("hand", hand_rows), ("camera", camera_rows), ("quest", quest_rows),
                                 ("vicon", pose_rows)) if rows}
    for name, values in stamps.items():
        print(stream_line(name, values, start_ns, stop_ns))
    if "hand" not in stamps:
        warnings.append("no leader-hand samples")
        return warnings
    hand_limit_checks(hand_rows, metadata, warnings)
    if piper_ns is not None:
        reference = stamps.get("camera", stamps["hand"])
        if "camera" in stamps and len(reference) > 2:
            reference = reference[1:-1]
        distance = nearest_ms(piper_ns, reference)
        late = int((distance > max_sync_error_ms).sum())
        ref_name = "camera" if "camera" in stamps else "hand"
        print(f"  sync    {ref_name}->piper nearest ms median {np.median(distance):.2f} "
              f"p95 {np.percentile(distance, 95):.2f} max {distance.max():.2f}; over {max_sync_error_ms:g} ms: {late}/{len(distance)}")
        if late:
            warnings.append(f"{late} {ref_name}->Piper sync distances exceed {max_sync_error_ms:g} ms")

    hand = np.array([[float(row[f"pos_{motor}"]) for motor in range(13)] for row in hand_rows])
    identical = np.all(np.diff(hand, axis=0) == 0, axis=1)
    run = longest_run(identical) + 1
    print(f"  hand    identical consecutive rows {int(identical.sum())}/{len(identical)}, longest run {run} "
          f"({run * 1000 / metadata['trimorph']['measure_rate_hz']:.0f} ms); joint ranges (rad) "
          + " ".join(f"{value:.2f}" for value in np.ptp(hand, axis=0)))
    if run * 1000 / metadata["trimorph"]["measure_rate_hz"] >= 500:
        warnings.append(f"hand identical for {run} samples: still hand or bus dropout (check the terminal log)")
    hand_rate = 1e9 * (len(stamps["hand"]) - 1) / max(stamps["hand"][-1] - stamps["hand"][0], 1)
    if hand_rate < 0.9 * metadata["trimorph"]["measure_rate_hz"]:
        warnings.append(f"hand rate {hand_rate:.1f} Hz below the {metadata['trimorph']['measure_rate_hz']:g} Hz target")

    if camera_rows:
        line = f"  camera  {len(camera_rows)} frames"
        if "sequence_num" in camera_rows[0]:
            sequence = np.array([int(row["sequence_num"]) for row in camera_rows])
            latency = np.array([int(row["receipt_monotonic_ns"]) - int(row["monotonic_ns"]) for row in camera_rows]) / 1e6
            dropped = int(sequence[-1] - sequence[0] + 1 - len(sequence))
            line += f", dropped {dropped}, capture-to-host ms median {np.median(latency):.1f} p95 {np.percentile(latency, 95):.1f}"
            if dropped:
                warnings.append(f"{dropped} camera frames dropped")
        else:
            line += " (receipt timestamps: recorded before capture-time stamping)"
        try:
            import cv2

            capture = cv2.VideoCapture(str(episode / "camera.mp4"))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            line += (f"; video {frames} frames {int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                     f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            capture.release()
            if frames != len(camera_rows):
                warnings.append(f"video has {frames} frames but camera_timestamps.csv has {len(camera_rows)}")
        except ImportError:
            pass
        print(line)

    if quest_rows:
        position = np.array([[float(row[key]) for key in ("px", "py", "pz")] for row in quest_rows])
        steps = np.linalg.norm(np.diff(position, axis=0), axis=1)
        sequence = np.array([int(row["seq"]) for row in quest_rows])
        jumps = int(np.sum(steps > QUEST_JUMP_M))
        print(f"  quest   position span {np.ptp(position, axis=0).max():.4f} u, path {steps.sum():.3f} u, "
              f"max step {steps.max(initial=0):.4f} u, jumps > {QUEST_JUMP_M} u: {jumps}, "
              f"seq gaps {int(np.sum(np.diff(sequence) - 1))}")
        if np.ptp(position, axis=0).max() < QUEST_FROZEN_SPAN_M:
            warnings.append("Quest position frozen for the whole episode: positional tracking lost")
        if jumps:
            warnings.append(f"{jumps} Quest position jumps over {QUEST_JUMP_M} u")

    if pose_rows:
        for topic in sorted({row["topic"] for row in pose_rows}):
            rows = [row for row in pose_rows if row["topic"] == topic]
            pose = np.array([[float(row[key]) for key in ("x_trans", "y_trans", "z_trans")] for row in rows])
            lost = np.abs(pose).sum(axis=1) == 0
            frames = np.array([int(row["frame_number"]) for row in rows])
            received = len(np.unique(frames)) / max(frames[-1] - frames[0] + 1, 1)
            valid = pose[~lost]
            path = np.linalg.norm(np.diff(valid, axis=0), axis=1).sum() if len(valid) > 1 else 0.0
            print(f"  vicon   {topic}: lost-object {int(lost.sum())}/{len(rows)} ({100 * lost.mean():.1f}%), "
                  f"Vicon frames received {100 * received:.0f}%, path {path:.0f} mm")
            if lost.mean() > 0.1:
                warnings.append(f"{topic} lost in {100 * lost.mean():.0f}% of samples")
        if marker_rows:
            occluded = sum(row["occluded"] in ("True", "true", "1") for row in marker_rows)
            print(f"  markers {len(marker_rows)} rows, occluded {occluded} ({100 * occluded / len(marker_rows):.1f}%)")

    if "camera" in stamps and len(stamps["camera"]) > 2:
        frames = stamps["camera"][1:-1]  # postprocessing trims one frame at each end
        sources = {"hand": stamps["hand"]}
        if "quest" in stamps:
            sources["quest"] = stamps["quest"]
        if pose_rows:
            sources["vicon"] = np.sort(np.array([int(row["monotonic_ns"]) for row in pose_rows
                                                 if any(float(row[k]) for k in ("x_trans", "y_trans", "z_trans"))]))
        for name, source in sources.items():
            if len(source) < 2:
                continue
            distance = nearest_ms(source, frames)
            late = int(np.sum(distance > max_sync_error_ms))
            print(f"  sync    camera->{name:5s} nearest ms median {np.median(distance):5.1f} "
                  f"p95 {np.percentile(distance, 95):6.1f} max {distance.max():7.1f}; over {max_sync_error_ms:g} ms: "
                  f"{late}/{len(distance)}")
            if late:
                warnings.append(f"{late} camera->{name} sync distances exceed {max_sync_error_ms:g} ms")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--max-sync-error-ms", type=float, default=10.0)
    args = parser.parse_args()
    episodes = sorted(path for path in (args.dataset_root / "raw").glob("episode_*") if path.is_dir())
    if not episodes:
        raise SystemExit(f"No episodes under {args.dataset_root / 'raw'}")
    summary = {}
    for episode in episodes:
        if not (episode / "metadata.json").exists():
            summary[episode.name] = ["no metadata.json: collection stopped before saving; delete it before postprocessing"]
            print(f"\n=== {episode.name}  NOT SAVED (files: {', '.join(sorted(p.name for p in episode.iterdir()))})")
            continue
        summary[episode.name] = check_episode(episode, args.max_sync_error_ms)
    print("\n=== Summary")
    for name, warnings in summary.items():
        print(f"  {name}: " + ("OK" if not warnings else "; ".join(warnings)))
    return 1 if any(summary.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
