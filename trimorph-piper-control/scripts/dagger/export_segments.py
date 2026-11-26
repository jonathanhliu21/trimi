#!/usr/bin/env python3
"""Select and postprocess successful DAgger rollouts containing human corrections.

  python scripts/dagger/export_dataset.py ROOT --out ROOT_human
  python scripts/dagger/export_dataset.py ROOT --out ROOT_human --trim

--out preserves the source and refuses nonempty destinations. --overwrite-raw
DELETES rejected episodes and replaces raw/processed data after validation.
By default videos and syncs cover the FULL retained rollout, and every frame is
labeled policy, human or handoff (syncs `control`). Every frame trains except a
kinesthetic takeover's handoff, which `train_mask` keeps out of training windows
unless --train-handoff. --trim exports each human correction as its own compact
episode. Stop collection/training before in-place export.
Arm poses are FK of the joints on their CAN receive clock (--pose-source piper_fk).
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_collection import postprocess_dataset as post
from scripts.dataset import postprocess_dataset as base

# The teleop hand ramp when the rollout's settings don't record it
# (--takeover-hand-ramp-s): the follower blends from the policy's last target to
# the leader over it. --trim drops it from each correction's start by default.
DEFAULT_TRIM_MS = {"teleop": 400.0, "kinesthetic": 0.0}
OUTPUT_DIRS = ("camera/mp4_files", "processed_encoder_t", "absolute_action_t", "syncs")
MANAGED = ("raw", "processed", *OUTPUT_DIRS, "segments.json")
CONTROL_LABELS = ("policy", "human", "handoff")
HAND_FIELDS = ("monotonic_ns", *(f"pos_{i}" for i in base.CANONICAL_TRIMORPH_MOTOR_IDS))


def write_csv(path, rows, fields=None):
    if not rows and fields is None:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hand_ramp_ms(metadata):
    """The recorded teleop hand ramp after the takeover, in ms; 0 for kinesthetic."""
    dagger = metadata["dagger"]
    ramp_s = (dagger.get("takeover_settings") or {}).get("hand_ramp_s")
    if dagger.get("takeover") == "teleop" and isinstance(ramp_s, (int, float)) and math.isfinite(ramp_s):
        return 1000.0 * max(0.0, float(ramp_s))
    return DEFAULT_TRIM_MS.get(dagger.get("takeover"), 0.0)


def piper_pose_source(raw):
    """Arm poses as data collection makes them: FK of the joints on their CAN receive clock.

    Rollouts recorded before the executor logged joint_rx_monotonic_ns have FK in
    x..yaw stamped at the poll time; those keep --pose-source piper.
    """
    with (raw / "piper.csv").open(newline="") as stream:
        header = next(csv.reader(stream), [])
    if "joint_rx_monotonic_ns" in header:
        return "piper_fk"
    print(f"Warning: {raw.name} predates joint_rx_monotonic_ns; using its poll-time FK poses (pose source piper).")
    return "piper"


def teleop_hand_labels(raw, metadata):
    """Teleop: label the hand with the follower's measured angles until the hand ramp ends.

    A teleop rollout's leader_hand.csv is the worn leader throughout. Until the
    takeover the policy drove the follower and the operator's hand was idle; during
    the hand ramp after it the follower still blends toward the leader. In the STAGED
    copy, leader_hand.csv becomes follower.csv's angles up to the ramp's end, so the
    label follows the robot's hand without a jump at the takeover, and the leader's
    from then on. The original stream is kept as leader.csv. Kinesthetic rollouts
    already label with the follower; exported rollouts were relabeled when exported.
    """
    takeover = metadata["dagger"].get("takeover_ns")
    if ((metadata.get("hand") or {}).get("label_source") != "leader" or takeover is None
            or metadata.get("export_provenance")):
        return
    switch = int(takeover) + round(hand_ramp_ms(metadata) * 1e6)
    leader = base.read_csv_rows(raw / "leader_hand.csv")
    follower = base.read_csv_rows(raw / "follower.csv")
    rows = [{key: row[key] for key in HAND_FIELDS} for row in follower if int(row["monotonic_ns"]) < switch]
    rows += [{key: row[key] for key in HAND_FIELDS} for row in leader if int(row["monotonic_ns"]) >= switch]
    if not (raw / "leader.csv").exists():
        shutil.copy2(raw / "leader_hand.csv", raw / "leader.csv")
    write_csv(raw / "leader_hand.csv", rows, HAND_FIELDS)


def control_labels(camera_ns, dagger):
    """Who drove the robot at each camera row: policy, human or handoff.

    policy   before the takeover pedal was read.
    human    from the takeover to the end: control never returns to the policy.
             Teleop includes the hand ramp and clutch-released pauses; kinesthetic
             includes the freeze after the end pedal.
    handoff  a kinesthetic takeover until "active": the arm is held at the takeover
             joints by a hold that releases over --kinesthetic-ramp-in-s and the hand
             is limp, so neither the policy nor the operator drives the robot.
    """
    labels = np.full(len(camera_ns), "policy", dtype=f"<U{max(map(len, CONTROL_LABELS))}")
    intervals = dagger.get("human_intervals") or []
    active = min((start for start, _ in intervals), default=None)
    takeover = dagger.get("takeover_ns") or active
    if takeover is None:
        return labels
    labels[camera_ns >= takeover] = "human"
    if dagger.get("takeover") == "kinesthetic" and active is not None:
        labels[(camera_ns >= takeover) & (camera_ns < active)] = "handoff"
    return labels


def select_episodes(root):
    selected, skipped = [], []
    for raw in sorted((root / "raw").glob("episode_*")):
        if not raw.is_dir() or not raw.name.removeprefix("episode_").isdigit():
            raise ValueError(f"Unexpected raw entry: {raw}")
        metadata = json.loads((raw / "metadata.json").read_text())
        dagger = metadata.get("dagger") or {}
        index = int(raw.name.removeprefix("episode_"))
        reason = None
        if metadata.get("status") != "complete":
            reason = f"status {metadata.get('status')}"
        elif dagger.get("outcome") != "success":
            reason = f"outcome {dagger.get('outcome')}"
        elif not dagger.get("human_intervals"):
            reason = "no human correction"
        if reason:
            skipped.append((index, reason))
            continue
        previous_end = -1
        for interval in dagger["human_intervals"]:
            if (len(interval) != 2 or any(type(t) is not int for t in interval)
                    or interval[0] < 0 or interval[1] <= interval[0] or interval[0] < previous_end):
                raise ValueError(f"Invalid/overlapping human intervals: {raw}")
            previous_end = interval[1]
        selected.append((raw, index, metadata))
    return selected, skipped


def source_state(root):
    """Detect writes/appends during export; never replace an actively changing dataset."""
    state = []
    for path in sorted((root / "raw").rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Raw symlinks are not supported for destructive-safe staging: {path}")
        stat = path.stat()
        state.append((str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns))
    return state


def trim_video(source, target, first, last, expected_count):
    """Decode in order, without approximate keyframe seeking; validate output decode."""
    cv2 = base.import_cv2()
    cap, writer = cv2.VideoCapture(str(source)), None
    count = written = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Unreadable video/FPS: {source}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if first <= count <= last:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    if not writer.isOpened():
                        raise ValueError(f"Cannot create video: {target}")
                writer.write(frame)
                written += 1
            count += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if count != expected_count or written != last - first + 1:
        raise ValueError(f"Video/timestamp frame count mismatch: decoded {count}, expected {expected_count}: {source}")
    cap, decoded = cv2.VideoCapture(str(target)), 0
    try:
        while cap.read()[0]:
            decoded += 1
    finally:
        cap.release()
    if decoded != written:
        raise ValueError(f"Trimmed video decode count mismatch: {target}")


def validate_alignment(sync, camera_ns, hand_ns, pose_ns, tolerance_s):
    n = len(camera_ns)
    for key, limit in (("frame_idx", n), ("encoder_idx", len(hand_ns)), ("pose_idx", len(pose_ns))):
        idx = sync[key]
        if (idx.ndim != 1 or len(idx) != n or not n or not np.issubdtype(idx.dtype, np.integer)
                or idx.min() < 0 or idx.max() >= limit or np.any(np.diff(idx) < 0)):
            raise ValueError(f"Invalid compact sync indices: {key}")
    if not np.array_equal(sync["frame_idx"], np.arange(n)):
        raise ValueError("Compact frame indices must start at zero without gaps")
    for name, stamps in (("camera", camera_ns), ("hand", hand_ns), ("pose", pose_ns)):
        if not len(stamps) or np.any(np.diff(stamps) <= 0):
            raise ValueError(f"Non-increasing {name} timestamps")
    errors = {}
    for key, stamps in (("encoder_idx", hand_ns), ("pose_idx", pose_ns)):
        delta = np.abs(stamps[sync[key]] - camera_ns)
        errors[key] = float(delta.max() / 1e6)
        if np.any(delta > round(tolerance_s * 1e9)):
            bad = int(np.count_nonzero(delta > round(tolerance_s * 1e9)))
            print(f"Warning: alignment exceeds {tolerance_s * 1000:g} ms: {key}, "
                  f"{bad}/{n} frames, worst {errors[key]:.3f} ms; keeping best-effort sync.")
    return errors


def compact_interval(work, episode, metadata, number, interval, out, output_index, trim_ms, min_rows, tolerance_s,
                     pose_source="piper_fk"):
    source = base.processed_paths(work, episode.index)
    with np.load(source.sync_path, allow_pickle=False) as archive:
        original = dict(archive)
    camera_rows = base.read_csv_rows(episode.camera_timestamps_path)
    camera_ns = base.int_column(camera_rows, "monotonic_ns", episode.camera_timestamps_path)
    frame_ids = base.int_column(camera_rows, "frame_idx", episode.camera_timestamps_path)
    if not np.array_equal(frame_ids, np.arange(len(frame_ids))):
        raise ValueError(f"Camera frame indices must be contiguous from zero: {episode.path}")
    times = camera_ns[original["frame_idx"]]
    start, end = interval
    chosen = np.flatnonzero((times >= start + round(trim_ms * 1e6)) & (times <= end))
    if len(chosen) < min_rows:
        return None, f"interval {number}: {len(chosen)} rows < {min_rows}"
    frames = original["frame_idx"][chosen]
    if np.any(np.diff(frames) != 1):
        raise ValueError("Correction frames are not contiguous")
    hi, pi = original["encoder_idx"][chosen], original["pose_idx"][chosen]
    h0, h1, p0, p1 = int(hi.min()), int(hi.max())+1, int(pi.min()), int(pi.max())+1
    hand = np.load(source.hand_path, allow_pickle=False)[h0:h1]
    poses = np.load(source.eef_path, allow_pickle=False)[p0:p1]
    hand_ns, pose_ns = original["encoder_monotonic_ns"][h0:h1], original["pose_monotonic_ns"][p0:p1]
    sync = dict(frame_idx=np.arange(len(frames), dtype=np.int64), encoder_idx=hi-h0, pose_idx=pi-p0,
                encoder_monotonic_ns=hand_ns, pose_monotonic_ns=pose_ns, monotonic_ns=times[chosen],
                start_pose=poses[0].astype(np.float64), pose_frame=original["pose_frame"], source=np.array("dagger_human"))
    errors = validate_alignment(sync, times[chosen], hand_ns, pose_ns, tolerance_s)
    dagger = metadata["dagger"]
    record = dict(episode=output_index, rollout=episode.index, interval=number, takeover=dagger.get("takeover"),
        round=dagger.get("round"), outcome="success", policy_sha256=(dagger.get("policy") or {}).get("sha256"),
        interval_ns=[start, end], trim_ms=trim_ms, rows=len(frames), original_frame_range=[int(frames[0]), int(frames[-1])],
        duration_s=float((times[chosen[-1]]-times[chosen[0]])/1e9), max_sync_error_ms=errors)
    record["trimmed"] = True
    target = base.processed_paths(out, output_index)
    raw = out / "raw" / f"episode_{output_index:06d}"
    raw.mkdir(parents=True)
    trim_video(episode.camera_path, target.video_path, int(frames[0]), int(frames[-1]), len(camera_rows))
    os.link(target.video_path, raw / "camera.mp4")  # one physical copy, no source link
    sync["provenance"] = np.array(json.dumps(record))
    np.save(target.hand_path, hand)
    np.save(target.eef_path, poses)
    np.savez_compressed(target.sync_path, **sync)
    write_csv(raw / "camera_timestamps.csv", [dict(camera_rows[int(f)], frame_idx=i) for i, f in enumerate(frames)])
    write_csv(raw / "leader_hand.csv", base.read_csv_rows(episode.leader_hand_path)[h0:h1])
    piper_rows = base.read_csv_rows(episode.piper_path)[p0:p1]
    write_csv(raw / "piper.csv", piper_rows)
    # The converted poses (FK for piper_fk), one row per Piper sample.
    replay_rows = base.read_csv_rows(post.processed_episode_dir(episode.path) / "piper.csv")[p0:p1]
    counts = {"camera_frames": len(frames), "leader_hand": len(hand), "piper": len(poses)}
    low, high = int(times[chosen[0]]), int(times[chosen[-1]])
    for name in ("follower", "leader", "commands", "events", "quest", "gravity_comp"):
        path = episode.path / f"{name}.csv"
        if path.exists():
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = [r for r in reader if r.get("monotonic_ns") and low <= int(r["monotonic_ns"]) <= high]
            write_csv(raw / path.name, rows, fields)
            counts[name] = len(rows)
    compact_meta = copy.deepcopy(metadata)
    compact_meta.update(episode_index=output_index, duration_s=record["duration_s"], counts=counts, export_provenance=record)
    compact_meta["piper"] = {**compact_meta.get("piper", {}), "start_pose": [float(replay_rows[0][k]) for k in base.RAW_EEF_FIELDS]}
    compact_meta["dagger"].update(human_intervals=[[low, high]], takeover_ns=low,
                                segments=[dict(source="human", start_ns=low, end_ns=high)])
    (raw / "metadata.json").write_text(json.dumps(compact_meta, indent=2)+"\n")
    processed = post.processed_episode_dir(raw)
    processed.mkdir(parents=True, exist_ok=True)
    replay = processed / "piper.csv"
    write_csv(replay, replay_rows)
    (processed / "metadata.json").write_text(json.dumps(dict(pose_source=pose_source, pose_frame="piper_base", compact_export=record), indent=2)+"\n")
    post.validate_outputs(target, replay, False)
    return record, None


def full_rollout(work, episode, metadata, out, output_index, min_rows, tolerance_s, train_handoff=False):
    """Keep all frames/samples; label each frame policy, human or handoff.

    Every frame trains except a kinesthetic handoff, which syncs `train_mask` keeps
    out of training windows unless train_handoff.
    """
    source = base.processed_paths(work, episode.index)
    with np.load(source.sync_path, allow_pickle=False) as archive:
        sync = dict(archive)
    camera_rows = base.read_csv_rows(episode.camera_timestamps_path)
    camera_ns = base.int_column(camera_rows, "monotonic_ns", episode.camera_timestamps_path)
    frame_ids = base.int_column(camera_rows, "frame_idx", episode.camera_timestamps_path)
    if not np.array_equal(frame_ids, np.arange(len(camera_rows))):
        raise ValueError("Full rollout camera indices must be contiguous from zero")
    errors = validate_alignment(sync, camera_ns, sync["encoder_monotonic_ns"],
                                sync["pose_monotonic_ns"], tolerance_s)
    labels = control_labels(camera_ns, metadata["dagger"])
    if not any(np.count_nonzero((labels == "human") & (camera_ns >= start) & (camera_ns <= end)) >= min_rows
               for start, end in metadata["dagger"]["human_intervals"]):
        return None, f"no human interval has at least {min_rows} rows"
    # Validate actual decoded length without re-encoding or modifying the full video.
    cv2 = base.import_cv2()
    cap, count = cv2.VideoCapture(str(source.video_path)), 0
    try:
        while cap.read()[0]:
            count += 1
    finally:
        cap.release()
    if count != len(camera_rows):
        raise ValueError(f"Video/timestamp frame count mismatch: {episode.path}")
    dagger = metadata["dagger"]
    record = dict(episode=output_index, rollout=episode.index, trimmed=False, outcome="success",
                  human_intervals=dagger["human_intervals"], rows=count,
                  control_rows={label: int(np.count_nonzero(labels == label)) for label in CONTROL_LABELS},
                  train_handoff=train_handoff, takeover=dagger.get("takeover"), round=dagger.get("round"),
                  policy_sha256=(dagger.get("policy") or {}).get("sha256"),
                  duration_s=float((camera_ns[-1] - camera_ns[0]) / 1e9), max_sync_error_ms=errors)
    target = base.processed_paths(out, output_index)
    raw = out / "raw" / f"episode_{output_index:06d}"
    # These are staged copies, so moving them never changes the original dataset.
    os.replace(episode.path, raw)
    os.link(source.video_path, target.video_path)
    os.replace(source.hand_path, target.hand_path)
    os.replace(source.eef_path, target.eef_path)
    processed = post.processed_episode_dir(raw)
    processed.parent.mkdir(parents=True, exist_ok=True)
    os.replace(post.processed_episode_dir(episode.path), processed)
    sync.update(source=np.array("dagger_rollout"), provenance=np.array(json.dumps(record)))
    # Every frame stays in the sync arrays. The trainer skips windows that touch a
    # train_mask=False frame: the kinesthetic handoff, unless train_handoff.
    sync["control"] = labels
    sync["train_mask"] = np.ones(len(labels), dtype=bool) if train_handoff else labels != "handoff"
    sync["intervention"] = labels == "human"
    np.savez_compressed(target.sync_path, **sync)
    saved = copy.deepcopy(metadata)
    saved.update(episode_index=output_index, export_provenance=record)
    (raw / "metadata.json").write_text(json.dumps(saved, indent=2) + "\n")
    post.validate_outputs(target, post.processed_episode_dir(raw) / "piper.csv", False)
    return record, None


def publish_in_place(root, result):
    """Rollback rename failures. A failed rollback leaves the backup for recovery.

    Not power-loss atomic across directories. No source deletion before validation.
    """
    backup = Path(tempfile.mkdtemp(prefix=f".{root.name}-export-backup-", dir=root.parent))
    moved, installed = [], []
    names = [*MANAGED, *(p.name for p in root.glob("midas_*.zarr.zip*"))]
    try:
        for name in names:
            dest = root / name
            if dest.exists() or dest.is_symlink():
                saved = backup / name
                saved.parent.mkdir(parents=True, exist_ok=True)
                os.replace(dest, saved)
                moved.append(name)
        for name in MANAGED:
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(result / name, dest)
            installed.append(name)
    except BaseException:
        for name in reversed(installed):
            os.replace(root / name, result / name)
        for name in reversed(moved):
            os.replace(backup / name, root / name)
        shutil.rmtree(backup)  # rollback completed; no original files remain here
        raise
    else:
        shutil.rmtree(backup)
    finally:
        if backup.exists() and not any(backup.iterdir()):
            backup.rmdir()


def export(root, out=None, *, overwrite_raw=False, trim=False, trim_ms=None, min_rows=32, max_sync_error_s=0.01,
           train_handoff=False):
    root = Path(root).expanduser().resolve()
    if (out is None and not overwrite_raw) or (out is not None and overwrite_raw):
        raise ValueError("Specify exactly one of --out or --overwrite-raw")
    if min_rows < 1 or not math.isfinite(max_sync_error_s) or max_sync_error_s <= 0:
        raise ValueError("min_rows and max_sync_error_s must be positive")
    if trim_ms is not None and (not math.isfinite(trim_ms) or trim_ms < 0):
        raise ValueError("trim_ms must be finite and nonnegative")
    if (trim_ms is not None and not trim) or (train_handoff and trim):
        raise ValueError("trim_ms applies only with trim, train_handoff only without it")
    if not (root / "raw").is_dir():
        raise FileNotFoundError(f"No raw directory: {root}")
    for name in MANAGED:
        for path in (root / name, (root / name).parent):
            if path.is_symlink():
                raise ValueError(f"Managed dataset paths must not be symlinks: {path}")
    out = root if overwrite_raw else Path(out).expanduser().resolve()
    if not overwrite_raw:
        if out == root or out in root.parents or root in out.parents:
            raise ValueError("--out must be separate from the source, not an ancestor or descendant")
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise FileExistsError(f"Destination must be empty; no files will be deleted: {out}")
    selected, skipped = select_episodes(root)
    if not selected:
        raise ValueError("No complete successful rollouts with human corrections; nothing changed")
    original_state = source_state(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    segments = []
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-export-", dir=out.parent) as tmp:
        scratch = Path(tmp)
        result, work = scratch / "result", scratch / "work"
        base.ensure_processed_dirs(result)
        (result / "raw").mkdir()
        for raw, index, metadata in selected:
            # Copy selected raw episodes BEFORE postprocessing, one at a time.
            staged = work / "raw" / raw.name
            shutil.copytree(raw, staged, ignore=shutil.ignore_patterns("processed"))
            teleop_hand_labels(staged, metadata)
            pose_source = piper_pose_source(staged)
            config = post.PostprocessConfig(dataset_root=work, raw_dir=work / "raw",
                overwrite_processed_files=False, process_only_raw_statuses={"complete"},
                max_sync_error_s=max_sync_error_s, trim_camera_frames=0, pose_source=pose_source, link_video=True)
            episode = base.discover_raw_episodes(config)[0]
            post.process_episode(config, episode)
            already_trimmed = metadata.get("export_provenance", {}).get("trimmed", bool(metadata.get("export_provenance")))
            trim_start = (0.0 if already_trimmed else hand_ramp_ms(metadata)) if trim_ms is None else trim_ms
            intervals = list(enumerate(metadata["dagger"]["human_intervals"])) if trim else [(None, None)]
            for number, interval in intervals:
                if trim:
                    record, reason = compact_interval(work, episode, metadata, number, interval, result,
                        len(segments), trim_start, min_rows, max_sync_error_s, pose_source)
                else:
                    record, reason = full_rollout(work, episode, metadata, result,
                        len(segments), min_rows, max_sync_error_s, train_handoff)
                if record is None:
                    skipped.append((index, reason))
                else:
                    segments.append(record)
            shutil.rmtree(work)
        if not segments:
            raise ValueError("No corrections survived minimum length; nothing changed")
        summary = dict(format_version=4, compact=trim, trimmed=trim, dagger_root=str(root), outcomes=["success"],
            control_labels=None if trim else list(CONTROL_LABELS), train_handoff=None if trim else train_handoff,
            min_rows=min_rows, max_sync_error_s=max_sync_error_s, segments=segments,
            skipped=[dict(rollout=i, reason=r) for i, r in skipped])
        (result / "segments.json").write_text(json.dumps(summary, indent=2)+"\n")
        if source_state(root) != original_state:
            raise RuntimeError("Source raw data changed during export; stop collection first. Nothing replaced.")
        if overwrite_raw:
            publish_in_place(root, result)
        else:
            # POSIX rename also replaces an empty destination directory atomically.
            # If someone populated it meanwhile, rename refuses instead of deleting files.
            os.replace(result, out)
    return segments, skipped


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", type=Path, help="Raw DAgger dataset")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path, help="New/empty output directory; preserve every source file")
    mode.add_argument("--overwrite-raw", action="store_true", help="DESTRUCTIVE: replace raw/processed data with selected exports; delete rejected episodes")
    parser.add_argument("--trim", action="store_true", help="Export only human correction video/syncs; default keeps the full qualifying rollout")
    parser.add_argument("--trim-start-ms", type=float, default=None,
                        help="With --trim: drop this much from each correction's start. Default: the rollout's "
                             "teleop hand ramp, 0 for kinesthetic")
    parser.add_argument("--train-handoff", action="store_true",
                        help="Full rollouts: also train on a kinesthetic takeover's handoff (arm held by the "
                             "releasing hold, hand limp); by default the trainer skips windows that touch it")
    parser.add_argument("--min-rows", type=int, default=32)
    parser.add_argument("--max-sync-error-ms", type=float, default=10.0, help="Warn on camera/sample timing differences above this threshold; keep nearest samples (default 10 ms)")
    args = parser.parse_args(argv)
    if args.min_rows < 1 or not math.isfinite(args.max_sync_error_ms) or args.max_sync_error_ms <= 0:
        parser.error("--min-rows and --max-sync-error-ms must be positive")
    if args.trim_start_ms is not None and (not math.isfinite(args.trim_start_ms) or args.trim_start_ms < 0):
        parser.error("--trim-start-ms must be finite and >= 0")
    if args.trim_start_ms is not None and not args.trim:
        parser.error("--trim-start-ms applies only with --trim")
    if args.train_handoff and args.trim:
        parser.error("--train-handoff applies only to full rollouts (without --trim)")
    return args


def main(argv=None):
    args = parse_args(argv)
    segments, skipped = export(args.dataset_root, args.out, overwrite_raw=args.overwrite_raw, trim=args.trim,
        trim_ms=args.trim_start_ms, min_rows=args.min_rows, max_sync_error_s=args.max_sync_error_ms/1000,
        train_handoff=args.train_handoff)
    print(f"Exported {len(segments)} {'corrections' if args.trim else 'full rollouts'} to {args.dataset_root if args.overwrite_raw else args.out}")
    for index, reason in skipped:
        print(f"  skipped rollout {index}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
