#!/usr/bin/env python3
"""Keep the human-controlled part of HG-DAgger rollouts as training episodes.

    python3 data_collection/postprocess_dataset.py DAGGER_ROOT --pose-source piper
    python3 scripts/dagger/export_segments.py DAGGER_ROOT            # -> DAGGER_ROOT_human

DAGGER_ROOT is an executor --dagger dataset after postprocessing. Each
human-controlled interval the executor recorded (metadata.json
dagger.human_intervals: the teleop clutch engagements, or the kinesthetic
takeover from the end of the gravity ramp-in to the end pedal) becomes one
episode of the output root, in the layout MidasImageDataset reads:

- syncs/NNNNNN.npz: the rollout's sync rows whose camera frame falls in
  [interval start + trim, interval end], plus provenance (rollout, interval,
  round, takeover kind, policy checkpoint sha256, outcome).
- processed_encoder_t/, absolute_action_t/: the rollout's arrays, unchanged
  (the sync rows index into them).
- camera/mp4_files/: the rollout's video (copy, symlink or hard link); frame
  indices stay those of the whole rollout.

--trim-start-ms drops the start of every interval: the operator's reaction
and the hand ramp, rows that would teach the policy to wait in the failure
state (default 400 ms for teleop, 0 for kinesthetic, whose intervals already
start after the ramp-in). Intervals with fewer than --min-rows camera rows are
skipped. By default only rollouts that ended as a success are kept.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataset.postprocess_dataset import ensure_processed_dirs, processed_paths  # noqa: E402

DEFAULT_TRIM_MS = {"teleop": 400.0, "kinesthetic": 0.0}
OUTPUT_DIRS = ("camera/mp4_files", "processed_encoder_t", "absolute_action_t", "syncs")


def camera_times(raw_episode):
    """monotonic_ns of every camera frame, indexed by frame_idx."""
    with (raw_episode / "camera_timestamps.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    times = np.full(len(rows), -1, dtype=np.int64)
    for row in rows:
        times[int(row["frame_idx"])] = int(row["monotonic_ns"])
    return times


def place_video(source, target, mode):
    if mode == "symlink":
        os.symlink(source.resolve(), target)
    elif mode == "hardlink":
        os.link(source, target)
    else:
        shutil.copy2(source, target)


def prepare_output(out, force):
    existing = [out / name for name in (*OUTPUT_DIRS, "segments.json") if (out / name).exists()]
    if any(path.is_file() or any(path.iterdir()) for path in existing):
        if not force:
            raise FileExistsError(f"{out} already has exported segments; pass -f to replace them")
        for path in existing:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    ensure_processed_dirs(out)


def export(root, out, *, trim_ms=None, min_rows=32, outcomes=("success",), video="copy", force=False):
    """Write every kept interval of root's rollouts to out; returns the segment records."""
    root, out = Path(root), Path(out)
    if out.resolve() == root.resolve():
        raise ValueError("The output root must differ from the DAgger dataset root")
    prepare_output(out, force)
    segments, skipped = [], []
    for metadata_path in sorted((root / "raw").glob("episode_*/metadata.json")):
        raw_episode = metadata_path.parent
        metadata = json.loads(metadata_path.read_text())
        dagger = metadata.get("dagger")
        index = int(raw_episode.name.removeprefix("episode_"))
        if dagger is None or metadata.get("status") != "complete":
            continue
        if dagger.get("outcome") not in outcomes:
            skipped.append((index, f"outcome {dagger.get('outcome')}"))
            continue
        intervals = dagger.get("human_intervals") or []
        if not intervals:
            continue
        source = processed_paths(root, index)
        missing = [str(path) for path in (source.sync_path, source.hand_path, source.eef_path, source.video_path)
                   if not path.exists()]
        if missing:
            skipped.append((index, "not postprocessed (run postprocess_dataset.py first)"))
            continue
        sync = dict(np.load(source.sync_path, allow_pickle=False))
        times = camera_times(raw_episode)[sync["frame_idx"]]
        trim = DEFAULT_TRIM_MS.get(dagger.get("takeover"), 0.0) if trim_ms is None else trim_ms
        policy = dagger.get("policy") or {}
        for number, (start_ns, end_ns) in enumerate(intervals):
            rows = np.flatnonzero((times >= start_ns + round(trim * 1e6)) & (times <= end_ns))
            if len(rows) < min_rows:
                skipped.append((index, f"interval {number}: {len(rows)} rows < {min_rows}"))
                continue
            target = processed_paths(out, len(segments))
            record = {"episode": len(segments), "rollout": index, "interval": number,
                      "takeover": dagger.get("takeover"), "round": dagger.get("round"),
                      "outcome": dagger.get("outcome"), "policy_sha256": policy.get("sha256"),
                      "interval_ns": [int(start_ns), int(end_ns)], "trim_ms": trim, "rows": int(len(rows)),
                      "duration_s": float((times[rows[-1]] - times[rows[0]]) / 1e9)}
            sliced = {key: sync[key][rows] for key in ("frame_idx", "encoder_idx", "pose_idx")}
            extra = {key: value for key, value in sync.items() if key not in sliced and key != "monotonic_ns"}
            np.savez_compressed(target.sync_path, **sliced, **extra,
                                source=np.array("dagger_human"),
                                provenance=np.array(json.dumps(record)))
            shutil.copy2(source.hand_path, target.hand_path)
            shutil.copy2(source.eef_path, target.eef_path)
            place_video(source.video_path, target.video_path, video)
            segments.append(record)
    summary = {"dagger_root": str(root.resolve()), "outcomes": list(outcomes), "min_rows": min_rows,
               "trim_ms": trim_ms if trim_ms is not None else DEFAULT_TRIM_MS, "segments": segments,
               "skipped": [{"rollout": index, "reason": reason} for index, reason in skipped]}
    (out / "segments.json").write_text(json.dumps(summary, indent=2) + "\n")
    return segments, skipped


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", type=Path, help="Postprocessed executor --dagger dataset")
    parser.add_argument("--out", type=Path, default=None, help="Output root (default: DATASET_ROOT_human)")
    parser.add_argument("--trim-start-ms", type=float, default=None,
                        help="Drop this much from the start of every interval "
                             "(default: 400 teleop, 0 kinesthetic)")
    parser.add_argument("--min-rows", type=int, default=32,
                        help="Skip intervals with fewer camera rows (default 32 = 16 trainer rows at stride 2)")
    parser.add_argument("--outcomes", nargs="+", choices=("success", "failure"), default=["success"],
                        help="Rollout outcomes to keep (default: success)")
    parser.add_argument("--video", choices=("copy", "symlink", "hardlink"), default="copy")
    parser.add_argument("-f", "--force", action="store_true", help="Replace segments already in the output root")
    args = parser.parse_args(argv)
    if args.trim_start_ms is not None and args.trim_start_ms < 0:
        parser.error("--trim-start-ms must be >= 0")
    if args.min_rows < 1:
        parser.error("--min-rows must be >= 1")
    if args.out is None:
        root = args.dataset_root.expanduser().resolve()
        args.out = root.with_name(f"{root.name}_human")
    return args


def main(argv=None):
    args = parse_args(argv)
    segments, skipped = export(args.dataset_root.expanduser(), args.out.expanduser(), trim_ms=args.trim_start_ms,
                               min_rows=args.min_rows, outcomes=tuple(args.outcomes), video=args.video,
                               force=args.force)
    rows = sum(segment["rows"] for segment in segments)
    seconds = sum(segment["duration_s"] for segment in segments)
    print(f"Exported {len(segments)} human segments ({rows} camera rows, {seconds:.1f} s) to {args.out}")
    for index, reason in skipped:
        print(f"  skipped rollout {index}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
