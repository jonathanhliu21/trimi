#!/usr/bin/env python3
"""Overlay replay conditions: one current figure and one position figure.

Each figure is a 4x4 grid with one row per finger (Thumb, Index, Middle, Ring)
and one subplot per joint. Every subplot overlays the conditions (idle / tpu /
human). The position figure also shows the commanded trajectory (dashed).
A per-joint summary (tracking error, current statistics) is written next to
the figures.

Example:
  python plot_trials.py grasp01
  python plot_trials.py --csv idle=a.csv --csv human=b.csv --out figs/
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rss_common as rc  # noqa: E402


# Reference palette from the dataviz skill (references/palette.md), light mode.
# Slots 1-3 pass the all-pairs CVD and normal-vision checks (overlapping lines).
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CONDITION_SLOTS = {"idle": 0, "tpu": 1, "human": 2}
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
LINE_WIDTH = 1.2


@dataclass
class Run:
    key: str
    label: str
    path: Path
    telemetry: rc.Telemetry
    metadata: dict | None
    color: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", nargs="?", help="Session name (under --data-root) or directory")
    parser.add_argument("--data-root", type=Path, default=rc.DEFAULT_DATA_ROOT, help="Where sessions live (default: %(default)s)")
    parser.add_argument("--csv", action="append", default=[], metavar="LABEL=PATH", help="Explicit run (repeatable); overrides session discovery")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: <session>/figures)")
    parser.add_argument("--degrees", action="store_true", help="Plot positions in degrees instead of radians")
    parser.add_argument("--raw-current-sign", action="store_true", help="Plot current in motor sign (default: multiplied by joint_sign, so positive current drives positive joint motion)")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], help="Figure formats (default: %(default)s)")
    parser.add_argument("--config", type=Path, default=rc.DEFAULT_CONFIG, help="Fallback for joint signs when a run has no metadata (default: %(default)s)")
    args = parser.parse_args()
    if not args.session and not args.csv:
        parser.error("give a session or at least one --csv LABEL=PATH")
    if args.csv and not args.session and args.out is None:
        parser.error("--out is required with --csv and no session")
    return args


def color_for(key: str, used: set[str]) -> str:
    slot = CONDITION_SLOTS.get(key)
    if slot is None:
        slot = next(i for i in range(3, len(SERIES_COLORS)) if SERIES_COLORS[i] not in used)
    return SERIES_COLORS[slot]


def discover_runs(args: argparse.Namespace) -> list[Run]:
    specs: list[tuple[str, Path]] = []
    if args.csv:
        for item in args.csv:
            if "=" not in item:
                raise SystemExit(f"error: --csv expects LABEL=PATH, got {item!r}")
            key, path = item.split("=", 1)
            specs.append((key.strip(), Path(path).expanduser()))
    else:
        paths = rc.session_paths(args.session, args.data_root)
        specs = [(c, paths.replay_csv(c)) for c in rc.CONDITIONS if paths.replay_csv(c).is_file()]
        extras = sorted(p for p in paths.replays_dir.glob("*.csv") if p.stem not in rc.CONDITIONS)
        specs += [(p.stem, p) for p in extras]
        if not specs:
            raise SystemExit(f"error: no replay CSVs in {paths.replays_dir}")

    runs: list[Run] = []
    used: set[str] = set()
    for key, path in specs:
        if not path.is_file():
            raise SystemExit(f"error: {path} does not exist")
        metadata = rc.read_json(path.with_suffix(".json"))
        label = rc.CONDITION_LABELS.get(key, key)
        if metadata and not metadata.get("complete", True):
            label += " (partial)"
            print(f"WARNING: {path.name} is a partial run (aborted or interrupted)")
        color = color_for(key, used)
        used.add(color)
        runs.append(Run(key, label, path, rc.load_telemetry(path), metadata, color))
    return runs


def joint_signs_for(run: Run, fallback: dict[int, float] | None) -> dict[int, float]:
    snapshot = (run.metadata or {}).get("config")
    if snapshot:
        return {int(m): float(s) for m, s in zip(snapshot["motor_ids"], snapshot["joint_signs"])}
    if fallback is None:
        raise SystemExit(f"error: {run.path} has no metadata; cannot sign-correct current")
    print(f"WARNING: {run.path.name} has no metadata; using joint signs from --config")
    return fallback


def check_same_trajectory(runs: list[Run]) -> None:
    hashes = {run.key: (run.metadata or {}).get("trajectory_sha256") for run in runs}
    known = {h for h in hashes.values() if h}
    if len(known) > 1:
        print(f"WARNING: runs replayed different trajectories: {hashes}")


def style_axis(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRIDLINE, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK_MUTED)
        axis.spines[side].set_linewidth(0.6)
    axis.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=8, width=0.6, labelbottom=True)


def grid_axes(plt, title: str):
    figure, axes = plt.subplots(4, 4, figsize=(20, 13), sharex=True, squeeze=False)
    figure.patch.set_facecolor(SURFACE)
    figure.suptitle(title, fontsize=16, color=INK_PRIMARY, fontweight="semibold")
    cells: dict[int, object] = {}
    for row, (finger, motor_ids) in enumerate(rc.FINGER_GROUPS):
        for column in range(4):
            axis = axes[row, column]
            if column >= len(motor_ids):
                axis.axis("off")
                continue
            motor_id = motor_ids[column]
            cells[motor_id] = axis
            style_axis(axis)
            axis.set_title(f"{finger} · ID {motor_id} · {rc.motor_label(motor_id)}", fontsize=10, color=INK_PRIMARY, loc="left")
            if row == len(rc.FINGER_GROUPS) - 1 or (column == 3 and row == 0):
                axis.set_xlabel("Time (s)", color=INK_SECONDARY, fontsize=9)
    legend_axis = axes[1, 3]
    note_axis = axes[2, 3]
    return figure, cells, legend_axis, note_axis


def settings_note(runs: list[Run]) -> str:
    lines = []
    for run in runs:
        meta = run.metadata or {}
        s = meta.get("settings", {})
        timing = meta.get("timing", {})
        if s:
            gains = s.get("position_pid_gains")
            profile = s.get("profile_velocity_rad_s")
            smooth = s.get("smooth_hz")
            extras = []
            if gains:
                extras.append("PID " + "/".join(str(g) for g in gains))
            if profile is not None:
                extras.append(f"profile {profile:g} rad/s" if profile else "no profile")
            if smooth is not None:
                extras.append(f"smooth {smooth:g} Hz" if smooth else "no smoothing")
            lines.append(
                f"{run.label}: mode {s.get('operating_mode')}, cap {s.get('current_limit_ma')} mA, "
                f"{timing.get('achieved_rate_hz', float('nan')):.0f} Hz"
                + (f"\n    {', '.join(extras)}" if extras else "")
            )
    return "\n".join(lines)


def finish(figure, legend_axis, note_axis, handles, labels, note: str, out_base: Path, formats) -> list[Path]:
    legend_axis.legend(handles, labels, loc="center left", frameon=False, fontsize=11, labelcolor=INK_PRIMARY, handlelength=2.6)
    if note:
        note_axis.text(0.0, 0.9, note, transform=note_axis.transAxes, va="top", ha="left", fontsize=8.5, color=INK_SECONDARY)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    written = []
    for fmt in formats:
        path = out_base.with_suffix(f".{fmt}")
        figure.savefig(path, dpi=200, facecolor=figure.get_facecolor())
        written.append(path)
    return written


def make_figures(runs: list[Run], out_dir: Path, *, degrees: bool, sign_correct: bool, fallback_signs, formats, title_suffix: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out_dir.mkdir(parents=True, exist_ok=True)
    scale = 180.0 / math.pi if degrees else 1.0
    unit = "deg" if degrees else "rad"
    written: list[Path] = []
    note = settings_note(runs)

    # Current figure.
    figure, cells, legend_axis, note_axis = grid_axes(plt, f"Motor current per joint{title_suffix}")
    for run in runs:
        signs = joint_signs_for(run, fallback_signs) if sign_correct else {}
        for index, motor_id in enumerate(run.telemetry.motor_ids):
            if motor_id not in cells:
                continue
            current = run.telemetry.current[:, index] * signs.get(motor_id, 1.0)
            cells[motor_id].plot(run.telemetry.times, current, color=run.color, linewidth=LINE_WIDTH)
    for axis in cells.values():
        axis.set_ylabel("Current (mA)", color=INK_SECONDARY, fontsize=9)
    handles = [Line2D([], [], color=run.color, linewidth=2.4) for run in runs]
    written += finish(
        figure, legend_axis, note_axis, handles, [run.label for run in runs],
        note + ("" if sign_correct else "\nCurrent in raw motor sign"),
        out_dir / "current_overlay", formats,
    )
    plt.close(figure)

    # Position figure: commanded once (first run), measured per run.
    figure, cells, legend_axis, note_axis = grid_axes(plt, f"Joint position per joint: commanded vs measured{title_suffix}")
    reference = runs[0].telemetry
    for index, motor_id in enumerate(reference.motor_ids):
        if motor_id in cells:
            cells[motor_id].plot(reference.times, reference.target[:, index] * scale, color=INK_SECONDARY, linewidth=1.4, linestyle=(0, (4, 2.5)), zorder=3)
    for run in runs:
        for index, motor_id in enumerate(run.telemetry.motor_ids):
            if motor_id in cells:
                cells[motor_id].plot(run.telemetry.times, run.telemetry.actual[:, index] * scale, color=run.color, linewidth=LINE_WIDTH, zorder=2)
    for axis in cells.values():
        axis.set_ylabel(f"Position ({unit})", color=INK_SECONDARY, fontsize=9)
    handles = [Line2D([], [], color=INK_SECONDARY, linewidth=1.8, linestyle=(0, (4, 2.5)))]
    handles += [Line2D([], [], color=run.color, linewidth=2.4) for run in runs]
    labels = ["Commanded"] + [f"Measured: {run.label}" for run in runs]
    written += finish(figure, legend_axis, note_axis, handles, labels, note, out_dir / "position_overlay", formats)
    plt.close(figure)
    return written


SUMMARY_FIELDS = (
    "condition", "motor_id", "joint",
    "rms_error_rad", "rms_error_deg", "max_abs_error_rad", "max_abs_error_deg",
    "mean_abs_current_ma", "rms_current_ma", "peak_abs_current_ma",
)


def summary_rows(runs: list[Run]) -> list[dict]:
    rows = []
    for run in runs:
        metrics = rc.tracking_metrics(run.telemetry)
        for motor_id, m in metrics.items():
            rows.append(_summary_row(run.key, motor_id, rc.motor_label(motor_id), m))
        mean = {k: float(np.mean([m[k] for m in metrics.values()])) for k in next(iter(metrics.values()))}
        rows.append(_summary_row(run.key, "all", "mean over joints", mean))
    return rows


def _summary_row(condition, motor_id, joint, m: dict) -> dict:
    return {
        "condition": condition,
        "motor_id": motor_id,
        "joint": joint,
        "rms_error_rad": round(m["rms_error_rad"], 6),
        "rms_error_deg": round(math.degrees(m["rms_error_rad"]), 4),
        "max_abs_error_rad": round(m["max_abs_error_rad"], 6),
        "max_abs_error_deg": round(math.degrees(m["max_abs_error_rad"]), 4),
        "mean_abs_current_ma": round(m["mean_abs_current_ma"], 2),
        "rms_current_ma": round(m["rms_current_ma"], 2),
        "peak_abs_current_ma": round(m["peak_abs_current_ma"], 2),
    }


def print_summary(rows: list[dict], runs: list[Run]) -> None:
    keys = [run.key for run in runs]
    by = {(row["condition"], str(row["motor_id"])): row for row in rows}
    motor_ids = [str(m) for m in runs[0].telemetry.motor_ids] + ["all"]
    width = max(10, *(len(k) for k in keys))
    header = f"{'ID':>3} {'joint':<17} | " + " ".join(f"{k:>{width}}" for k in keys) + " | " + " ".join(f"{k:>{width}}" for k in keys)
    print(f"{'':>3} {'':<17} | {'RMS tracking error (deg)':^{(width + 1) * len(keys) - 1}} | {'mean |current| (mA)':^{(width + 1) * len(keys) - 1}}")
    print(header)
    print("-" * len(header))
    for m in motor_ids:
        first = by.get((keys[0], m))
        joint = first["joint"] if first else ""
        errors = " ".join(f"{by[(k, m)]['rms_error_deg']:>{width}.2f}" if (k, m) in by else f"{'-':>{width}}" for k in keys)
        currents = " ".join(f"{by[(k, m)]['mean_abs_current_ma']:>{width}.0f}" if (k, m) in by else f"{'-':>{width}}" for k in keys)
        print(f"{m:>3} {joint:<17} | {errors} | {currents}")


def main() -> int:
    args = parse_args()
    runs = discover_runs(args)
    check_same_trajectory(runs)
    if args.out is not None:
        out_dir = args.out.expanduser()
        title_suffix = ""
    else:
        paths = rc.session_paths(args.session, args.data_root)
        out_dir = paths.figures_dir
        title_suffix = f" (session {paths.name})"

    fallback = None
    if not args.raw_current_sign and any(run.metadata is None for run in runs):
        config = rc.load_config(args.config, None)
        fallback = {int(m): float(s) for m, s in zip(config.motor_ids, config.joint_signs_array)}

    written = make_figures(
        runs, out_dir,
        degrees=args.degrees,
        sign_correct=not args.raw_current_sign,
        fallback_signs=fallback,
        formats=args.formats,
        title_suffix=title_suffix,
    )
    rows = summary_rows(runs)
    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    written.append(summary_path)

    print_summary(rows, runs)
    print()
    for path in written:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
