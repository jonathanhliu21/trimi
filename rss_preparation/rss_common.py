"""Shared constants and helpers for the RSS hand record/replay experiment.

Depends only on ``midas_hand_api`` (imported lazily so plotting works without
it), numpy, and the standard library. Small pieces mirror code in
``trimorph-piper-control`` (motor labels, compatible models, CSV schemas)
without importing it, because that package pulls in camera/mocap modules.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
TRIMI_ROOT = PACKAGE_DIR.parent
CONFIG_DIR = TRIMI_ROOT / "trimorph-piper-control" / "config"
# Known hands. Each YAML pins its own USB adapter in `port`; that is how the
# scripts tell which hand is plugged in (resolve_hand).
HAND_CONFIGS = (CONFIG_DIR / "trimi1.yaml", CONFIG_DIR / "trimi2.yaml")
DEFAULT_CONFIG = HAND_CONFIGS[0]  # joint-sign fallback for plotting old runs
DEFAULT_DATA_ROOT = PACKAGE_DIR / "data"
USB_SERIAL_SYSFS = Path("/sys/bus/usb-serial/devices")

# Same order and names as scripts/trimorph_api_test/gen_graphs_from_csvs.py.
MOTOR_LABELS = {
    0: "Thumb IP",
    1: "Thumb MCP",
    2: "Thumb CMC side",
    3: "Thumb CMC roll",
    4: "Index PIP",
    5: "Index MCP flex",
    6: "Index MCP abduct",
    7: "Middle PIP",
    8: "Middle MCP flex",
    9: "Middle MCP abduct",
    10: "Ring PIP",
    11: "Ring MCP flex",
    12: "Ring MCP abduct",
}
FINGER_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("Thumb", (0, 1, 2, 3)),
    ("Index", (4, 5, 6)),
    ("Middle", (7, 8, 9)),
    ("Ring", (10, 11, 12)),
)

# Mirrors data_collection/collection/hand_common.py: these may be mixed freely.
MODEL_NAMES = {
    1190: "XL330-M077-T",
    1230: "XC330-M181-T",
    1240: "XC330-M288-T",
}
COMPATIBLE_MODEL_NUMBERS = frozenset(MODEL_NAMES)

# Hardware Error Status (address 70) bits for XL330/XC330.
HARDWARE_ERROR_BITS = {
    0: "input voltage",
    2: "overheating",
    3: "motor encoder",
    4: "electrical shock",
    5: "overload",
}

CONDITIONS = ("idle", "tpu", "human")
CONDITION_LABELS = {
    "idle": "Idle (empty)",
    "tpu": "TPU pseudo-hand",
    "human": "Human hand (relaxed)",
}

TRAJECTORY_FILENAME = "leader_hand.csv"

# Replay operating mode: 5 = current-based position (the current cap is
# enforced). Recording switches the motors to it too (torque stays off), so no
# mode change happens between recording and replay.
REPLAY_OPERATING_MODE = 5

# Replay motion profile. The policy rollout (inference/executor/executor.py)
# uses 3 rad/s; the RSS replay uses 6 rad/s to follow faster human motion.
REPLAY_PROFILE_VELOCITY_RAD_S = 6.0
REPLAY_PROFILE_ACCELERATION_RAD_S2 = 50.0


def motor_label(motor_id: int) -> str:
    return MOTOR_LABELS.get(int(motor_id), f"Motor {motor_id}")


def decode_hardware_error(status: int) -> str:
    if not status:
        return "ok"
    names = [name for bit, name in HARDWARE_ERROR_BITS.items() if status & (1 << bit)]
    return ",".join(names) or f"0x{status:02x}"


# --------------------------------------------------------------------------
# CSV schemas
# --------------------------------------------------------------------------


def trajectory_fieldnames(motor_ids: Sequence[int]) -> list[str]:
    """Same columns as collect_hand_only.py's leader_hand.csv."""

    return ["sample_idx", "monotonic_ns", "wall_time_s", "elapsed_s"] + [
        f"pos_{motor_id}" for motor_id in motor_ids
    ]


TELEMETRY_METRICS = (
    "current_ma",
    "target_position_rad",
    "actual_position_rad",
    "target_velocity_rad_s",
    "actual_velocity_rad_s",
)


def telemetry_fieldnames(motor_ids: Sequence[int]) -> list[str]:
    """ReplayCurrentRecorder's schema (replay_episode.py) plus monotonic_ns."""

    fields = ["trial", "elapsed_s", "monotonic_ns"]
    for motor_id in motor_ids:
        fields.extend(f"motor_{motor_id}_{metric}" for metric in TELEMETRY_METRICS)
    return fields


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def load_trajectory(path: Path) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    """Return (motor_ids, times_s, positions[N, M]) from a leader_hand.csv."""

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    motor_ids = tuple(
        int(field.removeprefix("pos_")) for field in fieldnames if field.startswith("pos_")
    )
    if "elapsed_s" not in fieldnames or not motor_ids:
        raise ValueError(f"{path}: expected elapsed_s and pos_<id> columns")
    if len(rows) < 2:
        raise ValueError(f"{path}: needs at least two samples")
    times = np.asarray([float(row["elapsed_s"]) for row in rows])
    positions = np.asarray(
        [[float(row[f"pos_{motor_id}"]) for motor_id in motor_ids] for row in rows]
    )
    if not (np.isfinite(times).all() and np.isfinite(positions).all()):
        raise ValueError(f"{path}: contains NaN or Inf")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{path}: elapsed_s must strictly increase")
    return motor_ids, times - times[0], positions


@dataclass
class Telemetry:
    """One replay run loaded from a telemetry CSV; arrays are [N] or [N, M]."""

    motor_ids: tuple[int, ...]
    times: np.ndarray
    current: np.ndarray
    target: np.ndarray
    actual: np.ndarray


def load_telemetry(path: Path) -> Telemetry:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    motor_ids = tuple(
        int(field.split("_")[1])
        for field in fieldnames
        if field.startswith("motor_") and field.endswith("_current_ma")
    )
    if "elapsed_s" not in fieldnames or not motor_ids:
        raise ValueError(f"{path}: not a replay telemetry CSV")
    if not rows:
        raise ValueError(f"{path}: contains no samples")

    def column(metric: str) -> np.ndarray:
        return np.asarray(
            [[float(row[f"motor_{m}_{metric}"]) for m in motor_ids] for row in rows]
        )

    return Telemetry(
        motor_ids=motor_ids,
        times=np.asarray([float(row["elapsed_s"]) for row in rows]),
        current=column("current_ma"),
        target=column("target_position_rad"),
        actual=column("actual_position_rad"),
    )


def smooth_trajectory(
    times: np.ndarray, positions: np.ndarray, cutoff_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-phase Gaussian low-pass of a recorded trajectory.

    Resamples onto a uniform grid at the median sample period, then convolves
    each joint with a Gaussian whose -3 dB point is ``cutoff_hz``. A Gaussian
    is non-negative, so the output never leaves the recorded range of any
    joint (no overshoot) and adds no phase lag. Edges are padded with the end
    values. ``cutoff_hz <= 0`` returns the input unchanged.
    """

    if cutoff_hz <= 0:
        return times, positions
    dt = float(np.median(np.diff(times)))
    grid = np.arange(times[0], times[-1] + 0.5 * dt, dt)
    grid[-1] = min(grid[-1], times[-1])
    uniform = np.column_stack(
        [np.interp(grid, times, positions[:, j]) for j in range(positions.shape[1])]
    )
    # Gaussian H(f) = exp(-2 pi^2 sigma^2 f^2); H = 1/sqrt(2) at f = cutoff.
    sigma_s = math.sqrt(math.log(math.sqrt(2)) / (2 * math.pi**2)) / cutoff_hz
    half_width = max(1, int(math.ceil(4 * sigma_s / dt)))
    offsets = np.arange(-half_width, half_width + 1) * dt
    kernel = np.exp(-0.5 * (offsets / sigma_s) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(uniform, ((half_width, half_width), (0, 0)), mode="edge")
    smoothed = np.column_stack(
        [np.convolve(padded[:, j], kernel, mode="valid") for j in range(uniform.shape[1])]
    )
    return grid - grid[0], smoothed


def tracking_metrics(telemetry: Telemetry) -> dict[int, dict[str, float]]:
    """Per-motor tracking error (actual - commanded) and current statistics."""

    error = telemetry.actual - telemetry.target
    current = telemetry.current
    metrics = {}
    for index, motor_id in enumerate(telemetry.motor_ids):
        e = error[:, index]
        i = current[:, index]
        metrics[motor_id] = {
            "rms_error_rad": float(np.sqrt(np.mean(e**2))),
            "max_abs_error_rad": float(np.max(np.abs(e))),
            "mean_abs_current_ma": float(np.mean(np.abs(i))),
            "rms_current_ma": float(np.sqrt(np.mean(i**2))),
            "peak_abs_current_ma": float(np.max(np.abs(i))),
        }
    return metrics


# --------------------------------------------------------------------------
# Session layout
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionPaths:
    root: Path

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def trajectory_csv(self) -> Path:
        return self.root / "trajectory" / TRAJECTORY_FILENAME

    @property
    def trajectory_metadata(self) -> Path:
        return self.root / "trajectory" / "metadata.json"

    @property
    def replays_dir(self) -> Path:
        return self.root / "replays"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    def replay_csv(self, condition: str) -> Path:
        return self.replays_dir / f"{condition}.csv"

    def replay_metadata(self, condition: str) -> Path:
        return self.replays_dir / f"{condition}.json"


def session_paths(session: str | Path, data_root: Path = DEFAULT_DATA_ROOT) -> SessionPaths:
    """A bare name maps to data_root/<name>; anything path-like is used as is."""

    candidate = Path(session).expanduser()
    if candidate.is_dir() or len(candidate.parts) > 1:
        return SessionPaths(candidate.resolve())
    return SessionPaths((Path(data_root).expanduser() / candidate).resolve())


def default_session_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# --------------------------------------------------------------------------
# Hand connection
# --------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict:
    import yaml

    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: YAML root must be a mapping")
    return data


def config_port(path: Path) -> str | None:
    """The serial port a hand YAML pins in ``port``, or None."""

    port = _read_yaml(Path(path)).get("port")
    return str(port) if port else None


def resolve_hand(
    config_path: Path | None = None,
    port: str | None = None,
    candidates: Sequence[Path] = HAND_CONFIGS,
) -> tuple[Path, str]:
    """Pick the hand config and serial port for a hardware script.

    ``--config`` names the hand. Without it, use the known hand whose pinned
    USB adapter is plugged in. ``--port`` overrides the config's pinned port.
    """

    if config_path is None:
        known = [Path(p) for p in candidates if Path(p).is_file()]
        plugged = [p for p in known if (pinned := config_port(p)) and Path(pinned).exists()]
        if len(plugged) != 1:
            listing = ", ".join(f"{p.name} -> {config_port(p) or 'no port'}" for p in known)
            reason = "none of the known hands is" if not plugged else "more than one known hand is"
            raise RuntimeError(
                f"cannot pick a hand automatically: {reason} plugged in ({listing}). Pass --config."
            )
        config_path = plugged[0]
    config_path = Path(config_path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Hand config not found: {config_path}")
    if port is None:
        port = config_port(config_path)
        if port is None:
            raise RuntimeError(f"{config_path.name} does not pin a port; pass --port")
    return config_path, str(port)


def add_hand_arguments(parser) -> None:
    """--config / --port for the hardware scripts; resolve with resolve_hand_args."""

    names = " or ".join(p.name for p in HAND_CONFIGS)
    parser.add_argument("--config", type=Path, default=None, help=f"Hand YAML (default: whichever of {names} has its USB adapter plugged in)")
    parser.add_argument("--port", default=None, help="Serial port (default: the port pinned in the config)")


def resolve_hand_args(parser, args) -> None:
    """Replace args.config / args.port with the resolved hand, or exit with a usage error."""

    try:
        args.config, args.port = resolve_hand(args.config, args.port)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


def expected_motor_models(path: Path | None) -> dict[int, str]:
    """The per-ID models a hand YAML declares in ``motor_models`` ({} if none)."""

    if path is None:
        return {}
    declared = _read_yaml(Path(path)).get("motor_models") or {}
    return {int(motor_id): str(name).strip().upper() for motor_id, name in declared.items()}


def model_names(models: Mapping[int, int]) -> dict[int, str]:
    """{motor_id: model name} for ping results ({motor_id: model number})."""

    return {int(m): MODEL_NAMES.get(int(n), f"model {n}") for m, n in sorted(models.items())}


def _model_key(name: str) -> str:
    key = str(name).strip().upper()
    return key[:-2] if key.endswith("-T") else key


def model_mismatches(
    found: Mapping[int, str], expected: Mapping[int, str]
) -> dict[int, tuple[str, str]]:
    """{motor_id: (expected, found)} for IDs present in both whose models differ."""

    return {
        int(m): (expected[m], found[m])
        for m in sorted(set(found) & set(expected))
        if _model_key(found[m]) != _model_key(expected[m])
    }


def describe_mismatches(mismatches: Mapping[int, tuple[str, str]], expected_from: str) -> str:
    details = "; ".join(f"ID {m} expected {want}, found {got}" for m, (want, got) in mismatches.items())
    return f"motor models differ from {expected_from}: {details}"


def load_config(
    path: Path = DEFAULT_CONFIG,
    port: str | None = None,
    *,
    operating_mode: int | None = None,
    current_limit_ma: int | None = None,
    p_gain: int | None = None,
    i_gain: int | None = None,
    d_gain: int | None = None,
):
    from midas_hand_api import HandConfig

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Hand config not found: {path}")
    config = HandConfig.load(path)
    changes: dict[str, object] = {} if port is None else {"port": port}
    if operating_mode is not None:
        changes["operating_mode"] = int(operating_mode)
    if current_limit_ma is not None:
        changes["goal_current_limit"] = int(current_limit_ma)
    for field_name, value in (
        ("position_p_gain", p_gain),
        ("position_i_gain", i_gain),
        ("position_d_gain", d_gain),
    ):
        if value is not None:
            if not 0 <= int(value) <= 16383:
                raise ValueError(f"{field_name} must be between 0 and 16383")
            changes[field_name] = int(value)
    config = replace(config, **changes)
    config.validate()
    return config


def verify_hand(
    hand,
    expected_models: Mapping[int, str] | None = None,
    expected_from: str = "the config",
) -> dict[int, int]:
    """Raise unless every configured motor responds with a compatible model.

    With ``expected_models`` (a YAML's ``motor_models``), each ID must also
    have exactly that model, which catches using another hand's config.
    """

    responses = hand.ping()
    missing = sorted(set(hand.motor_ids) - set(responses))
    if missing:
        raise RuntimeError(f"Hand is missing motor IDs: {missing}")
    wrong = {
        motor_id: model
        for motor_id, model in responses.items()
        if model not in COMPATIBLE_MODEL_NUMBERS
        and model != hand.config.expected_model_number
    }
    if wrong:
        raise RuntimeError(f"Unexpected motor models: {wrong}")
    mismatches = model_mismatches(model_names(responses), expected_models or {})
    if mismatches:
        raise RuntimeError(
            describe_mismatches(mismatches, expected_from)
            + ". Wrong config for this hand? If you swapped motors, update motor_models "
            "in the YAML and re-record."
        )
    return responses


def open_hand(config, config_path: Path | None = None):
    """Connect (this also sets the 1 ms USB latency timer) and verify the bus.

    With ``config_path``, each motor's model is also checked against that
    YAML's ``motor_models``.
    """

    from midas_hand_api import MidasHand

    expected = expected_motor_models(config_path)
    hand = MidasHand(config, autoconnect=True)
    try:
        verify_hand(hand, expected, Path(config_path).name if config_path else "the config")
    except Exception:
        hand.close()
        raise
    return hand


def latency_timer_ms(port: str) -> int | None:
    tty = Path(port).resolve().name
    try:
        return int((USB_SERIAL_SYSFS / tty / "latency_timer").read_text().strip())
    except (OSError, ValueError):
        return None


def warn_if_slow_latency(port: str) -> None:
    value = latency_timer_ms(port)
    if value is not None and value > 1:
        print(
            f"WARNING: USB latency timer is {value} ms; reads are capped near "
            f"{1000 / value:.0f} Hz. Connecting through midas_hand_api normally "
            "sets it to 1 ms; see rss_preparation/README.md if it stays high.",
            file=sys.stderr,
        )


def config_snapshot(config) -> dict:
    return {
        "motor_model": config.motor_model,
        "motor_ids": [int(m) for m in config.motor_ids],
        "baudrate": int(config.baudrate),
        "operating_mode": int(config.operating_mode),
        "position_pid_gains": [
            int(config.position_p_gain),
            int(config.position_i_gain),
            int(config.position_d_gain),
        ],
        "goal_current_limit_ma": int(config.goal_current_limit),
        "home_offsets": [float(v) for v in config.home_offsets_array],
        "joint_signs": [float(v) for v in config.joint_signs_array],
        "joint_lower_limits": [float(v) for v in config.joint_lower_limits_array],
        "joint_upper_limits": [float(v) for v in config.joint_upper_limits_array],
    }


class ReadErrorCounter(logging.Handler):
    """Count ERROR records from the hand API while active.

    A failed sync read in midas_hand_api logs an error and returns the last
    cached values, so this count is the number of possibly stale samples.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1

    def __enter__(self) -> "ReadErrorCounter":
        logging.getLogger().addHandler(self)
        return self

    def __exit__(self, *_exc) -> None:
        logging.getLogger().removeHandler(self)


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


class FixedRateClock:
    """Deadline-based loop timing (same scheme as replay_episode.py).

    ``wait()`` sleeps until the next tick and returns seconds since ``start()``.
    Missed ticks are skipped rather than bunched, and counted as overruns.
    """

    def __init__(self, rate_hz: float) -> None:
        if not math.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("rate_hz must be finite and positive")
        self.period_s = 1.0 / rate_hz
        self.t0 = 0.0
        self.tick = 0
        self.overruns = 0

    def start(self) -> None:
        self.t0 = time.monotonic()
        self.tick = 0
        self.overruns = 0

    def wait(self) -> float:
        deadline = self.t0 + self.tick * self.period_s
        now = time.monotonic()
        if deadline > now:
            time.sleep(deadline - now)
        elif now - deadline > self.period_s:
            self.overruns += 1
        now = time.monotonic()
        self.tick = max(self.tick + 1, math.floor((now - self.t0) / self.period_s) + 1)
        return now - self.t0


def rate_stats(monotonic_ns: Sequence[int]) -> dict[str, float]:
    stamps = np.asarray(monotonic_ns, dtype=np.float64)
    if stamps.size < 2:
        return {"samples": int(stamps.size)}
    dt_ms = np.diff(stamps) / 1e6
    return {
        "samples": int(stamps.size),
        "duration_s": float((stamps[-1] - stamps[0]) / 1e9),
        "achieved_rate_hz": float(1000.0 / np.mean(dt_ms)),
        "dt_median_ms": float(np.median(dt_ms)),
        "dt_p95_ms": float(np.percentile(dt_ms, 95)),
        "dt_max_ms": float(np.max(dt_ms)),
    }


class EnterWatcher:
    """Background thread that sets ``pressed`` when the user hits Enter."""

    def __init__(self) -> None:
        self.pressed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "EnterWatcher":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            line = sys.stdin.readline()
        except (OSError, ValueError):
            return
        if line:  # "" means EOF (no terminal / piped input), not a key press
            self.pressed.set()


def prompt_enter(message: str) -> None:
    input(f"{message} ")


def confirm(message: str) -> bool:
    return input(f"{message} [y/N] ").strip().lower() in ("y", "yes")


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=TRIMI_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty": bool(run("status", "--porcelain")),
    }


def write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
