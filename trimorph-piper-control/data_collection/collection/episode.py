from __future__ import annotations

import glob
import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import (
    ARM_MODE_GRAVITY_COMP,
    EEF_FRESH_S,
    PIPER_CURRENT_FIELDS,
    PIPER_ENABLE_FIELDS,
    PIPER_JOINT_FIELDS,
    PIPER_STATUS_FIELDS,
    PIPER_TORQUE_FIELDS,
    PIPER_VELOCITY_FIELDS,
    QUEST_PACKET_FIELDS,
    RAW_EEF_FIELDS,
    TRIMORPH_MOTOR_IDS,
)
from .utils import WorkerError, utc_stamp, write_csv, write_json


class EpisodeConfig(Protocol):
    hand_config: Path
    hand_port: str
    hand_measure_rate_hz: float
    active_motors: list[int]


@dataclass(frozen=True)
class EpisodePaths:
    episode_index: int
    episode_dir: Path
    camera_path: Path
    camera_timestamps_path: Path
    leader_hand_path: Path
    follower_path: Path
    piper_path: Path
    quest_path: Path
    mocap_path: Path
    metadata_path: Path


def raw_root(dataset_root: Path) -> Path:
    return dataset_root / "raw"


def next_episode_index(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in root.glob("episode_*"):
        suffix = path.name.removeprefix("episode_")
        if path.is_dir() and suffix.isdigit():
            indices.append(int(suffix))
    return max(indices, default=-1) + 1


def next_dataset_root(name: Path) -> Path:
    """Return the next unused NAME_n folder; NAME may include parent directories.

    The folder is not created, so a run that records nothing leaves no folder.
    """

    name = Path(name)
    if not name.name or name.name in {".", ".."}:
        raise ValueError(f"Invalid dataset name: {str(name)!r}")
    prefix = f"{name.name}_"
    indices = []
    if name.parent.is_dir():
        for path in name.parent.glob(f"{glob.escape(prefix)}*"):
            suffix = path.name.removeprefix(prefix)
            if path.is_dir() and suffix.isdigit():
                indices.append(int(suffix))
    return name.parent / f"{prefix}{max(indices, default=-1) + 1}"


def episode_paths(dataset_root: Path, episode_index: int) -> EpisodePaths:
    episode_dir = raw_root(dataset_root) / f"episode_{episode_index:06d}"
    return EpisodePaths(
        episode_index=episode_index,
        episode_dir=episode_dir,
        camera_path=episode_dir / "camera.mp4",
        camera_timestamps_path=episode_dir / "camera_timestamps.csv",
        leader_hand_path=episode_dir / "leader_hand.csv",
        follower_path=episode_dir / "follower.csv",
        piper_path=episode_dir / "piper.csv",
        quest_path=episode_dir / "quest.csv",
        mocap_path=episode_dir / "mocap.csv",
        metadata_path=episode_dir / "metadata.json",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_or_none(path: Path | None) -> str | None:
    return _sha256(path) if path is not None and path.is_file() else None


def _hand_metadata(path: Path, port: str) -> dict[str, Any]:
    from midas_hand_api import HandConfig

    hand_config = HandConfig.load(path)
    motor_ids = [int(motor_id) for motor_id in hand_config.motor_ids]

    def by_motor(values) -> dict[str, float]:
        return {str(motor_id): float(value) for motor_id, value in zip(motor_ids, values)}

    return {
        "config_path": str(path.resolve()),
        "config_sha256": _sha256(path),
        "port": port,
        "motor_model": hand_config.motor_model,
        "expected_model_number": hand_config.expected_model_number,
        "baudrate": hand_config.baudrate,
        "current_limit_ma": hand_config.goal_current_limit,
        "motor_ids": list(hand_config.motor_ids),
        # The calibration itself, so a dataset stays readable if the YAML changes.
        "operating_mode": int(hand_config.operating_mode),
        "home_offsets": by_motor(hand_config.home_offsets_array),
        "joint_signs": by_motor(hand_config.joint_signs_array),
        "joint_lower_limits": by_motor(hand_config.joint_lower_limits_array),
        "joint_upper_limits": by_motor(hand_config.joint_upper_limits_array),
    }


def hand_samples_outside_limits(
    path: Path, rows: list[dict[str, object]]
) -> dict[str, dict[str, int]]:
    """Per motor, the leader samples below and above the YAML limits.

    The rollout's set_positions clips to these limits, so these samples are
    targets it would never reach.
    """
    from midas_hand_api import HandConfig

    hand_config = HandConfig.load(path)
    counts: dict[str, dict[str, int]] = {}
    for motor_id, lower, upper in zip(
        hand_config.motor_ids,
        hand_config.joint_lower_limits_array,
        hand_config.joint_upper_limits_array,
    ):
        key = f"pos_{int(motor_id)}"
        values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        counts[str(int(motor_id))] = {
            "below": int(sum(value < lower for value in values)),
            "above": int(sum(value > upper for value in values)),
        }
    return counts


def arm_health_summary(piper_rows: list[dict[str, object]]) -> dict[str, Any]:
    """How the gravity-comp arm feedback looked over an episode (arm_ok: every
    drive enabled and no error code)."""

    def values(name: str) -> list[str]:
        return sorted({str(row.get(name, "")) for row in piper_rows} - {""})

    not_ok = [row for row in piper_rows if str(row.get("arm_ok")) == "0"]
    return {
        "samples": len(piper_rows),
        "not_ok_samples": len(not_ok),
        "first_not_ok_elapsed_s": float(not_ok[0]["elapsed_s"]) if not_ok else None,
        "ctrl_modes": values("ctrl_mode"),
        "arm_statuses": values("arm_status"),
        "err_codes": values("err_code"),
        "eef_samples": sum(1 for row in piper_rows if row.get("x") not in (None, "")),
    }


class EpisodeBuffer:
    def __init__(self, paths: EpisodePaths, config: EpisodeConfig) -> None:
        self.paths = paths
        self.config = config
        self.start_ns = time.monotonic_ns()
        self.started_utc = utc_stamp()
        self.stop_ns: int | None = None
        self.stopped_utc: str | None = None
        self.stop_reason = ""
        self.errors: list[dict[str, str]] = []
        self.lock = threading.Lock()
        self.camera_rows: list[dict[str, object]] = []
        self.leader_hand_rows: list[dict[str, object]] = []
        self.follower_rows: list[dict[str, object]] = []
        self.piper_rows: list[dict[str, object]] = []
        self.quest_rows: list[dict[str, object]] = []
        self.mocap_rows: list[dict[str, object]] = []
        self.video_done = threading.Event()
        self.grav_comp_calib_sha256_at_start = (
            _sha256_or_none(getattr(config, "grav_comp_calib", None))
            if getattr(config, "arm_mode", None) == ARM_MODE_GRAVITY_COMP
            else None
        )

    def elapsed_s(self, sample_ns: int) -> float:
        return (sample_ns - self.start_ns) / 1e9

    def append_camera(self, row: dict[str, object]) -> None:
        with self.lock:
            self.camera_rows.append(row)

    def append_leader_hand(self, row: dict[str, object]) -> None:
        with self.lock:
            self.leader_hand_rows.append(row)

    def append_follower(self, row: dict[str, object]) -> None:
        with self.lock:
            self.follower_rows.append(row)

    def append_piper(self, row: dict[str, object]) -> None:
        with self.lock:
            self.piper_rows.append(row)

    def append_quest(self, row: dict[str, object]) -> None:
        with self.lock:
            self.quest_rows.append(row)

    def append_mocap(self, row: dict[str, object]) -> None:
        with self.lock:
            if self.stop_ns is None and row["monotonic_ns"] >= self.start_ns:
                self.mocap_rows.append(row)

    def mark_stopped(self, reason: str) -> None:
        self.stop_ns = time.monotonic_ns()
        self.stopped_utc = utc_stamp()
        self.stop_reason = reason

    def add_error(self, error: WorkerError) -> None:
        self.errors.append(
            {"worker": error.name, "message": error.message, "traceback": error.traceback_text}
        )

    def flush(self) -> None:
        self.paths.episode_dir.mkdir(parents=True, exist_ok=True)
        with self.lock:
            camera_rows = list(self.camera_rows)
            leader_hand_rows = list(self.leader_hand_rows)
            follower_rows = list(self.follower_rows)
            piper_rows = list(self.piper_rows)
            quest_rows = list(self.quest_rows)
            mocap_rows = list(self.mocap_rows)
        write_csv(self.paths.camera_timestamps_path, camera_rows)
        write_csv(self.paths.leader_hand_path, leader_hand_rows)
        if getattr(self.config, "follower_hand_config", None) is not None:
            write_csv(self.paths.follower_path, follower_rows)
        write_csv(self.paths.piper_path, piper_rows)
        if getattr(self.config, "use_quest", False):
            write_csv(self.paths.quest_path, quest_rows)
        if getattr(self.config, "use_mocap", False):
            from .mocap_worker import write_mocap_csv

            write_mocap_csv(self.paths.mocap_path, mocap_rows)
        write_json(
            self.paths.metadata_path,
            self._metadata(
                camera_rows,
                leader_hand_rows,
                follower_rows,
                piper_rows,
                quest_rows,
                mocap_rows,
            ),
        )

    def _metadata(
        self,
        camera_rows: list[dict[str, object]],
        leader_hand_rows: list[dict[str, object]],
        follower_rows: list[dict[str, object]],
        piper_rows: list[dict[str, object]],
        quest_rows: list[dict[str, object]],
        mocap_rows: list[dict[str, object]],
    ) -> dict[str, Any]:
        stop_ns = self.stop_ns or time.monotonic_ns()
        follower_config = getattr(self.config, "follower_hand_config", None)
        follower_port = getattr(self.config, "follower_hand_port", None)
        has_camera = hasattr(self.config, "task_camera_width") and bool(
            getattr(self.config, "add_camera", True)
        )
        has_piper = hasattr(self.config, "piper_rate_hz") and bool(
            getattr(self.config, "use_piper", True)
        )
        has_quest = bool(getattr(self.config, "use_quest", False))
        gravity_comp = has_piper and getattr(self.config, "arm_mode", None) == ARM_MODE_GRAVITY_COMP
        joint_level = gravity_comp or (
            has_piper and bool(getattr(self.config, "piper_joint_level", False))
        )
        flags: list[str] = []
        gravity_comp_metadata = None
        if gravity_comp:
            profile = getattr(self.config, "start_profile_snapshot", None)
            definition = bool(getattr(self.config, "define_start", False))
            guided_start = (profile["task"] if profile is not None else
                            getattr(self.config, "kinesthetic_start", None))
            if getattr(self.config, "no_start_guide", False) or definition:
                guided_start = None
            target = None
            if guided_start is not None:
                if profile is not None:
                    target = profile["arm"]["joints_deg"]
                else:
                    from .config import PIPER_START_JOINTS_DEG
                    target = list(PIPER_START_JOINTS_DEG[guided_start])
            if definition:
                flags.append("start_definition")
            calib = getattr(self.config, "grav_comp_calib", None)
            calib_sha256 = _sha256_or_none(calib)
            health = arm_health_summary(piper_rows)
            if not piper_rows:
                flags.append("no_piper_samples")
            if health["not_ok_samples"]:
                flags.append("arm_not_ok")
            gravity_comp_metadata = {
                "sends_can_frames": False,
                "row_rate": (
                    "one row per new joint feedback frame, deduplicated on its CAN "
                    "receive time"
                ),
                "poll_rate_hz": getattr(self.config, "piper_rate_hz", None),
                "timestamps": (
                    "robot_timestamp_s: CAN receive time of the joint frame (host wall "
                    "clock); joint_rx_monotonic_ns: the same instant on the host "
                    "monotonic clock; monotonic_ns: when the collector saw the frame"
                ),
                "motor_fields": [
                    *PIPER_VELOCITY_FIELDS,
                    *PIPER_CURRENT_FIELDS,
                    *PIPER_TORQUE_FIELDS,
                ],
                "motor_units": "vel rad/s, current A, torque N*m (current-derived)",
                "enable_fields": list(PIPER_ENABLE_FIELDS),
                "disabled_joints": [6] if getattr(self.config, "grav_comp_disable_j6", False) else [],
                "passive_j6": bool(getattr(self.config, "grav_comp_passive_j6", False)),
                "passive_j5_j6": bool(getattr(self.config, "grav_comp_passive_j5_j6", False)),
                "zero_torque_joints": ([5, 6] if getattr(self.config, "grav_comp_passive_j5_j6", False)
                                       else ([6] if getattr(self.config, "grav_comp_passive_j6", False) else [])),
                "status_fields": [*PIPER_STATUS_FIELDS, "arm_ok"],
                "eef_logged_when": (
                    f"the firmware end pose arrived within {EEF_FRESH_S:g}s of the "
                    "joint frame; blank otherwise"
                ),
                "calibration_path": str(calib.resolve()) if calib is not None else None,
                "calibration_sha256": calib_sha256,
                "calibration_changed_during_episode": (
                    calib_sha256 != self.grav_comp_calib_sha256_at_start
                ),
                "drag_session_id": getattr(self.config, "drag_session_id", None),
                "guided_start": guided_start,
                "guided_start_joints_deg": target,
                "start_guide_enabled": target is not None,
                "task": profile["task"] if profile is not None else None,
                "episode_role": "start_definition" if definition else "demonstration",
                "start_profile_path": (str(self.config.start_profile.resolve())
                                       if getattr(self.config, "start_profile", None) is not None else None),
                "start_profile_sha256": getattr(self.config, "start_profile_sha256", None),
                "start_profile": profile,
                "start_criterion": "end_effector_pose",
                "start_reference": "piper_base flange pose from recorded start joints (SDK MDH FK)",
                "start_position_tolerance_m": getattr(self.config, "kinesthetic_start_position_tolerance_m", 0.10),
                "start_orientation_tolerance_deg": getattr(self.config, "kinesthetic_start_orientation_tolerance_deg", 45.0),
                "arm_health": health,
            }
        kinesthetic_hand = getattr(self.config, "arm_mode", None) == ARM_MODE_GRAVITY_COMP
        return {
            "episode_index": self.paths.episode_index,
            "episode_role": ("start_definition" if getattr(self.config, "define_start", False)
                             else "demonstration"),
            "status": "complete" if not self.errors else "error",
            "flags": flags,
            "stop_reason": self.stop_reason,
            "started_utc": self.started_utc,
            "stopped_utc": self.stopped_utc,
            "started_monotonic_ns": self.start_ns,
            "stopped_monotonic_ns": stop_ns,
            "duration_s": (stop_ns - self.start_ns) / 1e9,
            "sync_clock": "host monotonic_ns",
            "paths": {
                "camera_mp4": self.paths.camera_path.name,
                "camera_timestamps_csv": self.paths.camera_timestamps_path.name,
                "leader_hand_csv": self.paths.leader_hand_path.name,
                "follower_csv": (
                    self.paths.follower_path.name
                    if follower_config is not None
                    else None
                ),
                "piper_csv": self.paths.piper_path.name,
                "quest_csv": self.paths.quest_path.name if has_quest else None,
                "mocap_csv": self.paths.mocap_path.name if getattr(self.config, "use_mocap", False) else None,
            },
            "counts": {
                "camera_frames": len(camera_rows),
                "leader_hand_samples": len(leader_hand_rows),
                "follower_samples": len(follower_rows),
                "piper_samples": len(piper_rows),
                "quest_samples": len(quest_rows),
                "mocap_rows": len(mocap_rows),
            },
            "camera": {
                "device": "oak1_depthai" if has_camera else None,
                "width": (
                    getattr(self.config, "task_camera_width", None)
                    if has_camera
                    else None
                ),
                "height": (
                    getattr(self.config, "task_camera_height", None)
                    if has_camera
                    else None
                ),
                "fps": (
                    getattr(self.config, "task_camera_fps", None)
                    if has_camera
                    else None
                ),
                "flip": (
                    getattr(self.config, "task_camera_flip", None)
                    if has_camera
                    else None
                ),
                "exposure_us": (
                    getattr(self.config, "task_camera_exposure_us", None)
                    if has_camera
                    else None
                ),
                "iso": (
                    getattr(self.config, "task_camera_iso", None)
                    if has_camera
                    else None
                ),
                "white_balance_k": (
                    getattr(self.config, "task_camera_white_balance_k", None)
                    if has_camera
                    else None
                ),
                "timestamp": (
                    "monotonic_ns: DepthAI mid-exposure capture time on the host "
                    "clock; receipt_monotonic_ns: host arrival time"
                    if has_camera
                    else None
                ),
            },
            "trimorph": {
                "motor_ids": list(TRIMORPH_MOTOR_IDS),
                "active_motors": list(self.config.active_motors),
                "hand": _hand_metadata(self.config.hand_config, self.config.hand_port),
                "measure_rate_hz": self.config.hand_measure_rate_hz,
                "force_feedback_rate_hz": getattr(
                    self.config, "force_feedback_rate_hz", None
                ),
                "disabled_force_feedback_motor_ids": list(
                    getattr(self.config, "disabled_force_feedback_motor_ids", ())
                ),
                "training_uses": "measured positions from leader_hand.csv",
                # Operating mode the positions were read in (None: left as found).
                "read_operating_mode": getattr(self.config, "hand_operating_mode", None),
                "torque": "off" if follower_config is None else "force_feedback",
                "samples_outside_yaml_limits": (
                    hand_samples_outside_limits(self.config.hand_config, leader_hand_rows)
                    if kinesthetic_hand
                    else None
                ),
            },
            "follower": {
                "enabled": follower_config is not None,
                "hand": (
                    _hand_metadata(follower_config, follower_port)
                    if follower_config is not None and follower_port is not None
                    else None
                ),
                "measure_rate_hz": getattr(self.config, "follower_measure_rate_hz", None),
                "command_rate_hz": getattr(self.config, "follower_command_rate_hz", None),
                "command_scale": getattr(self.config, "follower_command_scale", None),
                "joint_command_scales": getattr(
                    self.config, "follower_joint_command_scales", None
                ),
                "joint_command_offsets": getattr(
                    self.config, "follower_joint_command_offsets", None
                ),
                "thumb_retarget_model": (
                    str(self.config.thumb_retarget_model)
                    if getattr(self.config, "thumb_retarget_model", None) is not None
                    else None
                ),
                "interpolation_alpha": getattr(
                    self.config, "follower_interpolation_alpha", None
                ),
                "max_step_rad": getattr(self.config, "follower_max_step_rad", None),
            },
            "piper": {
                "mode": (
                    ("gravity_comp_kinesthetic" if gravity_comp else "passive_drag_teach_logging")
                    if has_piper
                    else None
                ),
                "eef_source": "arm_flange_pose_feedback" if has_piper else None,
                "eef_frame": "piper_base" if has_piper else None,
                "eef_position_units": "metres" if has_piper else None,
                "eef_orientation": "Euler xyz radians" if has_piper else None,
                "joint_level_recording": joint_level,
                "target_rate_hz": (
                    getattr(self.config, "piper_rate_hz", None)
                    if has_piper and not gravity_comp
                    else None
                ),
                "raw_eef_fields": list(RAW_EEF_FIELDS),
                "raw_joint_fields": list(PIPER_JOINT_FIELDS) if joint_level else [],
                "start_joints_deg": (
                    list(self.config.piper_start_joints_deg)
                    if getattr(self.config, "piper_start_joints_deg", None) is not None
                    else None
                ),
                "start_pose": (
                    list(self.config.piper_start_pose)
                    if getattr(self.config, "piper_start_pose", None) is not None
                    else None
                ),
                "measured_start_joints_rad": (
                    [float(piper_rows[0][key]) for key in PIPER_JOINT_FIELDS]
                    if piper_rows and all(piper_rows[0].get(key) not in (None, "") for key in PIPER_JOINT_FIELDS)
                    else None
                ),
                "gravity_comp": gravity_comp_metadata,
            },
            "quest": {
                "mode": "udp_pose_logging" if has_quest else None,
                "bind_host": (
                    getattr(self.config, "quest_host", None) if has_quest else None
                ),
                "port": (
                    getattr(self.config, "quest_port", None) if has_quest else None
                ),
                "startup_timeout_s": (
                    getattr(self.config, "quest_startup_timeout_s", None)
                    if has_quest
                    else None
                ),
                "jump_threshold": (
                    getattr(self.config, "quest_jump_threshold", None) if has_quest else None
                ),
                "raw_fields": list(QUEST_PACKET_FIELDS) if has_quest else [],
            },
            "mocap": {
                "enabled": bool(getattr(self.config, "use_mocap", False)),
                "requested_topics": list(getattr(self.config, "mocap_topics", ())),
                "recorded_topics": sorted({row["topic"] for row in mocap_rows}),
                "translation_units": "raw receiver units; see translation_type for Position",
                "rotation_fields": "raw quaternion x_rot, y_rot, z_rot, w; blank for markers",
                "source_clock": "ROS header stamp (sec, nanosec)",
                "receipt_clock": "host monotonic_ns and Unix wall_time_s",
            },
            "errors": self.errors,
        }


class RecordingManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: EpisodeBuffer | None = None

    def current(self) -> EpisodeBuffer | None:
        with self._lock:
            return self._current

    def is_recording(self) -> bool:
        return self.current() is not None

    def start(self, session: EpisodeBuffer) -> None:
        with self._lock:
            if self._current is not None:
                raise RuntimeError("Recording is already active")
            self._current = session

    def stop(self, reason: str) -> EpisodeBuffer | None:
        with self._lock:
            session = self._current
            self._current = None
        if session is not None:
            session.mark_stopped(reason)
        return session
