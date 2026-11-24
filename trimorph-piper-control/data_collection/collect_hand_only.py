import argparse
import hashlib
import json
import math
import queue
import shutil
import sys
import threading
import time

from dataclasses import dataclass, replace
from pathlib import Path

# Support both direct execution and package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_collection.collection.config import (
    WORKER_INIT_TIMEOUT_S,
    VIDEO_FLUSH_TIMEOUT_S,
    DEFAULT_DEMONSTRATION_DURATION_S,
    DEFAULT_WARMUP_TIME_S,
    DEFAULT_HAND_MEASURE_RATE_HZ,
    DEFAULT_PIPER_RATE_HZ,
    DEFAULT_PIPER_INTERFACE,
    DEFAULT_PIPER_CHANNEL,
    DEFAULT_PIPER_FIRMWARE,
    DEFAULT_PIPER_INSTALLATION_POS,
    DEFAULT_PIPER_PAYLOAD,
    DEFAULT_GRAVITY_COMP_POLL_HZ,
    DEFAULT_GRAV_COMP_CALIB,
    ARM_MODE_DRAG_TEACH,
    ARM_MODE_GRAVITY_COMP,
    ARM_MODES,
    KINESTHETIC_HAND_CONFIG,
    KINESTHETIC_HAND_OPERATING_MODE,
    KINESTHETIC_HAND_PORT,
    DEFAULT_TEACHER_MODE_POLL_HZ,
    DEFAULT_HAND_CONFIG,
    DEFAULT_HAND_PORT,
    DEFAULT_QUEST_JUMP_THRESHOLD,
    DEFAULT_EXECUTOR_START,
    PIPER_START_JOINTS_DEG,
    DATA_COLLECTION_MODE,
    RETARGET_ONLY_MODE,
    DEFAULT_THUMB_MODEL,
    HandOnlyCollectionConfig,
    DEFAULT_DATASET_NAME,
    TASK_VIEW_WINDOW_NAME,
    TRIMORPH_MOTOR_IDS,
    validate_camera_settings,
)
from data_collection.collection.depthai_task_camera_worker import run_depthai_task_camera_worker
from data_collection.collection.episode import (
    EpisodeBuffer,
    RecordingManager,
    episode_paths,
    next_dataset_root,
    next_episode_index,
    raw_root,
)
from data_collection.collection.follower_hand_worker import run_follower_hand_worker
from data_collection.collection.leader_hand_worker import run_leader_hand_worker
from data_collection.collection.piper_worker import run_piper_worker
from data_collection.collection.quest_worker import run_quest_worker
from data_collection.collection.mocap_worker import run_mocap_worker
from data_collection.collection.shared_state import SharedState
from data_collection.collection.tactile_worker import run_tactile_worker
from data_collection.collection.utils import WorkerError, import_cv2, worker_wrapper
from scripts.piper_api.foot_pedal import FootPedalTrigger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retarget a follower from the TRIMORPH leader hand, optionally recording "
            "episodes with a foot pedal or for a fixed duration."
        )
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--retarget-only",
        dest="mode",
        action="store_const",
        const=RETARGET_ONLY_MODE,
        help="Retarget indefinitely without recording; stop with Ctrl-C.",
    )
    mode_group.add_argument(
        "--data-collection",
        dest="mode",
        action="store_const",
        const=DATA_COLLECTION_MODE,
        help=(
            "Record episodes (the default; foot-pedal controlled unless "
            "--episode-s is set)."
        ),
    )
    parser.set_defaults(mode=DATA_COLLECTION_MODE)
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset-name",
        type=Path,
        default=Path(DEFAULT_DATASET_NAME),
        help=(
            "Save this run to the next free NAME_n folder (n = 0, 1, 2, ...); NAME "
            "may include parent directories (default: %(default)s)."
        ),
    )
    dataset_group.add_argument(
        "--dataset-root",
        type=Path,
        help="Save to exactly this folder, adding episodes after any already there.",
    )
    parser.add_argument(
        "--hand-config",
        type=Path,
        default=None,
        help=(
            f"hand YAML (default: {DEFAULT_HAND_CONFIG}; with --arm-mode gravity-comp, "
            f"the rollout's {KINESTHETIC_HAND_CONFIG})"
        ),
    )
    parser.add_argument(
        "--hand-port",
        default=None,
        help=(
            f"hand serial port (default: {DEFAULT_HAND_PORT}; with --arm-mode "
            f"gravity-comp, {KINESTHETIC_HAND_PORT})"
        ),
    )
    parser.add_argument("--follower-config", type=Path)
    parser.add_argument("--follower-port")
    parser.add_argument(
        "--show-ui",
        action="store_true",
        help="Show the read-only follower MIDAS live-contact visualization.",
    )
    # The leader hand, camera, Quest and mocap are recorded by default; disable
    # them with --no-camera, --no-quest and --no-mocap.
    parser.add_argument(
        "--add-camera",
        "--camera",
        dest="add_camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="record the DepthAI wrist camera and show its live feed (default: on)",
    )
    parser.add_argument(
        "--use-mocap",
        "--mocap",
        dest="use_mocap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "record ROS 2 Vicon poses/markers to each episode's mocap.csv "
            "(default: on, off with --retarget-only)"
        ),
    )
    parser.add_argument("--mocap-topic", dest="mocap_topics", action="append", default=[], help="Select a Vicon topic (repeatable); default: discover all supported topics. Requires mocap.")
    parser.add_argument(
        "--use-piper",
        action="store_true",
        help=(
            "passively record measured Piper flange XYZ (m) and roll/pitch/yaw (rad) "
            "instead of Quest; add --no-mocap for arm-only pose feedback. "
            "Joint logging is opt-in with --piper-joint-level"
        ),
    )
    parser.add_argument(
        "--arm-mode",
        choices=ARM_MODES,
        default=None,
        help=(
            "how the arm is moved while --use-piper records it. drag-teach (default): "
            "the firmware drag-teach button. gravity-comp: gravity_comp.drag_mode runs "
            "in another process and commands the arm; the collector only listens and "
            "sends nothing on the CAN bus. gravity-comp implies --use-piper --no-mocap, "
            "records joints, motor feedback and arm status for every joint frame, and "
            "reads the hand torque-off in the rollout's operating mode."
        ),
    )
    parser.add_argument(
        "--grav-comp-calib",
        type=Path,
        default=None,
        help=(
            "gravity-comp: the calibration drag mode runs on, recorded with its sha256 "
            f"(default: {DEFAULT_GRAV_COMP_CALIB})"
        ),
    )
    parser.add_argument(
        "--drag-session-id",
        default=None,
        help="gravity-comp: the launcher's drag-mode session id, recorded in metadata",
    )
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--kinesthetic-start", choices=sorted(PIPER_START_JOINTS_DEG),
                       help="gravity-comp: explicit legacy named start (no motion command)")
    start.add_argument("--start-profile", type=Path,
                       help="gravity-comp: saved task start JSON to guide by hand")
    start.add_argument("--no-start-guide", action="store_true",
                       help="gravity-comp: log without a start target")
    start.add_argument("--define-start", action="store_true",
                       help="gravity-comp: record a task-start definition clip, without a saved target")
    wrist = parser.add_mutually_exclusive_group()
    wrist.add_argument("--grav-comp-disable-j6", action="store_true",
                        help="expect J6 disabled by drag mode; this collector still sends no commands")
    wrist.add_argument("--grav-comp-passive-j6", action="store_true",
                        help="expect drag mode to keep all six drives enabled with zero J6 MIT gains/torque; "
                             "this collector still sends no commands")
    wrist.add_argument("--grav-comp-passive-j5-j6", action="store_true",
                        help="expect drag mode to keep all six drives enabled with zero J5/J6 MIT gains/torque; "
                             "this collector still sends no commands")
    parser.add_argument("--kinesthetic-start-position-tolerance-m", type=float, default=0.10,
                        help="maximum flange position distance from the recorded start (default: 0.10 m)")
    parser.add_argument("--kinesthetic-start-orientation-tolerance-deg", type=float, default=45.0,
                        help="maximum total flange rotation error from the recorded start (default: 45 degrees)")
    parser.add_argument(
        "--use-quest",
        "--quest",
        dest="use_quest",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "record raw Quest position and quaternion packets received over UDP "
            "alongside the hand (default: on, off with --use-piper or --retarget-only)"
        ),
    )
    parser.add_argument(
        "--monitor",
        dest="show_tracking_monitor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "show a live 3D view of the Quest and Vicon poses with green/red signal "
            "status boxes (default: on when Quest or mocap is recorded)"
        ),
    )
    parser.add_argument(
        "--monitor-calibration",
        type=Path,
        help=(
            "calibration JSON whose linear_matrix / vicon_reference map the monitor's "
            "Quest / Vicon motion to Piper axes (default: each tracker's own axes)"
        ),
    )
    parser.add_argument('--mocap-ik-calibration', type=Path,
                        help='Open a live Piper IK warning window using this Vicon-reference calibration; requires mocap')
    parser.add_argument('--mocap-ik-topic', default='/vicon/Yori/Yori',
                        help='Marker plate topic to check for IK (default: /vicon/Yori/Yori)')
    parser.add_argument('--mocap-ik-seed-deg', type=float, nargs=6,
                        default=(0., 45., -45., 0., 30., 0.),
                        help='Initial offline IK branch seed in degrees; no arm connection')
    parser.add_argument('--mocap-ik-start', choices=sorted(PIPER_START_JOINTS_DEG), default=DEFAULT_EXECUTOR_START,
                        help='Executor start pose marked on each IK joint plot; the window turns blue '
                             'when every joint is within --mocap-ik-start-tolerance-deg (default: %(default)s)')
    parser.add_argument('--mocap-ik-start-tolerance-deg', type=float, default=5.,
                        help='Per-joint tolerance for the start marker (default: %(default)s)')
    parser.add_argument(
        "--quest-jump-threshold", type=float, default=DEFAULT_QUEST_JUMP_THRESHOLD,
        help="Warn when consecutive Quest packets move farther than this distance in native position units (default: 0.05).",
    )
    parser.add_argument(
        "--piper-joint-level", "--joint-level", "--joint-position",
        dest="piper_joint_level",
        action="store_true",
        help="Record measured Piper joint angles alongside EEF poses for joint-space replay (still passive).",
    )
    parser.add_argument(
        "--piper-joint-monitor",
        dest="show_piper_joint_monitor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Show measured Piper joints against SDK limits in a separate window "
            "(red within 3 degrees of either limit). Enables joint recording; "
            "defaults to on with --piper-joint-level or --arm-mode gravity-comp."
        ),
    )
    parser.add_argument(
        "--piper-rate-hz",
        type=float,
        default=None,
        help=(
            f"Piper sampling rate in Hz (default: {DEFAULT_PIPER_RATE_HZ:g}). With "
            "--arm-mode gravity-comp, how often the SDK cache is checked for a new "
            f"joint frame (default: {DEFAULT_GRAVITY_COMP_POLL_HZ:g})."
        ),
    )
    parser.add_argument(
        "--piper-teacher-mode-poll-hz",
        type=float,
        default=DEFAULT_TEACHER_MODE_POLL_HZ,
        help="Piper status polling rate in Hz (default: %(default)s).",
    )
    parser.add_argument("--piper-channel", default=DEFAULT_PIPER_CHANNEL)
    parser.add_argument("--piper-interface", default=DEFAULT_PIPER_INTERFACE)
    parser.add_argument(
        "--piper-firmware",
        choices=("default", "v183", "v188"),
        default=DEFAULT_PIPER_FIRMWARE,
    )
    parser.add_argument(
        "--piper-installation-pos",
        choices=("unchanged", "horizontal", "left", "right"),
        default=None,
        help=(
            f"set on connect (default: {DEFAULT_PIPER_INSTALLATION_POS}; always "
            "unchanged with --arm-mode gravity-comp)"
        ),
    )
    parser.add_argument(
        "--piper-payload",
        choices=("unchanged", "empty", "half", "full"),
        default=None,
        help=(
            f"set on connect (default: {DEFAULT_PIPER_PAYLOAD}; always unchanged with "
            "--arm-mode gravity-comp)"
        ),
    )
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument(
        "--piper-start-joints-deg",
        type=float,
        nargs=6,
        metavar="DEG",
        help=(
            "Before collecting, enable Piper and move it in joint space to these "
            "angles, leaving it holding there; its measured flange pose is saved "
            "in every episode's metadata as the postprocessing start pose."
        ),
    )
    start_group.add_argument(
        "--piper-start-home",
        action="store_true",
        help="Like --piper-start-joints-deg, using the teleop home (config/teleop_home.json).",
    )
    parser.add_argument(
        "--piper-start-speed-deg-s",
        type=float,
        default=10.0,
        help="Joint speed of the move to the start pose (default: %(default)s).",
    )
    parser.add_argument(
        "--add-tactile",
        action="store_true",
        help=(
            "connect the auto-detected Paxini tactile sensor and show binary "
            "fingertip contacts in the live-contact UI"
        ),
    )
    parser.add_argument(
        "--paxini-port",
        metavar="PORT",
        help=(
            "Paxini serial port, for example /dev/cu.usbserial-XXXX; providing "
            "this option also enables tactile sensing"
        ),
    )
    parser.add_argument(
        "--tactile-debug-log",
        type=Path,
        metavar="PATH",
        help=(
            "write structured tactile contact-detector diagnostics to PATH; "
            "providing this option also enables tactile sensing"
        ),
    )
    parser.add_argument(
        "--thumb-model",
        type=Path,
        default=DEFAULT_THUMB_MODEL,
        help=(
            "Linear TRIMORPH-to-MIDAS thumb model produced by "
            "scripts.thumb_retargeting.fit_model_to_endpoints.py (default: %(default)s)."
        ),
    )
    episode_group = parser.add_mutually_exclusive_group()
    episode_group.add_argument(
        "--episode-s",
        "--episode-time",
        dest="episode_time",
        type=float,
        default=None,
        help=(
            "record one fixed-duration episode instead of using the foot pedal "
            "(the --episode-time spelling is retained for compatibility)"
        ),
    )
    episode_group.add_argument(
        "--use-foot-pedal",
        action="store_true",
        default=True,
        help=(
            "use the foot pedal to start and save episodes, or discard an episode "
            "with a long hold (default)"
        ),
    )
    parser.add_argument(
        "--warmup-time",
        type=float,
        default=DEFAULT_WARMUP_TIME_S,
    )
    parser.add_argument(
        "--active-motors",
        nargs="+",
        type=int,
        default=list(TRIMORPH_MOTOR_IDS),
        help=(
            "Motor IDs to read, command, and record. The hand and follower YAML "
            "files may contain additional motor IDs."
        ),
    )
    parser.add_argument(
        "--disable-force-feedback-motors",
        nargs="+",
        type=int,
        default=[],
        metavar="MOTOR_ID",
        help=(
            "Keep these active TRIMORPH motors torque-disabled while retaining "
            "their position readings and retargeting (for example: 8 or 7 8 9)."
        ),
    )
    args = parser.parse_args()
    gravity_comp = args.arm_mode == ARM_MODE_GRAVITY_COMP
    if gravity_comp:
        if args.kinesthetic_start is None and args.start_profile is None:
            args.no_start_guide = True
        # Drag mode holds the arm in MIT torque mode and is the only writer on can0.
        # Anything that would transmit is refused rather than silently dropped.
        if args.use_quest:
            parser.error("--arm-mode gravity-comp records the arm pose; remove --use-quest")
        if args.piper_start_joints_deg is not None or args.piper_start_home:
            parser.error(
                "--arm-mode gravity-comp never moves the arm; remove --piper-start-* "
                "and bring it to the start pose by hand"
            )
        for flag, value in (
            ("--piper-installation-pos", args.piper_installation_pos),
            ("--piper-payload", args.piper_payload),
        ):
            if value not in (None, "unchanged"):
                parser.error(f"--arm-mode gravity-comp sends no settings frames; remove {flag}")
        if args.follower_config is not None or args.follower_port is not None or args.show_ui:
            parser.error(
                "--arm-mode gravity-comp reads the hand torque-off; it has no follower "
                "(remove --follower-config/--follower-port/--show-ui)"
            )
        args.use_piper = True
        args.piper_installation_pos = args.piper_payload = "unchanged"
        if args.use_mocap is None:
            args.use_mocap = False
    else:
        for flag, enabled in (("--grav-comp-disable-j6", args.grav_comp_disable_j6),
                              ("--grav-comp-passive-j6", args.grav_comp_passive_j6),
                              ("--grav-comp-passive-j5-j6", args.grav_comp_passive_j5_j6),
                              ("--no-start-guide", args.no_start_guide),
                              ("--define-start", args.define_start)):
            if enabled:
                parser.error(f"{flag} requires --arm-mode gravity-comp")
        if args.arm_mode is not None and not args.use_piper:
            parser.error("--arm-mode requires --use-piper")
        for flag, value in (
            ("--grav-comp-calib", args.grav_comp_calib),
            ("--drag-session-id", args.drag_session_id),
            ("--kinesthetic-start", args.kinesthetic_start),
            ("--start-profile", args.start_profile),
        ):
            if value is not None:
                parser.error(f"{flag} requires --arm-mode gravity-comp")
        if args.piper_installation_pos is None:
            args.piper_installation_pos = DEFAULT_PIPER_INSTALLATION_POS
        if args.piper_payload is None:
            args.piper_payload = DEFAULT_PIPER_PAYLOAD
    if args.arm_mode is None:
        args.arm_mode = ARM_MODE_DRAG_TEACH
    if (not math.isfinite(args.kinesthetic_start_position_tolerance_m)
            or args.kinesthetic_start_position_tolerance_m <= 0):
        parser.error("--kinesthetic-start-position-tolerance-m must be finite and positive")
    if (not math.isfinite(args.kinesthetic_start_orientation_tolerance_deg)
            or not 0 < args.kinesthetic_start_orientation_tolerance_deg <= 180):
        parser.error("--kinesthetic-start-orientation-tolerance-deg must be finite and in (0, 180]")
    if args.hand_config is None:
        args.hand_config = KINESTHETIC_HAND_CONFIG if gravity_comp else DEFAULT_HAND_CONFIG
    if args.hand_port is None:
        args.hand_port = KINESTHETIC_HAND_PORT if gravity_comp else DEFAULT_HAND_PORT
    if args.piper_rate_hz is None:
        args.piper_rate_hz = DEFAULT_GRAVITY_COMP_POLL_HZ if gravity_comp else DEFAULT_PIPER_RATE_HZ
    if gravity_comp and args.grav_comp_calib is None:
        args.grav_comp_calib = DEFAULT_GRAV_COMP_CALIB
    # Quest and mocap only record, so retarget-only mode leaves them off unless asked.
    recording = args.mode == DATA_COLLECTION_MODE
    if args.use_quest and args.use_piper:
        parser.error("--use-piper and --use-quest are mutually exclusive pose sources")
    if args.use_quest is None:
        args.use_quest = recording and not args.use_piper
    if args.use_mocap is None:
        args.use_mocap = recording
    if args.mocap_topics and not args.use_mocap:
        parser.error("--mocap-topic requires mocap (remove --no-mocap)")
    if args.mocap_ik_calibration is not None:
        if not args.use_mocap:
            parser.error('--mocap-ik-calibration requires mocap (remove --no-mocap)')
        if args.mocap_topics and args.mocap_ik_topic not in args.mocap_topics:
            parser.error('--mocap-ik-topic must be included in the selected --mocap-topic values')
        if not all(math.isfinite(value) for value in args.mocap_ik_seed_deg):
            parser.error('--mocap-ik-seed-deg requires six finite values')
        if not math.isfinite(args.mocap_ik_start_tolerance_deg) or args.mocap_ik_start_tolerance_deg <= 0:
            parser.error('--mocap-ik-start-tolerance-deg must be finite and positive')
    if args.show_tracking_monitor is None:
        args.show_tracking_monitor = args.use_quest or args.use_mocap or gravity_comp
    elif args.show_tracking_monitor and not (args.use_quest or args.use_mocap or gravity_comp):
        parser.error("--monitor requires Quest, mocap or --arm-mode gravity-comp")
    if args.monitor_calibration is not None and not args.show_tracking_monitor:
        parser.error("--monitor-calibration requires the tracking monitor")
    if args.show_piper_joint_monitor is None:
        args.show_piper_joint_monitor = args.piper_joint_level or gravity_comp
    if args.show_piper_joint_monitor:
        if not args.use_piper:
            parser.error("--piper-joint-monitor requires --use-piper or --arm-mode gravity-comp")
        args.piper_joint_level = True
    if args.episode_time is not None:
        args.use_foot_pedal = False
    return args


def worker_specs(config: HandOnlyCollectionConfig):
    specs = []
    if config.add_camera:
        specs.append(("task_camera", run_depthai_task_camera_worker))
    if config.use_piper:
        specs.append(("piper", run_piper_worker))
    if config.use_quest:
        specs.append(("quest", run_quest_worker))
    if config.use_mocap:
        specs.append(("mocap", run_mocap_worker))
    specs.append(("leader_hand", run_leader_hand_worker))
    if config.follower_hand_config is not None:
        specs.append(("follower", run_follower_hand_worker))
    if config.add_tactile:
        specs.append(("tactile", run_tactile_worker))
    return specs


def initialize_worker(
    config: HandOnlyCollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    errors: queue.Queue[WorkerError],
) -> list[threading.Thread]:
    specs = worker_specs(config)

    threads: list[threading.Thread] = []
    try:
        for name, target in specs:
            init_done = threading.Event()
            thread = threading.Thread(
                target=worker_wrapper,
                name=f"{name}-worker",
                args=(
                    name,
                    errors,
                    shared,
                    init_done,
                    target,
                    config,
                    shared,
                    recorder,
                ),
                daemon=True,
            )
            print(f"Starting {name} worker...")
            thread.start()
            threads.append(thread)
            if not init_done.wait(timeout=WORKER_INIT_TIMEOUT_S):
                raise TimeoutError(
                    f"{name.capitalize()} worker did not initialize within "
                    f"{WORKER_INIT_TIMEOUT_S}s"
                )
            if not errors.empty():
                error = errors.get_nowait()
                raise RuntimeError(f"{error.name} worker failed: {error.message}")
    except Exception:
        shared.shutdown.set()
        for thread in threads:
            thread.join(timeout=3.0)
        raise

    return threads


def start_episode(
    config: HandOnlyCollectionConfig, recorder: RecordingManager
) -> EpisodeBuffer:
    index = next_episode_index(raw_root(config.dataset_root))
    paths = episode_paths(config.dataset_root, index)
    paths.episode_dir.mkdir(parents=True, exist_ok=False)
    session = EpisodeBuffer(paths, config)
    recorder.start(session)
    if config.use_foot_pedal:
        print(
            f"Recording episode {index:06d}. Press pedal to save; "
            f"hold {config.hold_delete_s:g}s to delete."
        )
    else:
        print(f"Recording episode {index:06d} for {config.episode_time:.1f}s...")
    return session


def stop_episode(recorder: RecordingManager, reason: str) -> EpisodeBuffer | None:
    session = recorder.stop(reason)
    if session is None:
        return None
    if session.config.add_camera and not session.video_done.wait(
        timeout=VIDEO_FLUSH_TIMEOUT_S
    ):
        print(
            f"Warning: camera writer did not finish within {VIDEO_FLUSH_TIMEOUT_S:.1f}s"
        )
    session.flush()
    print(
        f"Saved episode {session.paths.episode_index:06d}: "
        f"{len(session.leader_hand_rows)} leader-hand samples, "
        f"{len(session.follower_rows)} follower samples, "
        f"{len(session.camera_rows)} camera frames, "
        f"{len(session.piper_rows)} Piper samples, "
        f"{len(session.quest_rows)} Quest samples, "
        f"{len(session.mocap_rows)} mocap rows"
    )
    return session


def delete_episode(recorder: RecordingManager) -> EpisodeBuffer | None:
    """Stop and discard the current episode after a long pedal hold."""

    session = recorder.stop("foot_pedal_delete")
    if session is None:
        return None
    if session.config.add_camera:
        session.video_done.wait(timeout=VIDEO_FLUSH_TIMEOUT_S)
    shutil.rmtree(session.paths.episode_dir, ignore_errors=True)
    print(f"Episode {session.paths.episode_index:06d} deleted.")
    return session


@dataclass
class FootPedalCollectionController:
    config: HandOnlyCollectionConfig
    recorder: RecordingManager
    pedal: FootPedalTrigger
    pending_start_deadline: float | None = None
    recording_pedal_down: bool = False

    def poll(self) -> None:
        """Handle pedal start, save on release, and discard after a long hold."""

        pedal_pressed = bool(self.pedal.poll())
        if self.recorder.is_recording():
            hold_s = self.pedal.current_hold_s()
            if pedal_pressed and not self.recording_pedal_down:
                self.recording_pedal_down = True
                print(
                    "Pedal held. Release to save; "
                    f"hold {self.config.hold_delete_s:g}s to delete."
                )
            if self.recording_pedal_down:
                if hold_s is not None and hold_s >= self.config.hold_delete_s:
                    self.recording_pedal_down = False
                    delete_episode(self.recorder)
                elif hold_s is None:
                    self.recording_pedal_down = False
                    stop_episode(self.recorder, "foot_pedal")
            return

        if self.pending_start_deadline is None:
            if pedal_pressed:
                self.pending_start_deadline = (
                    time.monotonic() + self.config.ready_seconds
                )
                print(
                    "Pedal pressed. Recording starts in "
                    f"{self.config.ready_seconds:.1f}s."
                )
        elif pedal_pressed:
            print("Start cancelled by second pedal press during ready delay.")
            self.pending_start_deadline = None
        elif time.monotonic() >= self.pending_start_deadline:
            start_episode(self.config, self.recorder)
            self.pending_start_deadline = None


def run_follower_contact_ui(
    shared: SharedState,
    *,
    duration_s: float | None,
    show_camera: bool = False,
    camera_window_name: str = TASK_VIEW_WINDOW_NAME,
    on_tick=None,
    poll_windows: bool = False,
) -> str:
    """Run the read-only follower contact view until close or shutdown.

    poll_windows services OpenCV windows (such as the tracking monitor) even
    without the camera view.
    """

    import tkinter as tk

    from data_collection.collection.live_contact_widget import LiveContactWidget

    root = tk.Tk()
    root.title("Follower hand contact")
    root.minsize(1000, 500)
    widget = LiveContactWidget(root)
    widget.pack(fill="both", expand=True, padx=12, pady=12)
    cv2 = import_cv2() if show_camera or poll_windows else None
    deadline = None if duration_s is None else time.monotonic() + duration_s
    exit_reason = "ui_closed"

    def close_window() -> None:
        nonlocal exit_reason
        exit_reason = "ui_closed"
        shared.shutdown.set()

    def refresh() -> None:
        nonlocal exit_reason
        if on_tick is not None:
            on_tick()
        snapshot = shared.latest_follower_contact_snapshot()
        tactile_contacts = shared.latest_tactile_contacts()
        if snapshot is not None:
            widget.show_by_motor_id(
                snapshot.motor_ids,
                snapshot.contact_feedback,
                snapshot.normalized_joint_difference,
                tactile_contacts=tactile_contacts,
            )
        elif tactile_contacts is not None:
            widget.show(tactile_contacts=tactile_contacts)
        if cv2 is not None:
            frame = shared.latest_task_frame() if show_camera else None
            if frame is not None:
                cv2.imshow(camera_window_name, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                exit_reason = "camera_view_closed"
                shared.shutdown.set()
        if shared.shutdown.is_set():
            root.destroy()
            return
        if deadline is not None and time.monotonic() >= deadline:
            exit_reason = "duration_complete"
            shared.shutdown.set()
        if shared.shutdown.is_set():
            root.destroy()
            return
        root.after(30, refresh)

    root.protocol("WM_DELETE_WINDOW", close_window)
    root.after(0, refresh)
    try:
        root.mainloop()
    finally:
        shared.shutdown.set()
        if show_camera:
            try:
                cv2.destroyWindow(camera_window_name)
            except Exception:
                pass
    return exit_reason


def run_camera_view(
    shared: SharedState,
    *,
    duration_s: float | None,
    window_name: str,
    on_tick=None,
    show_camera: bool = True,
) -> str:
    """Show camera frames until the duration expires, shutdown, or q/Escape.

    With show_camera=False it only services other OpenCV windows (such as the
    tracking monitor drawn by on_tick) and their q/Escape keys.
    """

    cv2 = import_cv2()
    deadline = None if duration_s is None else time.monotonic() + duration_s
    reason = "duration_complete"
    try:
        while not shared.shutdown.is_set():
            if on_tick is not None:
                on_tick()
            frame = shared.latest_task_frame() if show_camera else None
            if frame is not None:
                cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                reason = "camera_view_closed"
                shared.shutdown.set()
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(0.005)
    finally:
        if show_camera:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass
    return reason


def move_piper_to_start(config: HandOnlyCollectionConfig) -> tuple[float, ...]:
    """Enable Piper, ramp it in joint space to the start joints and leave it holding.

    Returns the measured flange pose [x, y, z, roll, pitch, yaw] (m, rad).
    """

    import numpy as np

    from scripts.piper_api.piper_api import PiperAPI
    from scripts.teleop.quest_teleop import POSITION_MODE_SETTLE_S, go_home, home_warnings

    for warning in home_warnings(config.piper_start_joints_deg):
        print(f"WARNING: start pose {warning}")
    api = PiperAPI(
        firmware=config.piper_firmware,
        interface=config.piper_interface,
        channel=config.piper_channel,
        installation_pos=config.piper_installation_pos,
        payload=config.piper_payload,
    )
    api.connect()
    try:
        if not api.is_ok():
            raise RuntimeError("Piper is not responding on the CAN bus")
        input(
            "Clear the workspace, keep a hand near the e-stop, and press Enter to "
            "move Piper to the start pose..."
        )
        if not api.enable():
            raise RuntimeError("Timed out while enabling Piper")
        api.set_position_mode()
        time.sleep(POSITION_MODE_SETTLE_S)
        api.set_speed_percent(100)
        go_home(
            api,
            np.radians(config.piper_start_joints_deg),
            config.piper_start_speed_deg_s,
            50.0,
        )
        pose = tuple(float(value) for value in api.get_eef_pose())
    finally:
        # Disconnecting does not disable the arm: it keeps holding the start pose.
        api.disconnect()
    return pose


def _load_start_profile(path: Path):
    from data_collection.collection.task_start import validate_task_start
    raw = path.read_bytes()
    return validate_task_start(json.loads(raw)), hashlib.sha256(raw).hexdigest()


def run_hand_only(config: HandOnlyCollectionConfig) -> int:
    if config.mocap_ik_calibration is not None and not config.use_mocap:
        raise ValueError('Mocap IK monitoring requires mocap')
    if config.show_piper_joint_monitor and not config.use_piper:
        raise ValueError('Piper joint monitoring requires --use-piper')
    if config.show_piper_joint_monitor and not config.piper_joint_level:
        config = replace(config, piper_joint_level=True)
    gravity_comp = config.arm_mode == ARM_MODE_GRAVITY_COMP
    if config.start_profile is not None and config.start_profile_snapshot is None:
        profile, profile_hash = _load_start_profile(config.start_profile)
        config = replace(config, start_profile_snapshot=profile, start_profile_sha256=profile_hash)
    gravity_vector = None
    if gravity_comp and config.grav_comp_calib is not None:
        gravity_vector = json.loads(config.grav_comp_calib.read_text()).get("gravity", {}).get("vector_m_s2")
    if gravity_comp and getattr(config, "piper_start_joints_deg", None) is not None:
        raise ValueError("--arm-mode gravity-comp never moves the arm (no start move)")
    shared = SharedState()
    recorder = RecordingManager()
    errors: queue.Queue[WorkerError] = queue.Queue()
    pedal: FootPedalTrigger | None = None
    pedal_controller: FootPedalCollectionController | None = None

    if getattr(config, "piper_start_joints_deg", None) is not None:
        pose = move_piper_to_start(config)
        config = replace(config, piper_start_pose=pose)
        print(
            "Piper is holding the start pose; postprocessing uses it from each "
            "episode's metadata (--piper-start-pose "
            + " ".join(f"{value:.4f}" for value in pose)
            + ")."
        )

    if config.mode == DATA_COLLECTION_MODE:
        print(f"Saving episodes to {config.dataset_root}")
        print(f"Warming up for {config.warmup_time}s")
        time.sleep(config.warmup_time)
        print("Warmup done, starting data collection...")
    elif config.mode != RETARGET_ONLY_MODE:
        raise ValueError(f"Unknown hand-only mode: {config.mode!r}")

    if gravity_comp:
        print(
            "Piper logging is receive-only: gravity_comp.drag_mode commands the arm "
            "and this process sends nothing on the CAN bus."
        )
    elif config.use_piper:
        print(
            "Piper logging is passive: use the arm's built-in drag-teach control."
        )
    if config.use_quest:
        print(
            f"Waiting for Quest UDP poses on {config.quest_host}:"
            f"{config.quest_port}."
        )

    if config.use_foot_pedal and config.mode == DATA_COLLECTION_MODE:
        pedal = FootPedalTrigger(
            config.foot_pedal_device,
            key_code=config.foot_pedal_key_code,
            debounce_s=config.foot_pedal_debounce_s,
        )
        pedal.start()
        print(f"Foot pedal input: {pedal.active_device}")
        pedal_controller = FootPedalCollectionController(config, recorder, pedal)

    try:
        threads = initialize_worker(config, shared, recorder, errors)
    except Exception:
        if pedal is not None:
            pedal.stop()
        raise
    # Retain compatibility with callers that replace the initializer in tests.
    if isinstance(threads, threading.Thread) or hasattr(threads, "join"):
        threads = [threads]
    reason = "duration_complete"
    monitor = ik_monitor = joint_monitor = None
    try:
        if config.show_piper_joint_monitor:
            from data_collection.collection.piper_joint_monitor import PiperJointMonitor

            joint_monitor = PiperJointMonitor(shared, recorder=recorder)
        if config.mocap_ik_calibration is not None:
            from data_collection.collection.mocap_ik_monitor import MocapIKMonitor

            ik_monitor = MocapIKMonitor(shared, config.mocap_ik_calibration,
                                        topic=config.mocap_ik_topic, seed_deg=config.mocap_ik_seed_deg,
                                        recorder=recorder, start_name=config.mocap_ik_start,
                                        start_tolerance_deg=config.mocap_ik_start_tolerance_deg)
        if config.show_tracking_monitor:
            from data_collection.collection.tracking_monitor import (
                TrackingMonitor,
                load_monitor_calibration,
            )

            monitor = TrackingMonitor(
                shared,
                show_quest=config.use_quest,
                show_vicon=config.use_mocap,
                show_arm=gravity_comp,
                start_name=(config.start_profile_snapshot["task"]
                            if config.start_profile_snapshot is not None else
                            (None if config.no_start_guide else config.kinesthetic_start)),
                start_joints_deg=(config.start_profile_snapshot["arm"]["joints_deg"]
                                  if config.start_profile_snapshot is not None else None),
                definition_mode=config.define_start,
                gravity_vector=gravity_vector,
                start_position_tolerance_m=config.kinesthetic_start_position_tolerance_m,
                start_orientation_tolerance_deg=config.kinesthetic_start_orientation_tolerance_deg,
                passive_j6=config.grav_comp_disable_j6 or config.grav_comp_passive_j6,
                passive_j5_j6=config.grav_comp_passive_j5_j6,
                recorder=recorder,
                calibration=load_monitor_calibration(config.monitor_calibration),
            )
        ticks = [
            tick
            for tick in (
                pedal_controller.poll if pedal_controller is not None else None,
                monitor.update if monitor is not None else None,
                ik_monitor.update if ik_monitor is not None else None,
                joint_monitor.update if joint_monitor is not None else None,
            )
            if tick is not None
        ]

        def on_tick() -> None:
            for tick in ticks:
                tick()

        duration_s = None
        if config.mode == RETARGET_ONLY_MODE:
            print("Retarget-only mode active. Press Ctrl-C to stop.")
        elif config.use_foot_pedal:
            print("Foot-pedal collection active. Press q or Ctrl-C to stop.")
        else:
            duration_s = config.episode_time
            start_episode(config, recorder)
        if config.show_ui:
            reason = run_follower_contact_ui(
                shared,
                duration_s=duration_s,
                show_camera=config.add_camera,
                camera_window_name=config.task_view_window_name,
                on_tick=on_tick if ticks else None,
                poll_windows=any(view is not None for view in (monitor, ik_monitor, joint_monitor)),
            )
        elif config.add_camera or any(view is not None for view in (monitor, ik_monitor, joint_monitor)):
            reason = run_camera_view(
                shared,
                duration_s=duration_s,
                window_name=config.task_view_window_name,
                on_tick=on_tick if ticks else None,
                show_camera=config.add_camera,
            )
        elif ticks:
            while not shared.shutdown.wait(0.005):
                on_tick()
        else:
            shared.shutdown.wait(duration_s)
        if not errors.empty():
            error = errors.get_nowait()
            session = recorder.current()
            if session is not None:
                session.add_error(error)
            reason = f"{error.name}_worker_error"
            raise RuntimeError(f"{error.name} worker failed: {error.message}")
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
        if config.mode == RETARGET_ONLY_MODE:
            print("\nRetargeting stopped.")
        else:
            print("\nCollection interrupted; saving recorded samples.")
    finally:
        shared.shutdown.set()
        if pedal is not None:
            pedal.stop()
        if monitor is not None:
            monitor.close()
        if ik_monitor is not None:
            ik_monitor.close()
        if joint_monitor is not None:
            joint_monitor.close()
        for thread in threads:
            thread.join(timeout=3.0)
        if config.mode == DATA_COLLECTION_MODE:
            stop_episode(recorder, reason)
    return 0


def main() -> int:
    args = parse_args()
    profile, profile_hash = (_load_start_profile(args.start_profile.expanduser())
                             if args.start_profile is not None else (None, None))
    disabled_force_feedback_motor_ids = tuple(
        dict.fromkeys(args.disable_force_feedback_motors)
    )
    inactive_disabled_ids = sorted(
        set(disabled_force_feedback_motor_ids) - set(args.active_motors)
    )
    if inactive_disabled_ids:
        raise ValueError(
            "--disable-force-feedback-motors contains IDs that are not active: "
            f"{inactive_disabled_ids}"
        )
    add_tactile = (
        args.add_tactile
        or args.paxini_port is not None
        or args.tactile_debug_log is not None
    )
    piper_start_joints_deg = args.piper_start_joints_deg
    if args.piper_start_home:
        from scripts.teleop.quest_teleop import saved_home_deg

        piper_start_joints_deg = saved_home_deg()
    if piper_start_joints_deg is not None:
        piper_start_joints_deg = tuple(piper_start_joints_deg)
        if not all(math.isfinite(value) for value in piper_start_joints_deg):
            raise ValueError("--piper-start-joints-deg must be finite")
        if not math.isfinite(args.piper_start_speed_deg_s) or args.piper_start_speed_deg_s <= 0:
            raise ValueError("--piper-start-speed-deg-s must be finite and greater than 0")
    dataset_root = (
        args.dataset_root.expanduser()
        if args.dataset_root is not None
        else next_dataset_root(args.dataset_name.expanduser())
    )
    config = HandOnlyCollectionConfig(
        dataset_root=dataset_root,
        hand_config=args.hand_config.expanduser(),
        hand_port=args.hand_port,
        hand_measure_rate_hz=DEFAULT_HAND_MEASURE_RATE_HZ,
        episode_time=(
            args.episode_time
            if args.episode_time is not None
            else DEFAULT_DEMONSTRATION_DURATION_S
        ),
        use_foot_pedal=args.use_foot_pedal,
        warmup_time=args.warmup_time,
        mode=args.mode,
        show_ui=args.show_ui,
        add_camera=args.add_camera,
        use_piper=args.use_piper,
        arm_mode=args.arm_mode,
        grav_comp_calib=(
            args.grav_comp_calib.expanduser() if args.grav_comp_calib is not None else None
        ),
        drag_session_id=args.drag_session_id,
        kinesthetic_start=args.kinesthetic_start,
        start_profile=args.start_profile.expanduser() if args.start_profile is not None else None,
        start_profile_snapshot=profile,
        start_profile_sha256=profile_hash,
        no_start_guide=args.no_start_guide,
        define_start=args.define_start,
        kinesthetic_start_position_tolerance_m=args.kinesthetic_start_position_tolerance_m,
        kinesthetic_start_orientation_tolerance_deg=args.kinesthetic_start_orientation_tolerance_deg,
        grav_comp_disable_j6=args.grav_comp_disable_j6,
        grav_comp_passive_j6=args.grav_comp_passive_j6,
        grav_comp_passive_j5_j6=args.grav_comp_passive_j5_j6,
        hand_operating_mode=(
            KINESTHETIC_HAND_OPERATING_MODE
            if args.arm_mode == ARM_MODE_GRAVITY_COMP
            else None
        ),
        use_quest=args.use_quest,
        use_mocap=args.use_mocap,
        mocap_topics=tuple(args.mocap_topics),
        mocap_ik_calibration=(args.mocap_ik_calibration.expanduser()
                              if args.mocap_ik_calibration is not None else None),
        mocap_ik_topic=args.mocap_ik_topic,
        mocap_ik_seed_deg=tuple(args.mocap_ik_seed_deg),
        mocap_ik_start=args.mocap_ik_start,
        mocap_ik_start_tolerance_deg=args.mocap_ik_start_tolerance_deg,
        show_tracking_monitor=args.show_tracking_monitor,
        monitor_calibration=(
            args.monitor_calibration.expanduser()
            if args.monitor_calibration is not None
            else None
        ),
        quest_jump_threshold=args.quest_jump_threshold,
        piper_joint_level=args.piper_joint_level,
        show_piper_joint_monitor=args.show_piper_joint_monitor,
        piper_rate_hz=args.piper_rate_hz,
        teacher_mode_poll_hz=args.piper_teacher_mode_poll_hz,
        piper_channel=args.piper_channel,
        piper_interface=args.piper_interface,
        piper_firmware=args.piper_firmware,
        piper_installation_pos=args.piper_installation_pos,
        piper_payload=args.piper_payload,
        piper_start_joints_deg=piper_start_joints_deg,
        piper_start_speed_deg_s=args.piper_start_speed_deg_s,
        add_tactile=add_tactile,
        paxini_port=args.paxini_port,
        tactile_debug_log=(
            args.tactile_debug_log.expanduser()
            if args.tactile_debug_log is not None
            else None
        ),
        active_motors=args.active_motors,
        disabled_force_feedback_motor_ids=disabled_force_feedback_motor_ids,
        follower_hand_config=(
            args.follower_config.expanduser()
            if args.follower_config is not None
            else None
        ),
        follower_hand_port=args.follower_port,
        thumb_retarget_model=(
            args.thumb_model.expanduser()
            if args.follower_config is not None
            else None
        ),
    )
    if (config.follower_hand_config is None) != (config.follower_hand_port is None):
        raise ValueError("--follower-config and --follower-port must be provided together")
    if config.show_ui and config.follower_hand_config is None:
        raise ValueError("--show-ui requires --follower-config and --follower-port")
    if config.add_tactile and not config.show_ui and config.tactile_debug_log is None:
        raise ValueError(
            "Tactile sensing requires --show-ui or --tactile-debug-log"
        )
    if config.add_camera:
        validate_camera_settings(config)
    if config.piper_joint_level and not config.use_piper:
        raise ValueError("--piper-joint-level requires --use-piper")
    if config.arm_mode == ARM_MODE_GRAVITY_COMP:
        if not config.use_piper:
            raise ValueError("--arm-mode gravity-comp requires the Piper")
        if config.piper_installation_pos != "unchanged" or config.piper_payload != "unchanged":
            raise ValueError("--arm-mode gravity-comp must leave installation and payload unchanged")
        if config.grav_comp_calib is not None and not config.grav_comp_calib.is_file():
            raise ValueError(f"Gravity-comp calibration not found: {config.grav_comp_calib}")
    if config.use_piper:
        if config.piper_rate_hz <= 0:
            raise ValueError("PIPER_RATE_HZ must be greater than 0")
        if config.teacher_mode_poll_hz <= 0:
            raise ValueError("TEACHER_MODE_POLL_HZ must be greater than 0")
        if config.piper_firmware not in {"default", "v183", "v188"}:
            raise ValueError("PIPER_FIRMWARE must be one of: default, v183, v188")
    if config.use_quest:
        if not math.isfinite(config.quest_jump_threshold) or config.quest_jump_threshold <= 0:
            raise ValueError("--quest-jump-threshold must be finite and greater than 0")
        if not config.quest_host:
            raise ValueError("QUEST_HOST must not be empty")
        if not 1 <= config.quest_port <= 65535:
            raise ValueError("QUEST_PORT must be in range 1..65535")
        if config.quest_startup_timeout_s <= 0:
            raise ValueError("QUEST_STARTUP_TIMEOUT_S must be greater than 0")
    if config.follower_hand_port == config.hand_port:
        raise ValueError("Primary and follower hand ports must be different")
    if config.paxini_port is not None and config.paxini_port in {
        config.hand_port,
        config.follower_hand_port,
    }:
        raise ValueError("The Paxini and motor serial ports must be different")
    return run_hand_only(config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
