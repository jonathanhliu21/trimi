from __future__ import annotations

import time

from scripts.piper_api.record_trajectory import (
    JOINT_LIMIT_WARNING_REPEAT_NS,
    joint_limit_warning,
    read_joint_limits,
)

from .config import (
    ARM_MODE_GRAVITY_COMP,
    EEF_FRESH_S,
    PIPER_CURRENT_FIELDS,
    PIPER_ENABLE_FIELDS,
    PIPER_JOINT_FIELDS,
    PIPER_TORQUE_FIELDS,
    PIPER_VELOCITY_FIELDS,
    RAW_EEF_FIELDS,
    HandOnlyCollectionConfig,
)
from .episode import EpisodeBuffer, RecordingManager
from .shared_state import ArmHealth, SharedState


PIPER_LIMIT_WARNING_INTERVAL_S = 2.0
# Warn when no new joint feedback frame has arrived for this long.
FEEDBACK_GAP_WARNING_S = 0.5
# A CAN receive time further than this from the host clock is not on it.
RX_CLOCK_SANITY_S = 5.0
PIPER_KINEMATIC_LIMIT_STATUSES = (
    "TARGET_POS_EXCEEDS_LIMIT",
    "NO_SOLUTION",
    "SINGULARITY_POINT",
)


def piper_workspace_warning(state: dict[str, object]) -> str | None:
    """Describe Piper workspace/joint-limit feedback, if present."""

    arm_status = str(state.get("arm_status", ""))
    reasons = [
        f"arm_status={arm_status}"
        for status in PIPER_KINEMATIC_LIMIT_STATUSES
        if status in arm_status
    ]
    try:
        err_code = int(state.get("err_code", 0))
    except (TypeError, ValueError):
        err_code = 0
    limited_joints = [
        joint for joint in range(1, 7) if err_code & (1 << (joint + 7))
    ]
    if limited_joints:
        reasons.append(f"joint_angle_limit={limited_joints}")
    if not reasons:
        return None
    return (
        "WARNING: Piper reached a workspace/kinematic limit ("
        + ", ".join(reasons)
        + "). Move the arm away from the boundary before recording/replay."
    )


def arm_health_problem(enable: list[bool | None], err_code: int | None) -> str:
    """Empty while every drive is enabled with no error code, else what is wrong."""

    problems = []
    off = [joint for joint, enabled in enumerate(enable, start=1) if enabled is False]
    unknown = [joint for joint, enabled in enumerate(enable, start=1) if enabled is None]
    if off:
        problems.append("DRIVES OFF J" + ",".join(map(str, off)))
    if unknown:
        problems.append("NO DRIVER STATE J" + ",".join(map(str, unknown)))
    if err_code is None:
        problems.append("NO ARM STATUS")
    elif err_code:
        problems.append(f"ERR 0x{err_code:04X}")
    return " ".join(problems)


def _fmt(value: float) -> str:
    return f"{float(value):.9f}"


def gravity_comp_sample(joints, motors, enable, status, eef, *, now_ns: int, wall_s: float):
    """One piper.csv row (without the per-episode index fields) and its health problem.

    Every row has the same columns; values that have not arrived yet are blank.
    """

    rx_s = float(joints.timestamp)
    row: dict[str, object] = {
        # CAN receive time of the joint frame (the SDK's timestamp, host wall clock)
        # and the same instant on the host monotonic clock used by the other streams.
        "robot_timestamp_s": f"{rx_s:.6f}",
        "joint_rx_monotonic_ns": (
            now_ns - int(round((wall_s - rx_s) * 1e9))
            if abs(wall_s - rx_s) <= RX_CLOCK_SANITY_S
            else ""
        ),
        "robot_hz": f"{float(joints.hz):.3f}",
    }
    row.update({name: _fmt(value) for name, value in zip(PIPER_JOINT_FIELDS, joints.joints)})
    row["motor_timestamp_s"] = f"{motors.timestamp:.6f}" if motors is not None else ""
    for names, attr in (
        (PIPER_VELOCITY_FIELDS, "velocities"),
        (PIPER_CURRENT_FIELDS, "currents"),
        (PIPER_TORQUE_FIELDS, "torques"),
    ):
        values = [_fmt(value) for value in getattr(motors, attr)] if motors is not None else [""] * 6
        row.update(zip(names, values))
    row.update(
        {
            name: "" if enabled is None else int(bool(enabled))
            for name, enabled in zip(PIPER_ENABLE_FIELDS, enable)
        }
    )
    row.update(
        {
            "ctrl_mode": status.ctrl_mode if status is not None else "",
            "arm_status": status.arm_status if status is not None else "",
            "mode_feedback": status.mode_feedback if status is not None else "",
            "err_code": status.err_code if status is not None else "",
        }
    )
    problem = arm_health_problem(enable, None if status is None else status.err_code)
    row["arm_ok"] = int(not problem)
    fresh = eef is not None and abs(float(eef.timestamp) - rx_s) <= EEF_FRESH_S
    row.update(zip(RAW_EEF_FIELDS, [_fmt(value) for value in eef.pose] if fresh else [""] * 6))
    row["eef_timestamp_s"] = f"{eef.timestamp:.6f}" if eef is not None else ""
    row["eef_hz"] = f"{eef.hz:.3f}" if eef is not None else ""
    return row, problem


def run_gravity_comp_listener(
    config: HandOnlyCollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    *,
    init_done,
) -> None:
    """Receive-only Piper logger for kinesthetic teaching under gravity comp.

    gravity_comp.drag_mode, in another process, is the only writer on the bus.
    This worker sends nothing: ReceiveOnlyPiperAPI only has cached reads. It polls
    the SDK's cache at piper_rate_hz and keeps one row per new joint feedback
    frame, deduplicated on that frame's CAN receive time.
    """
    from scripts.piper_api.piper_api import ReceiveOnlyPiperAPI

    api = ReceiveOnlyPiperAPI(
        firmware=config.piper_firmware,
        interface=config.piper_interface,
        channel=config.piper_channel,
    )
    try:
        api.connect()
        print("Piper listener connected:", api.is_ok(), "receive FPS:", api.get_fps())
        if not api.is_ok():
            raise RuntimeError(
                f"No Piper feedback on {config.piper_channel} (receive FPS=0). "
                "Check that the arm is powered on and the CAN cable is connected."
            )
        print(
            f"[piper_worker] Receive-only on {config.piper_channel}: drag mode "
            "commands the arm; this process sends nothing."
        )
        init_done.set()

        poll_period_s = 1.0 / config.piper_rate_hz
        last_stamp: float | None = None
        last_frame_ns = time.monotonic_ns()
        gap_warned = False
        last_problem: str | None = None
        last_limit_warning: str | None = None
        last_limit_warning_t = float("-inf")
        sample_session: EpisodeBuffer | None = None
        sample_idx = 0
        while not shared.shutdown.is_set():
            joints = api.get_joint_state()
            now_ns = time.monotonic_ns()
            if joints is None or joints.timestamp == last_stamp:
                if not gap_warned and now_ns - last_frame_ns > FEEDBACK_GAP_WARNING_S * 1e9:
                    print(
                        f"WARNING: no new Piper joint feedback for "
                        f"{(now_ns - last_frame_ns) / 1e9:.1f}s",
                        flush=True,
                    )
                    gap_warned = True
                shared.shutdown.wait(poll_period_s)
                continue

            last_stamp = joints.timestamp
            last_frame_ns = now_ns
            gap_warned = False
            wall_s = time.time()
            status = api.get_arm_status()
            row, problem = gravity_comp_sample(
                joints,
                api.get_motor_state(),
                api.get_driver_enable_status(),
                status,
                api.get_eef_state(),
                now_ns=now_ns,
                wall_s=wall_s,
            )
            shared.set_arm_health(ArmHealth(now_ns, not problem, problem, float(joints.hz)))
            session = recorder.current()
            if problem != last_problem:
                if problem:
                    print(
                        f"WARNING: Piper arm not OK: {problem}"
                        + (" (this episode will be flagged)" if session is not None else ""),
                        flush=True,
                    )
                elif last_problem is not None:
                    print("Piper arm OK again: all drives enabled, no error code.", flush=True)
                last_problem = problem
            if status is not None:
                limit_warning = piper_workspace_warning(
                    {"arm_status": status.arm_status, "err_code": status.err_code}
                )
                now_s = now_ns / 1e9
                if limit_warning is not None and (
                    limit_warning != last_limit_warning
                    or now_s - last_limit_warning_t >= PIPER_LIMIT_WARNING_INTERVAL_S
                ):
                    print(limit_warning, flush=True)
                    last_limit_warning_t = now_s
                last_limit_warning = limit_warning

            if session is not None:
                if session is not sample_session:
                    sample_session = session
                    sample_idx = 0
                session.append_piper(
                    {
                        "sample_idx": sample_idx,
                        "monotonic_ns": now_ns,
                        "wall_time_s": f"{wall_s:.6f}",
                        "elapsed_s": f"{session.elapsed_s(now_ns):.9f}",
                        **row,
                    }
                )
                sample_idx += 1
            shared.shutdown.wait(poll_period_s)
    finally:
        api.disconnect()


def run_piper_worker(
    config: HandOnlyCollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    *,
    init_done,
) -> None:
    """Passive Piper logger: no MIT/backdrive/cuRobo control, ever.

    The arm is expected to be moved with its own built-in drag-teach button.
    This worker only connects, polls teach_status + EEF state, and logs —
    it never calls enable()/set_mit_mode()/sends a motion command.

    With arm_mode gravity-comp it hands over to run_gravity_comp_listener, which
    sends nothing at all.
    """
    if getattr(config, "arm_mode", None) == ARM_MODE_GRAVITY_COMP:
        run_gravity_comp_listener(config, shared, recorder, init_done=init_done)
        return

    from scripts.piper_api.piper_api import PiperAPI

    api = PiperAPI(
        firmware=config.piper_firmware,
        interface=config.piper_interface,
        channel=config.piper_channel,
        installation_pos=config.piper_installation_pos,
        payload=config.piper_payload,
    )
    api.connect()
    print("Piper connected:", api.is_ok(), "receive FPS:", api.get_fps())
    if not api.is_ok():
        raise RuntimeError(
            "Piper arm is not responding on the CAN bus (receive FPS=0). "
            "Check that the arm is powered on and the CAN cable is connected."
        )
    print(
        "[piper_worker] Passive logging only — use the arm's built-in "
        "drag-teach button to move it."
    )
    joint_limits = (
        read_joint_limits(api)
        if getattr(config, "piper_joint_level", False)
        else {}
    )

    init_done.set()

    sample_period_ns = int(round(1_000_000_000 / config.piper_rate_hz))
    teacher_period_ns = int(round(1_000_000_000 / config.teacher_mode_poll_hz))
    now_ns = time.monotonic_ns()
    next_sample_ns = now_ns
    next_teacher_ns = now_ns
    sample_idx = 0
    sample_session: EpisodeBuffer | None = None
    teach_status: object = None
    last_limit_warning: str | None = None
    last_limit_warning_t = float("-inf")
    last_joint_limit_state: dict[int, str] = {}
    last_joint_limit_warning_ns: dict[int, int] = {}
    try:
        while not shared.shutdown.is_set():
            now_ns = time.monotonic_ns()

            if now_ns >= next_teacher_ns:
                state = api.get_state()
                teach_status = state.get("teach_status")
                limit_warning = piper_workspace_warning(state)
                now_s = now_ns / 1e9
                if limit_warning is not None and (
                    limit_warning != last_limit_warning
                    or now_s - last_limit_warning_t >= PIPER_LIMIT_WARNING_INTERVAL_S
                ):
                    print(limit_warning, flush=True)
                    last_limit_warning = limit_warning
                    last_limit_warning_t = now_s
                elif limit_warning is None:
                    last_limit_warning = None
                next_teacher_ns += teacher_period_ns
                while next_teacher_ns < now_ns:
                    next_teacher_ns += teacher_period_ns

            if now_ns >= next_sample_ns:
                session = recorder.current()
                eef = api.get_eef_state()
                joints = (
                    api.get_joint_state()
                    if session is not None and getattr(config, "piper_joint_level", False)
                    else None
                )
                sample_ns = time.monotonic_ns()
                if (
                    session is not None
                    and eef is not None
                    and (
                        not getattr(config, "piper_joint_level", False)
                        or joints is not None
                    )
                ):
                    if session is not sample_session:
                        sample_session = session
                        sample_idx = 0
                    row: dict[str, object] = {
                        "sample_idx": sample_idx,
                        "monotonic_ns": sample_ns,
                        "wall_time_s": f"{time.time():.6f}",
                        "elapsed_s": f"{session.elapsed_s(sample_ns):.9f}",
                        "robot_timestamp_s": f"{eef.timestamp:.6f}",
                        "robot_hz": f"{eef.hz:.3f}",
                        "teach_status": str(teach_status),
                    }
                    row.update(
                        {
                            name: f"{float(value):.9f}"
                            for name, value in zip(RAW_EEF_FIELDS, eef.pose)
                        }
                    )
                    if joints is not None:
                        for joint, angle in enumerate(joints.joints, start=1):
                            if joint not in joint_limits:
                                continue
                            warning = joint_limit_warning(
                                joint, float(angle), *joint_limits[joint]
                            )
                            if warning is None:
                                last_joint_limit_state.pop(joint, None)
                                continue
                            state, message = warning
                            if (
                                state != last_joint_limit_state.get(joint)
                                or sample_ns - last_joint_limit_warning_ns.get(joint, 0)
                                >= JOINT_LIMIT_WARNING_REPEAT_NS
                            ):
                                print(message, flush=True)
                                last_joint_limit_warning_ns[joint] = sample_ns
                            last_joint_limit_state[joint] = state
                        row.update(
                            {
                                name: f"{float(value):.9f}"
                                for name, value in zip(PIPER_JOINT_FIELDS, joints.joints)
                            }
                        )
                    session.append_piper(row)
                    sample_idx += 1
                next_sample_ns += sample_period_ns
                while next_sample_ns < sample_ns:
                    next_sample_ns += sample_period_ns

            next_wake_ns = min(next_sample_ns, next_teacher_ns)
            wait_s = (next_wake_ns - time.monotonic_ns()) / 1e9
            if wait_s > 0:
                shared.shutdown.wait(min(wait_s, 0.01))
    finally:
        api.disconnect()
