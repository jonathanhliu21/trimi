"""Live 3D view of the Quest and Vicon poses with a per-stream signal status.

With show_arm, an ARM box shows the gravity-comp arm feedback. With a saved
task start, START turns green when the flange is within 10 cm and 45 degrees
of that start; feedback health is shown as secondary text.

Each stream is drawn relative to its pose when the view started (or was last
re-centered), so both start at the origin. Displacements are shown in the
Piper base axes when a calibration provides the mapping (linear_matrix for the
Quest, vicon_reference for Vicon) and in the tracker's own axes otherwise.
Drag to rotate, right-drag or scroll to zoom, double-click to re-center.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .quest_worker import QUEST_TRACKER_NAME
from .shared_state import ArmHealth, SharedState, TrackerSample
from .utils import import_cv2
from .config import PIPER_START_JOINTS_DEG

TRACKING_WINDOW_NAME = "trimorph_tracking"
# Vicon over Wi-Fi arrives in bursts with gaps of up to ~0.3 s, so allow 0.5 s.
SIGNAL_TIMEOUT_S = 0.5
# Without positional tracking the Quest keeps resending its last position, while
# a tracked, hand-held controller moves by more than 0.1 mm every second.
FROZEN_WINDOW_S = 1.0
FROZEN_SPAN_M = 1e-4
TRAIL_S = 10.0
GRID_HALF_M = 0.5
GRID_STEP_M = 0.1
TRIAD_M = 0.06

# BGR colors.
OK_COLOR = (60, 170, 60)
FAIL_COLOR = (40, 40, 210)
QUEST_COLOR = (0, 165, 255)
VICON_COLOR = (255, 200, 0)
ARM_COLOR = (200, 120, 255)
CURRENT_ARM_COLOR = (0, 190, 255)
START_ARM_COLOR = (255, 210, 80)
AXIS_COLORS = ((60, 60, 230), (60, 200, 60), (230, 120, 40))


@dataclass(frozen=True)
class StreamStatus:
    ok: bool
    text: str


def stream_status(
    now_ns: int,
    sample: TrackerSample | None,
    recent_positions: np.ndarray | None,
    *,
    check_frozen: bool,
    timeout_s: float = SIGNAL_TIMEOUT_S,
) -> StreamStatus:
    """Green only when samples are fresh, the object is tracked, and (for the Quest) moving.

    recent_positions holds the positions seen over the last FROZEN_WINDOW_S, or
    None until that window has filled.
    """

    if sample is None:
        return StreamStatus(False, "NO DATA")
    age_s = (now_ns - sample.monotonic_ns) / 1e9
    if age_s > timeout_s:
        return StreamStatus(False, f"NO SIGNAL {age_s:.1f}s")
    if sample.lost:
        return StreamStatus(False, "OBJECT LOST")
    if (
        check_frozen
        and recent_positions is not None
        and np.ptp(recent_positions, axis=0).max() < FROZEN_SPAN_M
    ):
        return StreamStatus(False, "POSITION FROZEN")
    return StreamStatus(True, "OK")


def arm_status(
    now_ns: int, health: ArmHealth | None, *, timeout_s: float = SIGNAL_TIMEOUT_S
) -> StreamStatus:
    """Green only while arm feedback is fresh, every drive is enabled and there is no error code."""

    if health is None:
        return StreamStatus(False, "NO DATA")
    age_s = (now_ns - health.monotonic_ns) / 1e9
    if age_s > timeout_s:
        return StreamStatus(False, f"NO SIGNAL {age_s:.1f}s")
    if not health.ok:
        return StreamStatus(False, health.problem)
    return StreamStatus(True, f"OK {health.rate_hz:.0f} Hz")


def start_guidance(now_ns, health, name, position_tolerance_m=.10,
                   orientation_tolerance_deg=45., *, start_joints_deg=None):
    """Compare a fresh healthy flange with a saved start; never command motion."""
    from .arm_geometry import flange_pose_error

    status = arm_status(now_ns, health)
    if health is None or (now_ns - health.monotonic_ns) / 1e9 > SIGNAL_TIMEOUT_S:
        return StreamStatus(False, status.text), None
    if health.joints is None:
        return StreamStatus(False, "NO JOINTS"), None
    joints = np.asarray(health.joints)
    if joints.shape != (6,) or not np.isfinite(joints).all():
        return StreamStatus(False, "INVALID JOINTS"), None
    target = np.asarray(PIPER_START_JOINTS_DEG[name] if start_joints_deg is None
                        else start_joints_deg, dtype=float)
    if target.shape != (6,) or not np.isfinite(target).all():
        return StreamStatus(False, "INVALID START"), None
    tolerances = np.asarray((position_tolerance_m, orientation_tolerance_deg), dtype=float)
    if not np.isfinite(tolerances).all() or np.any(tolerances <= 0):
        return StreamStatus(False, "INVALID START TOLERANCE"), None
    error = flange_pose_error(joints, np.radians(target))
    # Numeric slack only at the exact threshold (SO(3) conversion roundoff).
    at_start = (status.ok and error[0] <= position_tolerance_m + 1e-12
                and error[1] <= orientation_tolerance_deg + 1e-9)
    return StreamStatus(at_start, f"{'AT START' if at_start else 'MOVE BY HAND TO'} {name}"), error


def current_arm_frames(now_ns: int, health: ArmHealth | None):
    """Fresh measured geometry, even with motors disabled; never retain stale poses."""
    from .arm_geometry import piper_frames

    if health is None:
        return None, "CURRENT hidden: no arm feedback"
    age_s = (now_ns - health.monotonic_ns) / 1e9
    if age_s > SIGNAL_TIMEOUT_S or age_s < -SIGNAL_TIMEOUT_S:
        return None, f"CURRENT hidden: stale feedback ({age_s:.1f}s)"
    if health.joints is None:
        return None, "CURRENT hidden: no measured joints"
    try:
        return piper_frames(health.joints), "CURRENT: measured joints"
    except ValueError:
        return None, "CURRENT hidden: invalid measured joints"


@dataclass(frozen=True)
class _ArmBox:
    """Status-bar entry for the arm; it has no pose in the 3D view."""

    label: str = "ARM"
    color: tuple[int, int, int] = ARM_COLOR
    pose: None = None


@dataclass(frozen=True)
class DisplayFrame:
    """Maps native displacements (linear) and rotations (conjugated by rotation) to display axes."""

    linear: np.ndarray
    rotation: np.ndarray
    label: str


def display_frames(calibration: dict | None) -> tuple[DisplayFrame, DisplayFrame]:
    """Quest and Vicon display frames: Piper base axes where the calibration maps them."""

    native = DisplayFrame(np.eye(3), np.eye(3), "own axes")
    quest = vicon = native
    if calibration:
        if "linear_matrix" in calibration:
            quest = DisplayFrame(
                np.asarray(calibration["linear_matrix"], dtype=float),
                np.asarray(calibration["orientation"]["rotation_matrix"], dtype=float),
                "Piper axes",
            )
        reference = calibration.get("vicon_reference") or {}
        if reference:
            world = np.asarray(next(iter(reference.values()))["vicon_to_piper_rotation"], dtype=float)
            vicon = DisplayFrame(world, world, "Piper axes")
    return quest, vicon


def projection_coordinates(points, yaw: float, pitch: float) -> np.ndarray:
    """Continuous orthographic coordinates, before screen scale and translation."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    view = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array(
        [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]
    )
    rotated = np.atleast_2d(np.asarray(points, dtype=float)) @ view.T
    return np.column_stack((rotated[:, 0], -rotated[:, 2]))


def project(points, yaw: float, pitch: float, scale: float, center) -> np.ndarray:
    """Orthographic view of display points (m): yaw about Z, then pitch toward the viewer."""
    return (np.asarray(center) + scale * projection_coordinates(points, yaw, pitch)).round().astype(np.int32)


class _Stream:
    def __init__(self, name: str, label: str, color, frame: DisplayFrame, check_frozen: bool):
        self.name, self.label, self.color, self.frame = name, label, color, frame
        self.check_frozen = check_frozen
        self.anchor: tuple[np.ndarray, Rotation] | None = None
        self.trail: deque[tuple[int, np.ndarray]] = deque()
        self.recent: deque[tuple[int, np.ndarray]] = deque()
        self.counts: deque[tuple[int, int]] = deque()
        self.last_count = 0
        self.pose: tuple[np.ndarray, np.ndarray] | None = None

    def recenter(self) -> None:
        self.anchor, self.pose = None, None
        self.trail.clear()

    def update(self, now_ns: int, sample: TrackerSample | None) -> StreamStatus:
        if sample is not None:
            position = np.asarray(sample.position)
            self.recent.append((now_ns, position))
            self.counts.append((now_ns, sample.count))
            if sample.count != self.last_count and not sample.lost:
                rotation = Rotation.from_quat(sample.quaternion)
                if self.anchor is None:
                    self.anchor = (position, rotation)
                origin, start = self.anchor
                display = self.frame.linear @ (position - origin)
                relative = (rotation * start.inv()).as_matrix()
                self.pose = (display, self.frame.rotation @ relative @ self.frame.rotation.T)
                self.trail.append((now_ns, display))
            self.last_count = sample.count
        for queue, keep_ns in (
            (self.trail, TRAIL_S * 1e9),
            (self.recent, FROZEN_WINDOW_S * 1e9),
            (self.counts, 1e9),
        ):
            while queue and now_ns - queue[0][0] > keep_ns:
                queue.popleft()
        window_full = bool(self.recent) and now_ns - self.recent[0][0] >= 0.9 * FROZEN_WINDOW_S * 1e9
        recent = np.array([position for _, position in self.recent]) if window_full else None
        status = stream_status(now_ns, sample, recent, check_frozen=self.check_frozen)
        if status.ok and len(self.counts) > 1:
            (t0, c0), (t1, c1) = self.counts[0], self.counts[-1]
            if t1 > t0:
                status = StreamStatus(True, f"OK {1e9 * (c1 - c0) / (t1 - t0):.0f} Hz")
        return status


class TrackingMonitor:
    """OpenCV window drawn from the main thread; call update() from its loop."""

    def __init__(
        self,
        shared: SharedState,
        *,
        show_quest: bool,
        show_vicon: bool,
        show_arm: bool = False,
        start_name: str | None = None,
        start_joints_deg=None,
        definition_mode: bool = False,
        start_position_tolerance_m: float = .10,
        start_orientation_tolerance_deg: float = 45.,
        passive_j6: bool = False,
        passive_j5_j6: bool = False,
        recorder=None,
        calibration: dict | None = None,
        gravity_vector=None,
        window_name: str = TRACKING_WINDOW_NAME,
        size: tuple[int, int] = (760, 620),
        cv2=None,
    ) -> None:
        self.cv2 = cv2 or import_cv2()
        self.shared, self.recorder, self.window_name = shared, recorder, window_name
        self.width, self.height = size
        self.show_quest, self.show_vicon, self.show_arm = show_quest, show_vicon, show_arm
        self.definition_mode = bool(definition_mode)
        self.task_label = start_name
        self.start_name = None if self.definition_mode else start_name
        self.start_position_tolerance_m = start_position_tolerance_m
        self.start_orientation_tolerance_deg = start_orientation_tolerance_deg
        self.start_joints_deg = None
        if not self.definition_mode and start_name is not None:
            self.start_joints_deg = np.asarray(
                PIPER_START_JOINTS_DEG[start_name] if start_joints_deg is None else start_joints_deg,
                dtype=float).copy()
            if self.start_joints_deg.shape != (6,) or not np.isfinite(self.start_joints_deg).all():
                raise ValueError("Start geometry needs six finite joint angles in degrees")
        elif start_joints_deg is not None and not self.definition_mode:
            raise ValueError("Explicit start joints require a start_name task label")
        self.passive_j6 = passive_j6
        self.passive_j5_j6 = passive_j5_j6
        self.quest_frame, self.vicon_frame = display_frames(calibration)
        from .arm_geometry import gravity_up_rotation, piper_frames
        self.arm_view_rotation = (np.eye(3) if gravity_vector is None
                                  else gravity_up_rotation(gravity_vector))
        self.arm_view_label = ("Piper base coordinates" if gravity_vector is None
                               else "Gravity-up view; fixed Piper base axes")
        self.start_frames = (piper_frames(np.radians(self.start_joints_deg))
                             if show_arm and self.start_joints_deg is not None else None)
        self.streams: dict[str, _Stream] = {}
        self.yaw, self.pitch, self.scale = (math.radians(35 if show_arm else -35),
                                           math.radians(25), 600.0 if show_arm else 900.0)
        self.arm_center = (self.width // 3, int(self.height * .46))
        self._arm_fit_pending = True
        self._arm_fit_has_current = False
        self._drag: tuple[int, int, int, float, float, float] | None = None
        self.cv2.namedWindow(window_name)
        self.cv2.setMouseCallback(window_name, self._on_mouse)

    def _stream(self, name: str) -> _Stream:
        if name not in self.streams:
            if name == QUEST_TRACKER_NAME:
                stream = _Stream(name, "QUEST", QUEST_COLOR, self.quest_frame, check_frozen=True)
            else:
                stream = _Stream(name, f"VICON {name.rstrip('/').split('/')[-1]}", VICON_COLOR,
                                 self.vicon_frame, check_frozen=False)
            self.streams[name] = stream
        return self.streams[name]

    def _on_mouse(self, event, x, y, flags, _param) -> None:
        cv2 = self.cv2
        if event == cv2.EVENT_LBUTTONDBLCLK:
            for stream in self.streams.values():
                stream.recenter()
            self._arm_fit_pending = True
            self._arm_fit_has_current = False
        elif event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self._arm_fit_pending = False
            self._arm_fit_has_current = True
            self._drag = (event, x, y, self.yaw, self.pitch, self.scale)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self._drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            button, x0, y0, yaw, pitch, scale = self._drag
            if button == cv2.EVENT_LBUTTONDOWN:
                self.yaw = yaw + 0.01 * (x - x0)
                self.pitch = float(np.clip(pitch + 0.01 * (y - y0), -1.55, 1.55))
            else:
                self.scale = float(np.clip(scale * math.exp(-0.01 * (y - y0)), 50.0, 20000.0))
        elif event == getattr(cv2, "EVENT_MOUSEWHEEL", None):
            self._arm_fit_pending = False
            self._arm_fit_has_current = True
            self.scale = float(np.clip(self.scale * (1.15 if flags > 0 else 1 / 1.15), 50.0, 20000.0))

    def render(self, now_ns: int | None = None) -> np.ndarray:
        cv2 = self.cv2
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        samples = self.shared.latest_tracker_poses()
        names = [QUEST_TRACKER_NAME] if self.show_quest else []
        if self.show_vicon:
            names += sorted(name for name in samples if name != QUEST_TRACKER_NAME)
        statuses = [(self._stream(name), self._stream(name).update(now_ns, samples.get(name)))
                    for name in names]
        if self.show_vicon and not any(name != QUEST_TRACKER_NAME for name in names):
            statuses.append((None, StreamStatus(False, "NO DATA")))
        guide_status = start_error = health_status = None
        if self.show_arm:
            health = self.shared.latest_arm_health()
            health_status = arm_status(now_ns, health)
            if self.definition_mode:
                guide_status = StreamStatus(False, "NO TASK START SAVED")
                statuses.append((_ArmBox(label="DEFINE START"), guide_status))
            elif self.start_name is None:
                statuses.append((_ArmBox(), health_status))
            else:
                guide_status, start_error = start_guidance(
                    now_ns, health, self.start_name, self.start_position_tolerance_m,
                    self.start_orientation_tolerance_deg,
                    start_joints_deg=self.start_joints_deg)
                statuses.append((_ArmBox(label="START"), StreamStatus(
                    guide_status.ok, "AT START" if guide_status.ok else "NOT AT START")))

            current, geometry_status = current_arm_frames(now_ns, health)
            if self._arm_fit_pending or (current is not None and not self._arm_fit_has_current):
                fit_frames = [frame for frame in (self.start_frames, current) if frame is not None]
                if fit_frames:
                    self._fit_arm_view(fit_frames)
                    self._arm_fit_pending = False
                    self._arm_fit_has_current = current is not None

        image = np.full((self.height, self.width, 3), 28, dtype=np.uint8)
        center = (self.arm_center if self.show_arm
                  else (self.width // 2, (self.height + 70) // 2))
        grid = np.arange(-GRID_HALF_M, GRID_HALF_M + 1e-9, GRID_STEP_M)
        for value in grid:
            for line in (((value, -GRID_HALF_M, 0), (value, GRID_HALF_M, 0)),
                         ((-GRID_HALF_M, value, 0), (GRID_HALF_M, value, 0))):
                a, b = project(line, self.yaw, self.pitch, self.scale, center)
                cv2.line(image, tuple(a), tuple(b), (55, 55, 55), 1, cv2.LINE_AA)
        origin = project((0, 0, 0), self.yaw, self.pitch, self.scale, center)[0]
        for axis, (color, name) in enumerate(zip(AXIS_COLORS, "XYZ")):
            vector = np.eye(3)[axis] * 0.12
            if self.show_arm:
                vector = self.arm_view_rotation @ vector
            tip = project(vector, self.yaw, self.pitch, self.scale, center)[0]
            cv2.line(image, tuple(origin), tuple(tip), color, 2, cv2.LINE_AA)
            cv2.putText(image, name, tuple(tip + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        for stream, _status in statuses:
            if stream is None or stream.pose is None:
                continue
            if len(stream.trail) > 1:
                points = project([p for _, p in stream.trail], self.yaw, self.pitch, self.scale, center)
                cv2.polylines(image, [points], False, stream.color, 2, cv2.LINE_AA)
            position, rotation = stream.pose
            tip0 = project(position, self.yaw, self.pitch, self.scale, center)[0]
            for axis, color in enumerate(AXIS_COLORS):
                tip = project(position + TRIAD_M * rotation[:, axis], self.yaw, self.pitch,
                              self.scale, center)[0]
                cv2.line(image, tuple(tip0), tuple(tip), color, 2, cv2.LINE_AA)
            cv2.circle(image, tuple(tip0), 5, stream.color, -1, cv2.LINE_AA)

        if self.show_arm:
            if self.start_frames is not None:
                self._draw_arm(image, self.start_frames, center, START_ARM_COLOR,
                               prefix="S", dashed=True)
            if current is not None:
                self._draw_arm(image, current, center, CURRENT_ARM_COLOR,
                               prefix="C", dashed=False)
            self._draw_arm_legend(image, geometry_status)

        self._draw_status_bar(image, statuses, now_ns)
        if self.show_arm and (self.start_name is not None or self.definition_mode):
            color = OK_COLOR if guide_status.ok else FAIL_COLOR
            cv2.rectangle(image, (12, 78), (self.width - 12, 180), color, -1)
            if self.definition_mode:
                label = "DEFINE START: NO TASK START SAVED"
                if self.task_label:
                    label += f" ({self.task_label})"
            else:
                label = f"{'AT START' if guide_status.ok else 'NOT AT START'} {self.start_name}"
                label += (" (J5/J6 passive: align manually)" if self.passive_j5_j6 else
                          (" (J6 passive: align manually)" if self.passive_j6 else ""))
            cv2.putText(image, label, (22, 101), cv2.FONT_HERSHEY_SIMPLEX, .6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            if start_error is not None:
                text = f"End effector: {start_error[0] * 100:.1f} cm away  |  {start_error[1]:.1f} deg rotation"
                cv2.putText(image, text, (22, 126), cv2.FONT_HERSHEY_SIMPLEX, .55,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, f"Ready within {self.start_position_tolerance_m * 100:g} cm AND "
                            f"{self.start_orientation_tolerance_deg:g} deg total rotation (flange frame)",
                            (22, 146), cv2.FONT_HERSHEY_SIMPLEX, .4, (225, 225, 225), 1, cv2.LINE_AA)
            elif self.definition_mode:
                cv2.putText(image, "Move by hand to the task's starting pose before saving its start profile.",
                            (22, 132), cv2.FONT_HERSHEY_SIMPLEX, .45,
                            (225, 225, 225), 1, cv2.LINE_AA)
            health_text = f"ARM FEEDBACK: {health_status.text}"
            if start_error is not None and (
                    start_error[0] > self.start_position_tolerance_m + 1e-12
                    or start_error[1] > self.start_orientation_tolerance_deg + 1e-9):
                health_text += " | Move by hand to the start before recording"
            cv2.putText(image, health_text, (22, 170), cv2.FONT_HERSHEY_SIMPLEX, .45,
                        (225, 225, 225), 1, cv2.LINE_AA)
        legend = [f"{stream.label}: {stream.frame.label}" for stream, _ in statuses
                  if isinstance(stream, _Stream)]
        cv2.putText(image, "   ".join(legend), (12, self.height - 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(image, "drag: rotate   right-drag/scroll: zoom   double-click: re-center   grid 10 cm",
                    (12, self.height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        return image

    def _fit_arm_view(self, frames_list):
        """Fit both arm skeletons and flange axes once; mouse edits remain manual."""
        points = [np.zeros((1, 3)), np.eye(3) * .12]
        for frames in frames_list:
            points.append(frames[:, :3, 3])
            flange = frames[-1]
            points.append(flange[:3, 3] + .085 * flange[:3, :3].T)
        coordinates = projection_coordinates(np.vstack(points) @ self.arm_view_rotation.T,
                                             self.yaw, self.pitch)
        low, high = coordinates.min(axis=0), coordinates.max(axis=0)
        top = 258 if self.start_name is not None or self.definition_mode else 145
        left, right, bottom = 50, self.width - 70, self.height - 90
        available = np.maximum((right - left, bottom - top), 40)
        span = np.maximum(high - low, 1e-6)
        self.scale = float(np.clip(np.min(available / span), 50, 20000))
        midpoint = (low + high) / 2
        self.arm_center = tuple(np.asarray(((left + right) / 2, (top + bottom) / 2))
                                - self.scale * midpoint)

    def _draw_arm(self, image, frames, center, color, *, prefix, dashed):
        """Link skeleton and flange orientation; J6 remains visible through its axes."""
        cv2 = self.cv2
        positions = frames[:, :3, 3]
        points = project(positions @ self.arm_view_rotation.T, self.yaw, self.pitch,
                         self.scale, center)

        def line(a, b, line_color, thickness):
            if not dashed:
                cv2.line(image, tuple(a), tuple(b), line_color, thickness, cv2.LINE_AA)
                return
            vector = np.asarray(b, dtype=float) - a
            length = float(np.linalg.norm(vector))
            if length < 1:
                return
            for start in np.arange(0, length, 15):
                p = np.rint(a + vector * start / length).astype(int)
                q = np.rint(a + vector * min(start + 9, length) / length).astype(int)
                cv2.line(image, tuple(p), tuple(q), line_color, thickness, cv2.LINE_AA)

        for a, b in zip(points, points[1:]):
            line(a, b, color, 3 if dashed else 5)
        groups = []
        for j in range(1, 7):
            for group in groups:
                if np.linalg.norm(positions[j] - positions[group[0]]) < 1e-8:
                    group.append(j)
                    break
            else:
                groups.append([j])
        for group in groups:
            point = points[group[0]]
            cv2.circle(image, tuple(point), 6, color, 2 if dashed else -1, cv2.LINE_AA)
            offset = np.array((9, -9 if dashed else 19))
            label = f"{prefix} J" + "/".join(map(str, group))
            cv2.putText(image, label, tuple(point + offset), cv2.FONT_HERSHEY_SIMPLEX,
                        .43, color, 1, cv2.LINE_AA)
        flange = frames[-1]
        for axis, axis_color in enumerate(AXIS_COLORS):
            tip = flange[:3, 3] + .085 * flange[:3, axis]
            endpoint = project(self.arm_view_rotation @ tip, self.yaw, self.pitch,
                               self.scale, center)[0]
            line(points[-1], endpoint, axis_color, 2)
            cv2.putText(image, prefix + "XYZ"[axis], tuple(endpoint + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, axis_color, 1, cv2.LINE_AA)

    def _draw_arm_legend(self, image, geometry_status):
        cv2 = self.cv2
        y = 204 if self.start_name is not None or self.definition_mode else 90
        cv2.line(image, (22, y - 4), (55, y - 4), CURRENT_ARM_COLOR, 4, cv2.LINE_AA)
        cv2.putText(image, "CURRENT / C: solid", (64, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .46, CURRENT_ARM_COLOR, 1, cv2.LINE_AA)
        if self.start_frames is not None:
            for x in (300, 315, 330):
                cv2.line(image, (x, y - 4), (x + 9, y - 4), START_ARM_COLOR, 3, cv2.LINE_AA)
            cv2.putText(image, "START / S: dashed", (350, y), cv2.FONT_HERSHEY_SIMPLEX,
                        .46, START_ARM_COLOR, 1, cv2.LINE_AA)
        cv2.putText(image, self.arm_view_label + "; flange XYZ shows wrist roll",
                    (22, y + 22), cv2.FONT_HERSHEY_SIMPLEX, .42, (200, 200, 200), 1, cv2.LINE_AA)
        if geometry_status.startswith("CURRENT hidden"):
            cv2.putText(image, geometry_status, (22, y + 44), cv2.FONT_HERSHEY_SIMPLEX,
                        .45, FAIL_COLOR, 1, cv2.LINE_AA)

    def _draw_status_bar(self, image: np.ndarray, statuses, now_ns: int) -> None:
        cv2 = self.cv2
        rec_width = 130
        box_width = (self.width - rec_width - 12 * (len(statuses) + 2)) // max(len(statuses), 1)
        x = 12
        for stream, status in statuses:
            color = OK_COLOR if status.ok else FAIL_COLOR
            cv2.rectangle(image, (x, 10), (x + box_width, 62), color, -1)
            if stream is not None:
                cv2.rectangle(image, (x, 10), (x + 8, 62), stream.color, -1)
            label = stream.label if stream is not None else "VICON"
            cv2.putText(image, label, (x + 16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, status.text, (x + 16, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            x += box_width + 12
        session = self.recorder.current() if self.recorder is not None else None
        rec_x = self.width - rec_width - 12
        if session is not None:
            elapsed = max(0.0, (now_ns - session.start_ns) / 1e9)
            cv2.circle(image, (rec_x + 16, 36), 9, (40, 40, 230), -1, cv2.LINE_AA)
            text, color = f"REC {elapsed:5.1f}s", (255, 255, 255)
        else:
            text, color = "NOT RECORDING", (150, 150, 150)
        cv2.putText(image, text, (rec_x + (32 if session is not None else 0), 41),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1 + (session is not None), cv2.LINE_AA)

    def update(self) -> None:
        self.cv2.imshow(self.window_name, self.render())

    def close(self) -> None:
        try:
            self.cv2.destroyWindow(self.window_name)
        except Exception:
            pass


def load_monitor_calibration(path: Path | None) -> dict | None:
    return None if path is None else json.loads(Path(path).read_text())
