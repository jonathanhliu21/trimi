"""Display measured Piper joints; never connect to or command the arm.

Limits come from the same SDK presets as local replay IK. Loading them does
not query the controller or transmit CAN frames. CSV recording stays in the
Piper worker at its sampling rate; rendering uses the latest sample only.
"""
from __future__ import annotations

import time
import textwrap

import numpy as np

from .config import PIPER_JOINT_FIELDS
from .utils import import_cv2
from .piper_worker import controller_status_problem

WINDOW_NAME = "trimorph_piper_joints"
WARNING_MARGIN_DEG = 3.0
SIGNAL_TIMEOUT_S = 0.5
GREEN, RED, GRAY = (60, 170, 60), (40, 40, 210), (100, 100, 100)
TEXT = (235, 235, 235)
WIDTH, HEIGHT = 940, 990
SCORE_TOP = 155
ROW_TOP, ROW_HEIGHT = 440, 85
BAR_LEFT, BAR_RIGHT = 230, 640


def piper_joint_limits_deg() -> np.ndarray:
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD

    preset = ROBOT_JOINT_LIMIT_PRESET_RAD["piper"]
    return np.degrees([preset[f"joint{i}"] for i in range(1, 7)])


def joint_limit_status(angle_deg: float, lower: float, upper: float) -> tuple[bool, str]:
    """Green only beyond the 3-degree margin, including red at its boundary."""
    if not np.isfinite(angle_deg):
        return False, "INVALID ANGLE"
    remaining = min(angle_deg - lower, upper - angle_deg)
    if remaining < 0:
        return False, f"OUTSIDE by {-remaining:.1f} deg"
    if remaining <= WARNING_MARGIN_DEG + 1e-9:
        return False, f"NEAR LIMIT {remaining:.1f} deg"
    return True, f"IN RANGE {remaining:.1f} deg"


class PiperJointMonitor:
    def __init__(self, shared, *, recorder=None, cv2=None, limits_deg=None):
        self.shared, self.recorder = shared, recorder
        self.cv2 = cv2 if cv2 is not None else import_cv2()
        self.limits = np.asarray(
            piper_joint_limits_deg() if limits_deg is None else limits_deg, dtype=float)
        if (self.limits.shape != (6, 2) or not np.isfinite(self.limits).all()
                or np.any(self.limits[:, 1] - self.limits[:, 0] <= 2 * WARNING_MARGIN_DEG)):
            raise ValueError("Piper monitor needs six finite joint ranges wider than 6 degrees")
        self.window_name = WINDOW_NAME
        self._last_draw_ns = None
        self._episode = None
        self._episode_ok = True
        self._rows_checked = 0
        self.samples_ok = self.samples_collected = 0
        self.cv2.namedWindow(self.window_name)

    def _check_recorded_rows(self):
        """Check every logged joint frame, including frames between display refreshes."""
        if self._episode is None:
            return
        with self._episode.lock:
            rows = self._episode.piper_rows[self._rows_checked:]
            self._rows_checked += len(rows)
        for row in rows:
            try:
                angles = np.degrees([float(row[name]) for name in PIPER_JOINT_FIELDS])
                ok = all(joint_limit_status(angle, *limits)[0]
                         for angle, limits in zip(angles, self.limits))
                if "arm_status" in row or "err_code" in row:
                    error = row.get("err_code")
                    ok &= not controller_status_problem(
                        str(row.get("arm_status") or ""),
                        int(error) if error not in (None, "") else None)
            except (KeyError, TypeError, ValueError):
                ok = False
            self._episode_ok &= ok

    def _track_collection(self, session, overall_ok):
        if session is not self._episode:
            if self._episode is not None:
                self._check_recorded_rows()
                if self._episode.stop_reason != "foot_pedal_delete":
                    self.samples_collected += 1
                    self.samples_ok += bool(self._episode_ok and self._rows_checked)
            self._episode = session
            self._episode_ok = True
            self._rows_checked = 0
        if session is not None:
            self._episode_ok &= overall_ok
            self._check_recorded_rows()

    def _draw_collection(self, image, session, now_ns):
        cv2 = self.cv2
        top = SCORE_TOP
        collecting = session is not None
        cv2.rectangle(image, (8, top), (WIDTH - 8, top + 54), RED if collecting else GRAY, -1)
        label = (f"COLLECTING SAMPLE {session.paths.episode_index:06d}"
                 if collecting else "IDLE - NOT COLLECTING")
        cv2.putText(image, label, (24, top + 38), cv2.FONT_HERSHEY_SIMPLEX,
                    1., TEXT, 3, cv2.LINE_AA)
        score = f"{self.samples_ok} / {self.samples_collected}"
        scale = 3.6
        while cv2.getTextSize(score, cv2.FONT_HERSHEY_SIMPLEX, scale, 7)[0][0] > WIDTH - 48:
            scale *= .9
        width = cv2.getTextSize(score, cv2.FONT_HERSHEY_SIMPLEX, scale, 7)[0][0]
        cv2.putText(image, score, ((WIDTH - width) // 2, top + 148),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT, 7, cv2.LINE_AA)
        cv2.putText(image, "SAMPLES WITHOUT IK ERRORS / TOTAL SAVED SAMPLES (this run)",
                    (24, top + 191), cv2.FONT_HERSHEY_SIMPLEX, .65, TEXT, 1, cv2.LINE_AA)
        if collecting:
            state = "NO WARNINGS SO FAR" if self._episode_ok else "WARNING SEEN IN THIS SAMPLE"
            elapsed = max(0., (now_ns - session.start_ns) / 1e9)
            cv2.putText(image, f"{elapsed:.1f} s - {state}", (24, top + 222),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, GREEN if self._episode_ok else RED, 2, cv2.LINE_AA)

    def render(self, now_ns=None) -> np.ndarray:
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        cv2 = self.cv2
        image = np.full((HEIGHT, WIDTH, 3), 24, dtype=np.uint8)

        def text(label, x, y, color=TEXT, scale=.55):
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, 1, cv2.LINE_AA)

        sample = self.shared.latest_piper_joints()
        angles = np.full(6, np.nan)
        signal = "NO JOINT FEEDBACK"
        fresh = False
        if sample is not None:
            age_s = max(0., (now_ns - sample.monotonic_ns) / 1e9)
            fresh = age_s <= SIGNAL_TIMEOUT_S
            signal = "LIVE FEEDBACK" if fresh else f"STALE FEEDBACK {age_s:.1f}s"
            measured = np.asarray(sample.joints, float)
            if measured.shape == (6,):
                angles = np.degrees(measured)
        controller = self.shared.latest_piper_controller_status()
        controller_problem = "NO CONTROLLER STATUS"
        if controller is not None:
            controller_age_s = max(0., (now_ns - controller.monotonic_ns) / 1e9)
            controller_problem = (f"CONTROLLER STATUS STALE {controller_age_s:.1f}s"
                                  if controller_age_s > SIGNAL_TIMEOUT_S else
                                  controller_status_problem(controller.arm_status, controller.err_code))
        controller_ok = not controller_problem
        overall_ok = fresh and controller_ok and all(
            joint_limit_status(angle, *limits)[0] for angle, limits in zip(angles, self.limits))
        cv2.rectangle(image, (0, 0), (WIDTH, 40), GREEN if overall_ok else RED, -1)
        text("Piper joint ranges / firmware: " + ("OK" if overall_ok else "WARNING"),
             20, 29, scale=.75)
        text(signal, 20, 59, GREEN if fresh else RED)
        session = self.recorder.current() if self.recorder is not None else None
        self._track_collection(session, overall_ok)
        self._draw_collection(image, session, now_ns)
        for i, line in enumerate(textwrap.wrap(
                "FIRMWARE: " + (controller_problem or controller.arm_status), width=105, max_lines=2)):
            text(line, 20, 86 + 24 * i, GREEN if controller_ok else RED)
        text("SDK preset limits | red at <= 3 deg from either limit, or outside", 20, 145)
        text("Joint / measured deg", 20, 426, GRAY)
        text("Min             Joint range (deg)             Max", 210, 426, GRAY)
        text("Nearest limit", 680, 426, GRAY)
        for j, (angle, (lower, upper)) in enumerate(zip(angles, self.limits)):
            y = ROW_TOP + j * ROW_HEIGHT
            ok, status = joint_limit_status(angle, lower, upper)
            if not fresh:
                status = "NO DATA" if sample is None else "STALE"
            elif not controller_ok and ok:
                status = "FIRMWARE WARNING"
            color = GREEN if fresh and ok and controller_ok else RED
            cv2.rectangle(image, (16, y), (WIDTH - 16, y + 73), color, 1)
            value = f"{angle:+.1f}" if np.isfinite(angle) else "--"
            text(f"J{j+1}    {value}", 28, y + 32, color, .7)
            text(status, 675, y + 32, color)
            # Fixed physical range with red strips marking both warning margins.
            left, right, top, bottom = BAR_LEFT, BAR_RIGHT, y + 18, y + 39
            cv2.rectangle(image, (left, top), (right, bottom), GRAY, -1)
            margin_px = round(WARNING_MARGIN_DEG / (upper - lower) * (right - left))
            cv2.rectangle(image, (left, top), (left + margin_px, bottom), RED, -1)
            cv2.rectangle(image, (right - margin_px, top), (right, bottom), RED, -1)
            if np.isfinite(angle):
                x = round(left + np.clip((angle - lower) / (upper - lower), 0, 1) * (right - left))
                cv2.line(image, (x, top - 6), (x, bottom + 6), color, 4)
            text(f"{lower:+.1f}", left - 18, y + 63, GRAY)
            text(f"{upper:+.1f}", right - 24, y + 63, GRAY)
        text("q / Esc: stop collection | Joint CSVs are saved only during an episode", 20, HEIGHT - 14, GRAY)
        return image

    def update(self) -> None:
        now_ns = time.monotonic_ns()
        if self._last_draw_ns is None or now_ns - self._last_draw_ns >= 50_000_000:
            self.cv2.imshow(self.window_name, self.render(now_ns))
            self._last_draw_ns = now_ns

    def close(self) -> None:
        try:
            self.cv2.destroyWindow(self.window_name)
        except Exception:
            pass
