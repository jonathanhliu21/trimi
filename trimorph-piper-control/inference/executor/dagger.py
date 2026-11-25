"""HG-DAgger mode for the executor: executor.py --dagger DATASET_ROOT --takeover teleop|kinesthetic.

The policy runs as usual until the operator steps on the takeover pedal. The
operator then finishes the episode; control never returns to the policy:

- teleop: the arm follows the Quest as quest_teleop.py does (deltas from the
  pose the policy last commanded, takeover pedal held = clutch engaged), sent
  through the executor's joint IK and speed caps; the follower hand follows
  the worn leader TriMI after a short ramp.
- kinesthetic: the arm runs midas-piper-control's calibrated gravity
  compensation (gravity_comp.DragController, MIT torque) and the hand goes
  limp, so the operator moves both by hand.

Each rollout is one raw episode in the collection layout
(data_collection/DATASET_SPEC.md): postprocess_dataset.py --pose-source piper_fk
converts it, and scripts/dagger/export_segments.py labels every frame policy,
human or handoff for training (DAGGER_PLAN.md at the repository root).

Pedals (--pedal-keys TAKEOVER SUCCESS FAILURE, default a b c; find yours with
python -m scripts.piper_api.foot_pedal):
  takeover  at the start pose: start a rollout. During the policy: take over.
            During a teleop takeover: held = the arm follows the Quest.
  success   end the rollout as a success.
  failure   end the rollout as a failure; held --discard-hold-s: discard it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import signal
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

ROLES = ("takeover", "success", "failure")
DEFAULT_PEDAL_KEYS = ("a", "b", "c")
TAKEOVER_MODES = ("teleop", "kinesthetic")
OUTCOMES = ("success", "failure")  # saved with status "complete"; "discard" deletes the episode
EULER_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
HAND_FIELDS = tuple(f"pos_{i}" for i in range(13))
TARGET_FIELDS = tuple(f"target_{i}" for i in range(13))
CAMERA_FIELDS = ("frame_idx", "monotonic_ns", "receipt_monotonic_ns", "sequence_num")
# raw piper.csv column <- executor Piper log column, in data collection's layout
# (piper_worker.gravity_comp_sample): one row per joint frame, its CAN receive
# time, and the firmware flange pose in x..yaw.
PIPER_COLUMNS = (("monotonic_ns", "monotonic_ns"), ("robot_timestamp_s", "joint_timestamp_s"),
                 ("joint_rx_monotonic_ns", "joint_rx_monotonic_ns"),
                 *((f"joint_{i}", f"joint_{i}") for i in range(1, 7)),
                 *((name, f"fw_{name}") for name in EULER_FIELDS), ("eef_timestamp_s", "fw_timestamp_s"),
                 *((f"vel_{i}", f"vel_{i}") for i in range(1, 7)),
                 *((f"current_{i}", f"current_{i}") for i in range(1, 7)),
                 ("arm_status", "arm_status"), ("motion_status", "motion_status"))
PIPER_FIELDS = tuple(name for name, _ in PIPER_COLUMNS)
COMMAND_FIELDS = ("monotonic_ns", "source", "cycle", "index", "engaged", *(f"raw_{name}" for name in EULER_FIELDS),
                  *EULER_FIELDS, *(f"cmd_joint_{i}" for i in range(1, 7)),
                  "sigma", "damping", "step_scale", "pos_err_mm", "rot_err_deg")
EVENT_FIELDS = ("monotonic_ns", "source", "cycle", "event", "detail")
QUEST_FIELDS = ("monotonic_ns", "seq", "px", "py", "pz", "qx", "qy", "qz", "qw")
GRAVITY_FIELDS = ("monotonic_ns", "state", *(f"q_{i}" for i in range(1, 7)), *(f"v_{i}" for i in range(1, 7)),
                  *(f"tau_cmd_{i}" for i in range(1, 7)), *(f"tau_fb_{i}" for i in range(1, 7)),
                  "joint_timestamp_s", "motor_timestamp_s", *(f"motor_timestamp_{i}_s" for i in range(1, 7)),
                  "stale_sources", "feedback_error", *(f"kp_{i}" for i in range(1, 7)),
                  *(f"kd_{i}" for i in range(1, 7)))
# A leader sample older than this is not followed; the follower keeps its last target.
LEADER_TIMEOUT_S = 0.25
# gravity_comp: consecutive stale feedback reads before a freeze, and the
# position-mode latch time (drag_mode.STALE_FREEZE_CYCLES, hw_session.MODE_SWITCH_SETTLE_S).
STALE_FREEZE_CYCLES = 10
MODE_SWITCH_SETTLE_S = 0.3
VIEW_REFRESH_S = 0.05
# Where run_gravity_comp.sh looks for midas-piper-control.
DEFAULT_GRAVITY_COMP_REPOS = (Path.home() / "midas_hand" / "midas-piper-control",
                              Path.home() / "workspace" / "midas-piper-control")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def as_pose(value, name):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be a finite pose of shape (6,)")
    return pose


def refresh_views(robot, camera, view):
    camera.preview()
    if view is not None:
        view.update(robot.eef_pose())


def take_rows(rows, start_ns, end_ns, key="monotonic_ns"):
    """Rows of a time-ordered list shared with a logger thread that fall in [start_ns, end_ns].

    Everything stamped up to end_ns is removed from the shared list, so it only
    ever holds the current episode. Rows appended meanwhile are kept: the
    logger only appends, and each list operation here is atomic.
    """
    count = len(rows)
    snapshot = rows[:count]
    cut = 0
    while cut < count and int(snapshot[cut][key]) <= end_ns:
        cut += 1
    del rows[:cut]
    return [row for row in snapshot[:cut] if int(row[key]) >= start_ns]


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Pedals


def pedal_codes(keys):
    """Key codes for the takeover, success and failure pedals."""
    from scripts.piper_api.foot_pedal import key_code

    codes = tuple(key_code(key) for key in keys)
    if len(codes) != len(ROLES) or len(set(codes)) != len(codes):
        raise ValueError(f"--pedal-keys needs {len(ROLES)} different keys (takeover, success, failure)")
    return codes


class DaggerPedals:
    """Three pedals on one input device, by role; presses are debounced per role."""

    def __init__(self, trigger, codes, debounce_s=0.3):
        self.trigger = trigger
        self.codes = dict(zip(ROLES, codes))
        self.roles = {code: role for role, code in self.codes.items()}
        self.debounce_s = debounce_s
        self.last_press = {}

    def poll(self):
        """Roles pressed since the last poll, in press order."""
        pressed = []
        for event in self.trigger.poll():
            role = self.roles.get(event.code)
            if role is None:
                continue
            now = time.monotonic()
            if now - self.last_press.get(role, -math.inf) < self.debounce_s:
                continue
            self.last_press[role] = now
            if role not in pressed:
                pressed.append(role)
        return pressed

    def held_s(self, role):
        """Seconds the role's pedal has been held, or None; updated by poll()."""
        return self.trigger.held_s(self.codes[role])

    def describe(self):
        from scripts.piper_api.foot_pedal import KEY_NAMES

        return ", ".join(f"{role} = {KEY_NAMES.get(code, code)} ({code})" for role, code in self.codes.items())

    def stop(self):
        self.trigger.stop()


class EndPedals:
    """Success ends at once; failure ends on release, or discards the episode when held long enough."""

    def __init__(self, pedals, discard_hold_s):
        self.pedals = pedals
        self.discard_hold_s = discard_hold_s
        self.failure_at = None

    @property
    def pending(self):
        """The failure pedal is down and it is not yet known whether it is held to discard."""
        return self.failure_at is not None

    def update(self, pressed):
        """'success', 'failure', 'discard', or None while undecided."""
        if self.failure_at is None and "success" in pressed:
            return "success"
        if self.failure_at is None and "failure" in pressed:
            self.failure_at = time.monotonic()
        if self.failure_at is not None:
            held = self.pedals.held_s("failure")
            if held is None:
                return "failure"
            if held >= self.discard_hold_s:
                return "discard"
        return None


# ---------------------------------------------------------------------------
# Recording


class EpisodeVideo:
    """Own the OAK queue for the whole run; write each episode's frames with their capture times.

    Takes the place of OakCamera.start_recording: the camera's newest_bgr()
    serves this thread's latest frame to the policy, and OakCamera.close()
    stops the thread. Capture times are DepthAI mid-exposure times on the host
    monotonic clock, as in collection (depthai_task_camera_worker.py).
    """

    def __init__(self, camera, codec, capture_ns=None):
        if capture_ns is None:
            from data_collection.collection.depthai_utils import frame_capture_monotonic_ns, host_clock_offset_ns

            offset_ns = host_clock_offset_ns(camera.dai)
            capture_ns = lambda packet: frame_capture_monotonic_ns(camera.dai, packet, offset_ns)  # noqa: E731
        self.camera = camera
        self.codec = codec
        self.capture_ns = capture_ns
        self.lock = threading.Lock()
        self.writer = None
        self.rows = None
        camera.newest_bgr()  # The newest frame from setup; the worker owns the queue from now on.
        camera.record_thread = threading.Thread(target=self.run, name="dagger-video", daemon=True)
        camera.record_thread.start()

    def run(self):
        camera = self.camera
        try:
            while not camera.record_stop.is_set():
                packet = camera.queue.tryGet()
                if packet is None:
                    camera.record_stop.wait(0.002)
                    continue
                receipt_ns = time.monotonic_ns()
                bgr = packet.getCvFrame()
                if bgr is None:
                    continue
                sample_ns = self.capture_ns(packet)
                bgr = camera.flip_frame(camera.cv2, bgr, camera.flip)
                camera.latest_frame = (bgr, time.time(), time.monotonic())
                with self.lock:
                    if self.writer is not None:
                        self.writer.write(bgr)
                        self.rows.append({"frame_idx": len(self.rows), "monotonic_ns": sample_ns,
                                          "receipt_monotonic_ns": receipt_ns,
                                          "sequence_num": packet.getSequenceNum()})
        except Exception as exc:
            camera.record_error = exc
        finally:
            with self.lock:
                writer, self.writer = self.writer, None
            if writer is not None:
                writer.release()

    def begin(self, path):
        camera = self.camera
        camera.check_recording()
        if camera.latest_frame is None:
            raise RuntimeError("No camera frame to size the episode video")
        height, width = camera.latest_frame[0].shape[:2]
        writer = camera.cv2.VideoWriter(str(path), camera.cv2.VideoWriter_fourcc(*self.codec),
                                        float(camera.fps), (width, height))
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Could not open the episode video writer: {path}")
        with self.lock:
            self.writer, self.rows = writer, []

    def end(self):
        """Close the episode video; its frame rows (frame i of camera.mp4 is row i)."""
        with self.lock:
            writer, rows = self.writer, self.rows
            self.writer = self.rows = None
        if writer is not None:
            writer.release()
        return rows or []


class HandSampler:
    """Read a hand's 13 calibrated positions at a fixed rate; keep the current episode's rows."""

    def __init__(self, read, hz, name, *, lock=None, target=None):
        self.read = read
        self.period = 1.0 / hz
        self.name = name
        self.lock = lock
        self.target = target
        self.latest = None  # (monotonic_ns, positions)
        self.rows = None
        self.errors = 0
        self.first_error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name=f"{name}-hand-sampler", daemon=True)
        self.thread.start()

    def sample(self):
        with self.lock if self.lock is not None else nullcontext():
            positions = np.asarray(self.read(), dtype=float)
        if positions.shape != (13,) or not np.isfinite(positions).all():
            raise ValueError(f"{self.name} hand read is not 13 finite positions")
        now = time.monotonic_ns()
        target = self.target() if self.target is not None else None
        self.latest = (now, positions)
        rows = self.rows
        if rows is not None:
            rows.append((now, positions, None if target is None else np.asarray(target, dtype=float)))

    def run(self):
        deadline = time.perf_counter()
        while not self.stop_event.is_set():
            try:
                self.sample()
            except Exception as exc:  # One failed read skips one sample.
                self.errors += 1
                self.first_error = self.first_error or repr(exc)
            deadline += self.period
            now = time.perf_counter()
            if deadline < now:
                deadline = now + self.period
            self.stop_event.wait(max(0.0, deadline - now))

    def begin(self):
        self.rows = []

    def end(self):
        rows, self.rows = self.rows, None
        return rows or []

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        if self.errors:
            print(f"WARNING: {self.errors} {self.name} hand reads failed; first: {self.first_error}", flush=True)


@dataclass
class Episode:
    index: int
    path: Path
    start_ns: int
    start_s: float
    started_utc: str
    start_record: dict | None
    events: list = field(default_factory=list)
    intervals: list = field(default_factory=list)  # closed [start_ns, end_ns] human-controlled intervals
    open_interval: int | None = None
    takeover_ns: int | None = None
    teleop_commands: list = field(default_factory=list)
    tables: dict = field(default_factory=dict)  # file name -> (fields, rows)


# ---------------------------------------------------------------------------
# Session


class DaggerSession:
    """Rollout control for --dagger, handed to the policy loops in place of the pedal.

    The loops only call poll() (anything truthy interrupts the policy);
    executor.pause_at_start and wait_for_policy_start pass control to
    interrupt() and wait_for_start(), recognizing this object by the DAGGER
    class attribute.
    """

    DAGGER = True

    def __init__(self, args, robot, camera, pedals, takeover, *, video, follower, label_hand, log,
                 quest=None, leader_hand=None):
        self.args = args
        self.robot = robot
        self.camera = camera
        self.pedals = pedals
        self.takeover = takeover
        self.video = video
        self.follower = follower
        # The hand stream saved as the training label (leader_hand.csv): the
        # worn leader for teleop, the robot's own hand for kinesthetic.
        self.label_hand = follower if label_hand is None else label_hand
        self.log = log
        self.quest = quest
        self.leader_hand = leader_hand
        self.view = None
        self.raw = Path(args.dagger).expanduser() / "raw"
        self.pending = []
        self.episode = None
        self.closed = False
        self.counts = {"success": 0, "failure": 0, "discard": 0, "takeover": 0}

    # -- interface for the executor loops ----------------------------------

    def poll(self):
        pressed = self.pedals.poll()
        episode = self.episode
        if (not pressed and episode is not None and self.args.episode_max_s
                and time.monotonic() - episode.start_s > self.args.episode_max_s):
            pressed = ["timeout"]
        if pressed:
            self.pending = pressed
        return pressed

    def wait_for_start(self, robot, camera, view=None, resume=False):
        """Hold at the start until the takeover pedal, then start recording a rollout."""
        self.view = view
        self.pedals.poll()  # Discard presses made while moving to the start.
        done = sum(self.counts[key] for key in OUTCOMES)
        print(f"At start ({done} saved: {self.counts['success']} success, {self.counts['failure']} failure, "
              f"{self.counts['takeover']} with a takeover). Takeover pedal = start the next rollout; "
              "Ctrl-C = quit.", flush=True)
        while "takeover" not in self.pedals.poll():
            refresh_views(robot, camera, view)
            time.sleep(0.02)
        self.begin_episode()

    def interrupt(self, args, robot, camera, view=None):
        """pause_at_start in DAgger mode: take over or end the rollout, save it, and start the next."""
        roles, self.pending = self.pending, []
        self.view = view
        if view is not None:
            view.set_chunk(None)
        if self.episode is None:
            self.begin_episode()
        if "takeover" in roles:
            outcome, reason = self.take_over(), "pedal"
        else:
            outcome, reason = self.end_without_takeover(roles)
        self.finish_episode(outcome, reason)
        robot.return_to_start(args)
        self.wait_for_start(robot, camera, view)

    def stop(self):
        self.close()

    # -- rollout -------------------------------------------------------------

    def event(self, name, detail=""):
        if self.episode is not None:
            self.episode.events.append({"monotonic_ns": time.monotonic_ns(), "source": "dagger", "cycle": "",
                                        "event": name, "detail": detail})

    def human_interval_start(self):
        episode = self.episode
        if episode is not None and episode.open_interval is None:
            episode.open_interval = time.monotonic_ns()

    def human_interval_end(self):
        episode = self.episode
        if episode is not None and episode.open_interval is not None:
            episode.intervals.append([episode.open_interval, time.monotonic_ns()])
            episode.open_interval = None

    def add_table(self, name, fields, rows):
        if self.episode is not None:
            self.episode.tables[name] = (fields, rows)

    def teleop_command(self, row):
        if self.episode is not None:
            self.episode.teleop_commands.append(row)

    def take_over(self):
        episode = self.episode
        episode.takeover_ns = time.monotonic_ns()
        self.counts["takeover"] += 1
        self.event("takeover", self.takeover.kind)
        print(f"Pedal: TAKEOVER ({self.takeover.kind}) in episode {episode.index}; the policy is stopped. "
              "End with the success or failure pedal.", flush=True)
        try:
            return self.takeover.run(self)
        finally:
            self.human_interval_end()

    def end_without_takeover(self, roles):
        if "timeout" in roles:
            print(f"Episode exceeded --episode-max-s {self.args.episode_max_s:g}; ending it as a failure.", flush=True)
            return "failure", "timeout"
        end = EndPedals(self.pedals, self.args.discard_hold_s)
        outcome = end.update(roles)
        while outcome is None:
            time.sleep(0.02)
            refresh_views(self.robot, self.camera, self.view)
            outcome = end.update(self.pedals.poll())
        return outcome, "pedal"

    def last_start_record(self):
        """The executor's most recent start (output/executor_starts.jsonl), or None without start moves."""
        path = Path(self.args.start_log)
        if getattr(self.args, "no_start_move", False) or not path.is_file():
            return None
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        return json.loads(lines[-1]) if lines else None

    def begin_episode(self):
        from data_collection.collection.episode import next_episode_index

        index = next_episode_index(self.raw)
        path = self.raw / f"episode_{index:06d}"
        path.mkdir(parents=True)
        start_ns = time.monotonic_ns()
        piper_log = getattr(self.robot, "piper_log", None)
        if piper_log is not None:
            take_rows(piper_log.rows, start_ns, start_ns - 1)  # Drop rows from between episodes.
        if self.log is not None:
            take_rows(self.log.commands, start_ns, start_ns - 1)
            take_rows(self.log.events, start_ns, start_ns - 1)
            self.log.chunks[:] = [chunk for chunk in self.log.chunks if chunk[1] >= start_ns]
        if self.quest is not None:
            with self.quest.lock:
                self.quest.rows.clear()
        self.video.begin(path / "camera.mp4")
        self.follower.begin()
        if self.label_hand is not self.follower:
            self.label_hand.begin()
        self.episode = Episode(index=index, path=path, start_ns=start_ns, start_s=time.monotonic(),
                               started_utc=utc_now(), start_record=self.last_start_record())
        self.event("start")
        print(f"Episode {index}: policy running. Pedals: {self.pedals.describe()}.", flush=True)

    def finish_episode(self, outcome, reason=""):
        episode = self.episode
        if episode is None:
            return
        self.human_interval_end()
        self.event("end", f"{outcome} ({reason})" if reason else outcome)
        self.episode = None
        end_ns = time.monotonic_ns()
        frames = self.video.end()
        follower_rows = self.follower.end()
        label_rows = follower_rows if self.label_hand is self.follower else self.label_hand.end()
        piper_log = getattr(self.robot, "piper_log", None)
        piper_rows = take_rows(piper_log.rows, episode.start_ns, end_ns) if piper_log is not None else []
        policy_commands = events = chunks = []
        if self.log is not None:
            policy_commands = take_rows(self.log.commands, episode.start_ns, end_ns)
            events = take_rows(self.log.events, episode.start_ns, end_ns)
            chunks = [chunk for chunk in self.log.chunks if episode.start_ns <= chunk[1] <= end_ns]
            self.log.chunks[:] = [chunk for chunk in self.log.chunks if chunk[1] > end_ns]
        quest_rows = []
        if self.quest is not None:
            with self.quest.lock:
                quest_rows = [row for row in self.quest.rows if episode.start_ns <= row["monotonic_ns"] <= end_ns]
                self.quest.rows.clear()
        if outcome == "discard":
            shutil.rmtree(episode.path, ignore_errors=True)
            self.counts["discard"] += 1
            print(f"Episode {episode.index} discarded (failure pedal held).", flush=True)
            return
        streams = {"camera_frames": len(frames), "label_hand": len(label_rows), "piper": len(piper_rows)}
        status = "complete" if outcome in OUTCOMES and all(streams.values()) else "interrupted"
        if outcome in OUTCOMES and status != "complete":
            print(f"WARNING: episode {episode.index} is missing a stream {streams}; saved as interrupted.",
                  flush=True)
        self.write_episode(episode, end_ns, outcome, reason, status, frames, label_rows, follower_rows,
                           piper_rows, policy_commands, events, chunks, quest_rows)
        if status == "complete":
            self.counts[outcome] += 1
        human_s = sum(end - start for start, end in episode.intervals) / 1e9
        takeover = (f"takeover after {(episode.takeover_ns - episode.start_ns) / 1e9:.1f} s, "
                    f"{human_s:.1f} s human-controlled" if episode.takeover_ns else "no takeover")
        print(f"Saved episode {episode.index}: {outcome}, {status}, {(end_ns - episode.start_ns) / 1e9:.1f} s, "
              f"{takeover}, {len(frames)} frames -> {episode.path}", flush=True)

    def write_episode(self, episode, end_ns, outcome, reason, status, frames, label_rows, follower_rows,
                      piper_rows, policy_commands, events, chunks, quest_rows):
        path = episode.path
        args = self.args
        write_csv(path / "camera_timestamps.csv", CAMERA_FIELDS, frames)
        write_csv(path / "leader_hand.csv", ("monotonic_ns", *HAND_FIELDS),
                  [{"monotonic_ns": now, **dict(zip(HAND_FIELDS, positions))} for now, positions, _ in label_rows])
        write_csv(path / "follower.csv", ("monotonic_ns", *HAND_FIELDS, *TARGET_FIELDS),
                  [{"monotonic_ns": now, **dict(zip(HAND_FIELDS, positions)),
                    **({} if target is None else dict(zip(TARGET_FIELDS, target)))}
                   for now, positions, target in follower_rows])
        write_csv(path / "piper.csv", PIPER_FIELDS, [self.piper_row(row) for row in piper_rows])
        commands = [{"source": "policy", "engaged": "", **row} for row in policy_commands]
        commands += episode.teleop_commands
        write_csv(path / "commands.csv", COMMAND_FIELDS, sorted(commands, key=lambda row: int(row["monotonic_ns"])))
        executor_events = [{"source": "executor", **row} for row in events]
        write_csv(path / "events.csv", EVENT_FIELDS,
                  sorted(executor_events + episode.events, key=lambda row: int(row["monotonic_ns"])))
        if quest_rows:
            write_csv(path / "quest.csv", QUEST_FIELDS, quest_rows)
        for name, (fields, rows) in episode.tables.items():
            write_csv(path / name, fields, rows)
        if chunks:
            np.savez(path / "chunks.npz",
                     cycle=np.array([c for c, *_ in chunks]),
                     received_monotonic_ns=np.array([t for _, t, *_ in chunks]),
                     state=np.stack([s for *_, s, _ in chunks]),
                     actions=np.concatenate([a for *_, a in chunks]),
                     chunk_cycle=np.concatenate([np.full(len(a), c) for c, *_, a in chunks]))
        segments = [{"source": "policy", "start_ns": episode.start_ns, "end_ns": episode.takeover_ns or end_ns}]
        if episode.takeover_ns:
            segments.append({"source": "human", "start_ns": episode.takeover_ns, "end_ns": end_ns})
        metadata = {
            "status": status,
            "stop_reason": f"{outcome} ({reason})" if reason else outcome,
            "episode_index": episode.index,
            "started_utc": episode.started_utc,
            "stopped_utc": utc_now(),
            # Collection's episode clock (episode.py); postprocess --pose-source piper_fk reads it.
            "started_monotonic_ns": episode.start_ns,
            "stopped_monotonic_ns": end_ns,
            "duration_s": (end_ns - episode.start_ns) / 1e9,
            "counts": {"camera_frames": len(frames), "leader_hand": len(label_rows),
                       "follower": len(follower_rows), "piper": len(piper_rows),
                       "policy_commands": len(policy_commands), "teleop_commands": len(episode.teleop_commands),
                       "quest": len(quest_rows)},
            "hand": {"label_source": "leader" if self.label_hand is not self.follower else "follower",
                     "follower": hand_metadata(args.hand_config, args.hand_port),
                     "leader": (hand_metadata(args.leader_config, args.leader_port)
                                if args.takeover == "teleop" else None)},
            "piper": {"mode": f"executor_dagger_{args.takeover}", "pose": "firmware_flange_pose",
                      "pose_clock": "joint_rx_monotonic_ns", "training_pose_source": "piper_fk",
                      "firmware": args.piper_firmware},
            "camera": {"width": args.cam_width, "height": args.cam_height, "fps": args.cam_fps,
                       "flip": args.cam_flip, "exposure_us": args.cam_exposure_us, "iso": args.cam_iso,
                       "white_balance_k": args.cam_white_balance_k},
            "dagger": {
                "round": args.dagger_round,
                "takeover": args.takeover,
                "outcome": outcome,
                "reason": reason,
                "policy": getattr(self.log, "policy", None),
                "executor_mode": "rtc" if args.rtc else "async" if args.async_inference else "sync",
                "hz": args.hz,
                "start": episode.start_record,
                "takeover_ns": episode.takeover_ns,
                "segments": segments,
                "human_intervals": episode.intervals,
                "takeover_settings": self.takeover.describe(),
            },
            "executor_args": {key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()},
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    @staticmethod
    def piper_row(row):
        """Executor Piper log row -> raw piper.csv row in data collection's layout.

        x..yaw is the firmware flange pose. The training pose is FK of the joints
        on the CAN receive clock: postprocess_dataset.py --pose-source piper_fk.
        """
        return {name: row[source] for name, source in PIPER_COLUMNS if source in row}

    def close(self):
        """Save a rollout still in progress as interrupted, then stop the session's devices."""
        if self.closed:
            return
        self.closed = True
        try:
            if self.episode is not None:
                self.finish_episode("interrupted", "executor stopped")
        finally:
            for part in (self.follower, None if self.label_hand is self.follower else self.label_hand):
                if part is not None:
                    part.stop()
            if self.quest is not None:
                self.quest.stop()
            if self.leader_hand is not None:
                try:
                    self.leader_hand.shutdown()
                except Exception as exc:
                    print(f"WARNING: leader hand shutdown failed: {exc!r}", flush=True)
            self.pedals.stop()


def hand_metadata(path, port):
    path = Path(path).expanduser()
    return {"config_path": str(path.resolve()), "config_sha256": sha256(path) if path.is_file() else None,
            "port": str(port)}


# ---------------------------------------------------------------------------
# Teleop takeover


def command_pose(robot, pose, dt):
    """One IK step toward a Piper XYZ/Euler pose from the last joint target, sent as one move_j."""
    control = robot.joint_control
    if control.q is None:
        joints = robot.piper.get_joint_angles()
        if joints is None:
            raise RuntimeError("Piper joint feedback unavailable")
        control.reset(joints)
    target, info = control.command(as_pose(pose, "teleop pose"), dt)
    robot.piper.move_joint(target.tolist(), mode="smooth")
    robot.segment = None
    robot.sent_joint_target = target.copy()
    robot.commanded_pose = np.asarray(control.fk(target), dtype=float)
    info["joints"] = target
    return info


def open_leader_hand(args):
    """The worn leader TriMI, torque off, after checking it pairs with the follower calibration."""
    import yaml

    from data_collection.collection.hand_common import load_hand_config, verify_connected_hand
    from inference.executor.executor import MOTOR_IDS, load_hand_class

    leader_yaml = yaml.safe_load(Path(args.leader_config).expanduser().read_text())
    follower_yaml = yaml.safe_load(Path(args.hand_config).expanduser().read_text())
    try:
        from scripts.teleop.trimi_teleop import check_pair

        check_pair(leader_yaml, follower_yaml)
    except KeyError as missing:
        print(f"WARNING: cannot check the leader/follower calibration pair (no {missing} in a YAML).", flush=True)
    hand_class = load_hand_class(args.hand_api_repo)
    config = load_hand_config(Path(args.leader_config).expanduser(), args.leader_port, MOTOR_IDS)
    hand = hand_class(config=config, autoconnect=True)
    try:
        verify_connected_hand(hand, role="leader")
        if tuple(hand.motor_ids) != MOTOR_IDS:
            raise RuntimeError("Leader hand motor order must be exactly 0..12")
        hand.configure(enable_torque=False)
    except BaseException:
        hand.shutdown()
        raise
    return hand


class TeleopTakeover:
    """Arm from the Quest, anchored at the takeover pose (pedal held = clutch); hand from the leader."""

    kind = "teleop"

    def __init__(self, args, robot, leader, quest, calibration, calibration_path=None):
        self.args = args
        self.robot = robot
        self.leader = leader
        self.quest = quest
        self.calibration = calibration
        self.calibration_path = calibration_path

    def describe(self):
        args = self.args
        path = None if self.calibration_path is None else Path(self.calibration_path).expanduser()
        return {"calibration": None if path is None else str(path.resolve()),
                "calibration_sha256": None if path is None or not path.is_file() else sha256(path),
                "with_orientation": not args.teleop_position_only, "scale": args.teleop_scale,
                "hz": args.teleop_hz, "hand_ramp_s": args.takeover_hand_ramp_s,
                "max_offset_mm": args.teleop_max_offset_mm, "max_rotation_deg": args.teleop_max_rotation_deg,
                "max_speed_mm_s": args.teleop_max_speed_mm_s,
                "max_angular_speed_deg_s": args.teleop_max_angular_speed_deg_s}

    def run(self, session):
        from scripts.teleop.quest_teleop import TeleopMapper, pose_list, rate_limit

        args, robot = self.args, self.robot
        dt = 1.0 / args.teleop_hz
        control = robot.joint_control
        if control.q is None:
            joints = robot.piper.get_joint_angles()
            if joints is None:
                raise RuntimeError("Piper joint feedback unavailable")
            control.reset(joints)
        # The pose the arm is finishing, not the measured one: anchoring at the
        # measured pose would command a step back along the policy's motion.
        anchor = robot.commanded_pose if robot.commanded_pose is not None else control.fk(control.q)
        anchor = as_pose(anchor, "takeover anchor")
        command = (anchor[:3].copy(), Rotation.from_euler("xyz", anchor[3:]))
        mapper = TeleopMapper(self.calibration, anchor, scale=args.teleop_scale,
                              with_orientation=not args.teleop_position_only,
                              max_offset_m=args.teleop_max_offset_mm / 1000,
                              max_rotation_rad=math.radians(args.teleop_max_rotation_deg))
        max_step_m = args.teleop_max_speed_mm_s / 1000 * dt
        max_step_rad = math.radians(args.teleop_max_angular_speed_deg_s) * dt
        writer = robot.hand_writer
        hand_from = writer.target
        if hand_from is None:
            hand_from = self.leader.latest[1] if self.leader.latest is not None else None
        end = EndPedals(session.pedals, args.discard_hold_s)
        engaged = latched = False
        over_since = last_quest = None
        warned = set()
        t0 = next_tick = time.monotonic()
        next_view = t0
        session.event("teleop", f"anchor {np.round(anchor, 4).tolist()}")
        try:
            while True:
                now = time.monotonic()
                outcome = end.update(session.pedals.poll())
                if outcome is not None:
                    return outcome
                want = session.pedals.held_s("takeover") is not None and not end.pending
                sample = self.quest.get()
                fresh = sample is not None and now - sample[0] <= args.quest_timeout_s
                jump = (last_quest is not None and sample is not None and sample[3] != last_quest[3]
                        and 1000 * np.linalg.norm(sample[1] - last_quest[1]) > args.quest_jump_mm)
                last_quest = sample
                # Clutch: engage on a held pedal with fresh Quest data; a fault
                # latches until the pedal is released (quest_teleop.py).
                if not want:
                    latched = False
                if want and not engaged and not latched:
                    if fresh:
                        mapper.engage(sample[1], sample[2], *command)
                        engaged = True
                        session.human_interval_start()
                        session.event("engage")
                    elif "stale" not in warned:
                        print("Cannot engage: Quest data is stale", flush=True)
                        warned.add("stale")
                elif engaged and (not want or not fresh or jump):
                    mapper.disengage()
                    engaged = False
                    latched = want
                    session.human_interval_end()
                    session.event("disengage", "released" if not want else "Quest stale" if not fresh else "Quest jump")
                previous = command[0]
                if engaged:
                    command = rate_limit(command, mapper.target(sample[1], sample[2]), max_step_m, max_step_rad)
                pose = np.asarray(pose_list(*command), dtype=float)
                info = command_pose(robot, pose, dt)
                measured = robot.eef_pose()
                if engaged and measured is not None:
                    # The arm trails a moving target; a lasting excess means it is blocked.
                    speed_mm_s = 1000 * np.linalg.norm(command[0] - previous) / dt
                    error = 1000 * np.linalg.norm(np.asarray(measured[:3]) - robot.commanded_pose[:3])
                    allowed = args.max_tracking_error_mm + speed_mm_s * args.tracking_lag_s
                    over_since = (over_since or now) if error > allowed else None
                    if over_since is not None and now - over_since >= args.tracking_error_hold_s:
                        mapper.disengage()
                        engaged, latched, over_since = False, True, None
                        command = (np.asarray(measured[:3], dtype=float), Rotation.from_euler("xyz", measured[3:]))
                        session.human_interval_end()
                        session.event("disengage", f"tracking error {error:.0f} mm > {allowed:.0f} mm")
                        print(f"Disengaged: tracking error {error:.0f} mm (blocked?). Release the takeover pedal "
                              "and press it again to continue.", flush=True)
                leader = self.leader.latest
                if leader is not None and (time.monotonic_ns() - leader[0]) / 1e9 <= LEADER_TIMEOUT_S:
                    target = leader[1]
                    if hand_from is not None and args.takeover_hand_ramp_s > 0:
                        alpha = min(1.0, (now - t0) / args.takeover_hand_ramp_s)
                        target = hand_from + alpha * (leader[1] - hand_from)
                    writer.submit(target)
                elif "leader" not in warned:
                    print("WARNING: no fresh leader hand reading; the follower keeps its last target.", flush=True)
                    warned.add("leader")
                session.teleop_command({
                    "monotonic_ns": time.monotonic_ns(), "source": "teleop", "engaged": int(engaged),
                    **{f"raw_{name}": f"{v:.9f}" for name, v in zip(EULER_FIELDS, pose)},
                    **{name: f"{v:.9f}" for name, v in zip(EULER_FIELDS, robot.commanded_pose)},
                    **{f"cmd_joint_{i}": f"{v:.9f}" for i, v in enumerate(info["joints"], start=1)},
                    **{name: f"{info[name]:.6g}" for name in ("sigma", "damping", "step_scale", "pos_err_mm",
                                                               "rot_err_deg")},
                })
                if now >= next_view:
                    refresh_views(robot, session.camera, session.view)
                    next_view = now + VIEW_REFRESH_S
                next_tick += dt
                remaining = next_tick - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    next_tick = time.monotonic()
        finally:
            session.event("teleop_end")


# ---------------------------------------------------------------------------
# Kinesthetic takeover


def import_gravity_comp(repo=None):
    """midas-piper-control's gravity_comp package (calib, control, drag_mode, hw_session, model)."""
    candidates = gravity_comp_candidates(repo)
    root = next((path for path in candidates if (path / "gravity_comp" / "__init__.py").is_file()), None)
    if root is None:
        raise FileNotFoundError("midas-piper-control with gravity_comp/ not found in "
                                f"{[str(path) for path in candidates]}; pass --gravity-comp-repo")
    # Appended, not inserted: this repository's data_collection and scripts must win.
    if str(root) not in sys.path:
        sys.path.append(str(root))
    import gravity_comp

    if not Path(gravity_comp.__file__).resolve().is_relative_to(root.resolve()):
        raise RuntimeError(f"Wrong gravity_comp loaded: {gravity_comp.__file__}")
    from gravity_comp import calib, control, drag_mode, hw_session, model

    return SimpleNamespace(root=root, calib=calib, control=control, drag_mode=drag_mode,
                           hw_session=hw_session, model=model)


def gravity_comp_candidates(repo=None):
    """Explicit path > environment > sibling checkout > known local checkouts."""
    if repo is not None:
        return [Path(repo).expanduser()]
    configured = os.environ.get("TRIMORPH_GRAVITY_COMP_REPO")
    if configured:
        return [Path(configured).expanduser()]
    sibling = Path(__file__).resolve().parents[3] / "midas-piper-control"
    return [sibling, *DEFAULT_GRAVITY_COMP_REPOS]


@dataclass
class KinestheticPlan:
    """Everything the kinesthetic takeover needs from the calibration, built before any hardware moves."""

    gc: SimpleNamespace
    artifact: object
    model: object
    config: object
    info: dict
    driver: str
    calibration: Path
    calibration_sha256: str


def prepare_kinesthetic(args, gc=None):
    gc = gc or import_gravity_comp(args.gravity_comp_repo)
    path = Path(args.gravity_calib).expanduser()
    artifact = gc.calib.CalibrationArtifact.load(path)
    artifact.require("firmware", "noise", "feedback", "gravity", "delivery", "payload", "workspace")
    # As drag_mode --force: a friction section saved with failed asymmetry/spread
    # gates already stores the weaker-direction Coulomb value used for assist sizing.
    artifact.require("friction", allow_failed_gates=args.gravity_force)
    driver = artifact.firmware_driver()
    if driver != args.piper_firmware:
        raise ValueError(f"The gravity calibration is for Piper driver {driver!r}; run with --piper-firmware {driver}")
    mass, com = artifact.payload_mass_com()
    model = gc.model.ArmGravityModel(gravity=artifact.gravity_vector(), payload_mass=mass, payload_com=com)
    try:
        config, info, _ = gc.drag_mode.build_drag_config(artifact, model, args.kinesthetic_assist_frac,
                                                         allow_underpowered=args.allow_underpowered)
    except ValueError as exc:
        raise ValueError(f"{exc}\nThe arm needs more torque than the firmware cap somewhere in the workspace; "
                         "add --allow-underpowered and keep supporting the arm where it sags.") from exc
    config.ramp_in_s = args.kinesthetic_ramp_in_s
    config.validate()
    if info.get("underpowered_joints"):
        joints = ", ".join(f"J{j}" for j in info["underpowered_joints"])
        print(f"WARNING: gravity compensation is capped on {joints}: the arm sags where those joints need "
              "more torque; keep supporting it during kinesthetic takeovers.", flush=True)
    return KinestheticPlan(gc=gc, artifact=artifact, model=model, config=config, info=info, driver=driver,
                           calibration=path.resolve(), calibration_sha256=sha256(path))


def start_sag_warning(plan, joints_deg):
    """A warning when gravity compensation cannot hold the arm at these joints, else None."""
    gravity = np.asarray(plan.model.gravity_torques(np.radians(joints_deg)), dtype=float)
    needed = np.abs(gravity) / np.asarray(plan.config.alpha, dtype=float)
    short = [j for j in range(6) if needed[j] > plan.config.tau_cap_nm[j]]
    if not short:
        return None
    return ("gravity compensation cannot hold the arm at the start pose ("
            + ", ".join(f"J{j + 1} needs {needed[j]:.1f} > cap {plan.config.tau_cap_nm[j]:.1f} N*m" for j in short)
            + "): it sags there during a takeover; support it, or use a start it holds "
              "(quest_0, home_4a, home_cube_newhw, home_unscrew1)")


class FeedbackWatchdog:
    """Check actual source clocks, including each motor, rather than unchanged q/v."""

    def __init__(self, max_age_s=0.05):
        self.max_age_s = max_age_s
        self.seen = {}
        self.stale_sources = []

    def update(self, timestamps, now):
        self.stale_sources = []
        for name, stamp in timestamps.items():
            if not math.isfinite(stamp) or stamp <= 0:
                self.stale_sources.append(name)
                continue
            previous = self.seen.get(name)
            if previous is not None and stamp < previous[0]:
                self.stale_sources.append(name)
            elif previous is None or stamp > previous[0]:
                self.seen[name] = (stamp, now)
            elif now - previous[1] > self.max_age_s:
                self.stale_sources.append(name)
        return not self.stale_sources


class KinestheticTakeover:
    """The arm under gravity compensation in MIT torque mode, the hand limp; the operator moves both.

    Entry streams full gravity support immediately after selecting MIT mode,
    then releases a latched position hold over --kinesthetic-ramp-in-s.
    Physical support is still required across the firmware mode transition.
    The end pedal freezes the arm
    (brake, then a position hold with gravity retained), and it returns to
    position control at the measured joints in the order gravity_comp's
    goto_pose uses. Any error or Ctrl-C freezes first and never leaves the arm
    in MIT mode without a hold.
    """

    kind = "kinesthetic"

    def __init__(self, args, robot, plan):
        self.args = args
        self.robot = robot
        self.plan = plan
        hw = plan.gc.hw_session.HardwareSession(plan.driver, channel=args.piper_channel, calib=plan.artifact)
        hw.api = robot.piper  # The executor's connection; command scales and limits come from the calibration.
        hw._resolve_scales()
        self.hw = hw

    def read_feedback(self):
        js, ms = self.robot.piper.get_joint_state(), self.robot.piper.get_motor_state()
        if js is None or ms is None:
            raise RuntimeError("Missing Piper joint or motor feedback")
        stamps = getattr(ms, "timestamps", None)
        if stamps is None or len(stamps) != 6:
            raise RuntimeError("Piper feedback lacks six per-motor timestamps; update PiperAPI")
        clocks = {"joint": float(js.timestamp),
                  **{f"motor{i}": float(stamp) for i, stamp in enumerate(stamps, 1)}}
        sample = self.plan.gc.hw_session.StateSample(
            q=np.asarray(js.joints, dtype=float), v=np.asarray(ms.velocities, dtype=float),
            tau_fb=np.asarray(ms.torques, dtype=float), current=np.asarray(ms.currents, dtype=float),
            t_wall=time.monotonic(), feedback_hz=float(min(js.hz, ms.hz)),
            t_feedback=float(ms.timestamp), t_joint=float(js.timestamp))
        for values in (sample.q, sample.v, sample.tau_fb, sample.current):
            if values.shape != (6,) or not np.isfinite(values).all():
                raise RuntimeError("Invalid Piper feedback: expected six finite values")
        self.hw._last_q = sample.q.copy()
        return sample, clocks

    def preflight_feedback(self, watchdog):
        """Require every source clock to advance while still in position control."""
        sample, initial = self.read_feedback()
        watchdog.update(initial, time.monotonic())
        advanced = set()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            time.sleep(1.0 / self.args.kinesthetic_rate_hz)
            sample, clocks = self.read_feedback()
            fresh = watchdog.update(clocks, time.monotonic())
            advanced.update(name for name in clocks if clocks[name] > initial[name])
            if fresh and len(advanced) == len(initial):
                return sample
        missing = sorted(set(initial) - advanced)
        raise RuntimeError("Takeover rejected before MIT: feedback not advancing: "
                           + ", ".join(missing or watchdog.stale_sources))

    def describe(self):
        config = self.plan.config
        return {"gravity_calibration": str(self.plan.calibration),
                "gravity_calibration_sha256": self.plan.calibration_sha256,
                "gravity_comp_repo": str(self.plan.gc.root), "driver": self.plan.driver,
                "assist_frac": self.args.kinesthetic_assist_frac, "ramp_in_s": config.ramp_in_s,
                "entry": "full_gravity_with_releasing_position_hold",
                "feedback_watchdog": "joint_and_per_motor_timestamps",
                "rate_hz": self.args.kinesthetic_rate_hz, "hand": self.args.kinesthetic_hand,
                "tau_cap_nm": np.asarray(config.tau_cap_nm).tolist(),
                "underpowered_joints": self.plan.info.get("underpowered_joints", [])}

    def to_position_hold(self):
        """MIT hold -> position control at the measured joints (gravity_comp goto_pose order)."""
        piper = self.robot.piper
        piper.enable()
        piper.set_position_mode()
        for settle in (0.0, MODE_SWITCH_SETTLE_S):
            time.sleep(settle)
            joints = piper.get_joint_angles()
            if joints is not None:
                piper.move_joint(list(joints), mode="exact")
        if getattr(self.robot, "joint_control", None) is not None:
            self.robot.joint_control.q = None
        self.robot.commanded_pose = None

    def run(self, session):
        args, robot, hw = self.args, self.robot, self.hw
        control = self.plan.gc.control
        state_enum = control.DragState
        # Friction assistance stays off until the supported handoff is complete.
        handoff_config = replace(self.plan.config, assist_nm=np.zeros(6))
        controller = control.DragController(handoff_config, self.plan.model.gravity_torques)
        velocity_filter = control.VelocityFilter()
        watchdog = FeedbackWatchdog()
        end = EndPedals(session.pedals, args.discard_hold_s)
        rows = []
        session.add_table("gravity_comp.csv", GRAVITY_FIELDS, rows)
        # Check clocks and build the supported first command BEFORE changing torque modes.
        initial = self.preflight_feedback(watchdog)
        q0 = initial.q.copy()
        controller.start(q0)
        # Enter the controller's full-gravity phase before sending any commands.
        # The handoff below releases position stiffness, not gravity torque. Keeping
        # the controller ACTIVE also preserves full gravity if braking interrupts it.
        first = controller.update(q0, np.zeros(6), self.plan.config.ramp_in_s)
        first = replace(first, p_des=q0.copy(), kp=np.full(6, self.plan.config.hold_kp),
                        kd=np.full(6, self.plan.config.hold_kd))
        if args.kinesthetic_hand == "limp":
            robot.hand_writer.pause()
            with robot.hand_writer.lock:
                robot.hand.disable_torque()
            session.event("hand_limp")
        print(f"Kinesthetic takeover: SUPPORT THE ARM. Full gravity support with a position hold; "
              f"the hold releases over {self.plan.config.ramp_in_s:g} s. Wait for 'active' before teaching.", flush=True)
        sigint = {"count": 0}

        def on_sigint(signum, frame):
            sigint["count"] += 1

        previous_handler = signal.signal(signal.SIGINT, on_sigint)
        armed = False
        outcome = abort = None
        frozen = False
        try:
            session.event("mit_switch")
            # Do not use HardwareSession.set_mit_mode(): it sleeps 0.3 s with
            # no torque commands. Stream the supported command without that gap.
            armed = True
            robot.piper.set_mit_mode()
            sent = hw.send_mit(first)
            rows.append({"monotonic_ns": time.monotonic_ns(), "state": "HANDOFF",
                         "joint_timestamp_s": initial.t_joint, "motor_timestamp_s": initial.t_feedback,
                         **{f"motor_timestamp_{i}_s": watchdog.seen[f"motor{i}"][0] for i in range(1, 7)},
                         "stale_sources": "", "feedback_error": "",
                         **{f"kp_{i}": x for i, x in enumerate(first.kp, 1)},
                         **{f"kd_{i}": x for i, x in enumerate(first.kd, 1)},
                         **{f"q_{i}": f"{x:.6f}" for i, x in enumerate(q0, 1)},
                         **{f"v_{i}": f"{x:.6f}" for i, x in enumerate(initial.v, 1)},
                         **{f"tau_cmd_{i}": f"{x:.4f}" for i, x in enumerate(sent, 1)},
                         **{f"tau_fb_{i}": f"{x:.4f}" for i, x in enumerate(initial.tau_fb, 1)}})
            session.event("supported_handoff")
            period = 1.0 / args.kinesthetic_rate_hz
            next_tick = time.monotonic()
            handoff_start = next_tick
            next_view = next_tick
            last_t = hold_since = None
            last_sample = initial
            clocks = {}
            stale = cycles = overruns = 0
            while True:
                now = time.monotonic()
                sample = None
                reading = None
                clocks = {}
                feedback_error = ""
                try:
                    reading, clocks = self.read_feedback()
                    if watchdog.update(clocks, now):
                        sample = reading
                    else:
                        feedback_error = "stale sources: " + ", ".join(watchdog.stale_sources)
                except RuntimeError as exc:
                    feedback_error = str(exc)
                if sample is not None:
                    stale, last_sample = 0, sample
                else:
                    stale += 1
                    if stale >= STALE_FREEZE_CYCLES and abort is None:
                        abort = "stale Piper feedback: " + feedback_error
                        controller.freeze(abort)
                if last_sample is None:
                    q, v, tau_fb = q0, np.zeros(6), np.zeros(6)
                else:
                    q, v, tau_fb = last_sample.q, last_sample.v, last_sample.tau_fb
                dt = period if last_t is None else max(1e-4, now - last_t)
                last_t = now
                velocity = velocity_filter.update(v, dt)
                handoff = max(0.0, 1.0 - (now - handoff_start) / self.plan.config.ramp_in_s)
                if handoff == 0:
                    controller.config = self.plan.config
                command = controller.update(q, velocity, dt)
                if state_enum.ACTIVE is controller.state and handoff > 0:
                    command = replace(command, p_des=q0.copy(),
                                      kp=np.full(6, self.plan.config.hold_kp * handoff),
                                      kd=np.maximum(command.kd, self.plan.config.hold_kd * handoff))
                sent = hw.send_mit(command)
                state = controller.state
                cycles += 1
                rows.append({"monotonic_ns": time.monotonic_ns(),
                             "state": "HANDOFF" if state is state_enum.ACTIVE and handoff > 0 else state.name,
                             "joint_timestamp_s": clocks.get("joint", ""),
                             "motor_timestamp_s": reading.t_feedback if reading is not None else "",
                             **{f"motor_timestamp_{i}_s": clocks.get(f"motor{i}", "") for i in range(1, 7)},
                             "stale_sources": ",".join(watchdog.stale_sources), "feedback_error": feedback_error,
                             **{f"kp_{i}": x for i, x in enumerate(command.kp, 1)},
                             **{f"kd_{i}": x for i, x in enumerate(command.kd, 1)},
                             **{f"q_{i}": f"{x:.6f}" for i, x in enumerate(q, start=1)},
                             **{f"v_{i}": f"{x:.6f}" for i, x in enumerate(velocity, start=1)},
                             **{f"tau_cmd_{i}": f"{x:.4f}" for i, x in enumerate(sent, start=1)},
                             **{f"tau_fb_{i}": f"{x:.4f}" for i, x in enumerate(tau_fb, start=1)}})
                if state is state_enum.ACTIVE and session.episode is not None \
                        and session.episode.open_interval is None and not frozen and handoff == 0:
                    session.human_interval_start()
                    session.event("active")
                    print("Kinesthetic takeover active: position hold released; begin teaching.", flush=True)
                if sigint["count"] and abort is None:
                    abort = "Ctrl-C"
                    controller.freeze(abort)
                if outcome is None and abort is None:
                    outcome = end.update(session.pedals.poll())
                if (outcome is not None or end.pending or abort is not None) and not frozen:
                    frozen = True
                    session.human_interval_end()
                    if state in (state_enum.RAMP_IN, state_enum.ACTIVE):
                        controller.freeze("episode end" if abort is None else abort)
                    session.event("freeze", controller.freeze_reason or "")
                if not frozen and state in (state_enum.BRAKE, state_enum.FREEZE_HOLD):
                    # The controller froze itself (runaway): end the episode as a failure.
                    frozen = True
                    session.human_interval_end()
                    session.event("freeze", controller.freeze_reason or "")
                    print(f"Gravity compensation froze: {controller.freeze_reason}; ending the episode.", flush=True)
                    outcome = outcome or "failure"
                if controller.state is state_enum.FREEZE_HOLD:
                    hold_since = hold_since or now
                    if (outcome is not None or abort is not None) and now - hold_since >= self.plan.config.hold_ramp_s:
                        break
                if now >= next_view:
                    refresh_views(robot, session.camera, session.view)
                    next_view = now + VIEW_REFRESH_S
                next_tick += period
                remaining = next_tick - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    overruns += 1
                    next_tick = time.monotonic()
            session.event("hold", f"cycles={cycles} overruns={overruns}")
            if cycles and overruns / cycles > 0.2:
                print(f"WARNING: {overruns / cycles:.0%} gravity-compensation loop overruns at "
                      f"{args.kinesthetic_rate_hz:g} Hz", flush=True)
        except BaseException as exc:
            if armed:
                print(f"\n*** kinesthetic takeover aborted: {type(exc).__name__}: {exc}", flush=True)
                try:
                    controller.freeze(f"exception: {type(exc).__name__}")
                except Exception:
                    pass
                try:
                    self.plan.gc.hw_session.damp_brake(hw, controller._gravity_ff)
                except Exception as brake_exc:
                    print(f"*** damp brake failed ({brake_exc!r}); SUPPORT THE ARM.", flush=True)
                self.to_position_hold()
                session.event("position_mode", "after abort")
            raise
        finally:
            signal.signal(signal.SIGINT, previous_handler)
        self.to_position_hold()
        session.event("position_mode")
        if abort == "Ctrl-C":
            raise KeyboardInterrupt
        if abort is not None:
            raise RuntimeError(f"Kinesthetic takeover stopped: {abort}")
        return outcome


# ---------------------------------------------------------------------------
# Command line and construction


def add_arguments(parser):
    group = parser.add_argument_group("HG-DAgger (--dagger)")
    group.add_argument("--dagger", type=Path, metavar="DATASET_ROOT",
                       help="Record each rollout to DATASET_ROOT/raw/episode_N; the takeover pedal hands the "
                            "rest of the episode to the operator (inference/executor/dagger.py)")
    group.add_argument("--takeover", choices=TAKEOVER_MODES,
                       help="With --dagger: defaults to kinesthetic (gravity comp); teleop needs leader/Quest options")
    group.add_argument("--dagger-round", type=int, default=None, help="DAgger round number saved with each episode")
    group.add_argument("--pedal-keys", nargs=3, default=list(DEFAULT_PEDAL_KEYS),
                       metavar=("TAKEOVER", "SUCCESS", "FAILURE"),
                       help="Keys (names or codes) of the three pedals (default: %(default)s); check them with "
                            "python -m scripts.piper_api.foot_pedal")
    group.add_argument("--discard-hold-s", type=float, default=2.0,
                       help="Hold the failure pedal this long to discard the episode instead (default: %(default)s)")
    group.add_argument("--episode-max-s", type=float, default=0.0,
                       help="End a policy-only rollout as a failure after this long; 0 = no limit")
    group.add_argument("--dagger-hand-hz", type=float, default=50.0,
                       help="Follower hand recording rate (default: %(default)s, as collection)")
    teleop = parser.add_argument_group("teleop takeover (--takeover teleop)")
    teleop.add_argument("--teleop-calibration", type=Path, help="Quest calibration JSON (fit_quest_calibration.py)")
    teleop.add_argument("--quest-host", default="0.0.0.0")
    teleop.add_argument("--quest-port", type=int, default=5005)
    teleop.add_argument("--leader-port", help="Serial port of the worn leader TriMI")
    teleop.add_argument("--leader-config", type=Path, help="Leader TriMI YAML (the one the base dataset used)")
    teleop.add_argument("--leader-hz", type=float, default=100.0, help="Leader read rate (default: %(default)s)")
    teleop.add_argument("--teleop-hz", type=float, default=60.0, help="Arm teleop command rate (default: %(default)s)")
    teleop.add_argument("--teleop-scale", type=float, default=1.0, help="Robot mm per Quest mm")
    teleop.add_argument("--teleop-position-only", action="store_true", help="Do not follow the controller rotation")
    teleop.add_argument("--teleop-max-offset-mm", type=float, default=300.0,
                        help="Clamp targets to this distance from the takeover pose (default: %(default)s)")
    teleop.add_argument("--teleop-max-rotation-deg", type=float, default=90.0,
                        help="Clamp target rotation from the takeover pose (default: %(default)s)")
    teleop.add_argument("--teleop-max-speed-mm-s", type=float, default=300.0)
    teleop.add_argument("--teleop-max-angular-speed-deg-s", type=float, default=180.0)
    teleop.add_argument("--takeover-hand-ramp-s", type=float, default=0.4,
                        help="Blend the follower from the policy's last hand target to the leader over this long")
    teleop.add_argument("--quest-timeout-s", type=float, default=0.15)
    teleop.add_argument("--quest-jump-mm", type=float, default=50.0)
    teleop.add_argument("--max-tracking-error-mm", type=float, default=20.0)
    teleop.add_argument("--tracking-lag-s", type=float, default=0.3)
    teleop.add_argument("--tracking-error-hold-s", type=float, default=0.3)
    kinesthetic = parser.add_argument_group("kinesthetic takeover (--takeover kinesthetic)")
    kinesthetic.add_argument("--gravity-comp-repo", type=Path, default=None,
                             help="midas-piper-control checkout with gravity_comp/ "
                                  "(otherwise TRIMORPH_GRAVITY_COMP_REPO, sibling checkout, then known home paths)")
    kinesthetic.add_argument("--gravity-calib", type=Path,
                             default=Path(__file__).resolve().parents[2] / "scripts" / "grav_comp" / "calib"
                             / "robot1_trimi.json", help="Gravity calibration artifact (default: %(default)s)")
    kinesthetic.add_argument("--kinesthetic-rate-hz", type=float, default=200.0)
    kinesthetic.add_argument("--kinesthetic-ramp-in-s", type=float, default=1.0,
                             help="Supported handoff: release position stiffness over this duration while "
                                  "keeping full gravity compensation (at least 1 s)")
    kinesthetic.add_argument("--kinesthetic-assist-frac", type=float, default=0.0,
                             help="Friction assist as a fraction of the measured Coulomb friction (<= 0.6; "
                                  "default %(default)s: none until drag_mode has validated one)")
    kinesthetic.add_argument("--kinesthetic-hand", choices=("limp", "hold"), default="limp",
                             help="limp: hand torque off so the operator moves the fingers (default); "
                                  "hold: keep the policy's last hand target")
    kinesthetic.add_argument("--allow-underpowered", action=argparse.BooleanOptionalAction, default=None,
                             help="Allow capped gravity support (default on for kinesthetic DAgger; "
                                  "--no-allow-underpowered restores strict checking)")
    kinesthetic.add_argument("--gravity-force", action=argparse.BooleanOptionalAction, default=None,
                             help="Accept failed friction gates (default on for kinesthetic DAgger; "
                                  "--no-gravity-force restores strict checking)")


def apply_defaults(args, argv):
    """Apply only DAgger-specific defaults; leave executor settings unchanged."""
    from data_collection.collection import config as defaults
    if args.dagger is not None:
        args.takeover = args.takeover or "kinesthetic"
    kinesthetic = args.dagger is not None and args.takeover == "kinesthetic"
    if args.allow_underpowered is None:
        args.allow_underpowered = kinesthetic and defaults.DAGGER_ALLOW_UNDERPOWERED
    if args.gravity_force is None:
        args.gravity_force = kinesthetic and defaults.DAGGER_GRAVITY_FORCE


def check_arguments(parser, args):
    if args.dagger is None:
        return
    if args.takeover is None:
        parser.error("--dagger needs --takeover teleop or --takeover kinesthetic")
    for flag, bad in (("--fake", args.fake), ("--no-pedal", args.no_pedal), ("--no-log", args.no_log),
                      ("--record-video", args.record_video)):
        if bad:
            parser.error(f"--dagger cannot be combined with {flag}")
    if args.arm_control != "joint":
        parser.error("--dagger needs --arm-control joint")
    try:
        pedal_codes(args.pedal_keys)
    except ValueError as exc:
        parser.error(str(exc))
    positive = ["discard_hold_s", "dagger_hand_hz"]
    if args.takeover == "teleop":
        missing = [flag for flag, value in (("--teleop-calibration", args.teleop_calibration),
                                            ("--leader-port", args.leader_port),
                                            ("--leader-config", args.leader_config)) if value is None]
        if missing:
            parser.error(f"--takeover teleop needs {', '.join(missing)}")
        positive += ["leader_hz", "teleop_hz", "teleop_scale", "teleop_max_offset_mm", "teleop_max_rotation_deg",
                     "teleop_max_speed_mm_s", "teleop_max_angular_speed_deg_s", "quest_timeout_s", "quest_jump_mm",
                     "max_tracking_error_mm"]
        if args.teleop_scale > 3:
            parser.error("--teleop-scale is limited to 3")
        if args.takeover_hand_ramp_s < 0 or args.tracking_lag_s < 0 or args.tracking_error_hold_s < 0:
            parser.error("--takeover-hand-ramp-s, --tracking-lag-s and --tracking-error-hold-s must be >= 0")
    else:
        positive.append("kinesthetic_rate_hz")
        if args.kinesthetic_ramp_in_s < 1:
            parser.error("--kinesthetic-ramp-in-s must be at least 1 (gravity_comp's DragConfig rule)")
        if not 0 <= args.kinesthetic_assist_frac <= 0.6:
            parser.error("--kinesthetic-assist-frac must be in [0, 0.6]")
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.episode_max_s < 0:
        parser.error("--episode-max-s must be >= 0")


def start_joints_deg(args):
    """The executor's start joints (as executor.start_base, without the start jitter)."""
    from inference.executor.executor import start_base

    return np.asarray(start_base(args)[0], dtype=float)


def prepare(args):
    """Load what the takeover needs before any hardware moves, so a bad calibration fails early."""
    if args.takeover == "teleop":
        return {"calibration": json.loads(Path(args.teleop_calibration).expanduser().read_text())}
    return {"kinesthetic": prepare_kinesthetic(args)}


def create_pedals(args):
    from scripts.piper_api.foot_pedal import FootPedalTrigger

    # No trigger debounce: it would drop a second pedal's press; DaggerPedals debounces per role.
    trigger = FootPedalTrigger(args.foot_pedal_device, debounce_s=0.0)
    trigger.start()
    return DaggerPedals(trigger, pedal_codes(args.pedal_keys))


def create_session(args, prepared, robot, camera, pedals, log):
    from data_collection.collection import config as defaults

    if robot.piper_log is None:
        raise RuntimeError("--dagger needs the Piper feedback log")
    if args.takeover == "kinesthetic":
        warning = start_sag_warning(prepared["kinesthetic"], start_joints_deg(args))
        if warning:
            print(f"WARNING: {warning}", flush=True)
    video = follower = leader = leader_hand = quest = None
    try:
        video = EpisodeVideo(camera, defaults.VIDEO_CODEC)
        follower = HandSampler(robot.hand.read_pos, args.dagger_hand_hz, "follower", lock=robot.hand_writer.lock,
                               target=lambda: robot.hand_writer.target)
        if args.takeover == "teleop":
            from scripts.teleop.quest_teleop import QuestStream

            leader_hand = open_leader_hand(args)
            leader = HandSampler(leader_hand.read_pos, args.leader_hz, "leader")
            quest = QuestStream(args.quest_host, args.quest_port)
            takeover = TeleopTakeover(args, robot, leader, quest, prepared["calibration"],
                                      calibration_path=args.teleop_calibration)
            label_hand = leader
        else:
            takeover = KinestheticTakeover(args, robot, prepared["kinesthetic"])
            label_hand = follower
        return DaggerSession(args, robot, camera, pedals, takeover, video=video, follower=follower,
                             label_hand=label_hand, log=log, quest=quest, leader_hand=leader_hand)
    except BaseException:
        for part in (follower, leader, quest):
            if part is not None:
                part.stop()
        if leader_hand is not None:
            leader_hand.shutdown()
        raise
