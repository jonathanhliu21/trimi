"""HG-DAgger executor mode without hardware: pedals, rollout control, recording, takeovers, export."""
import contextlib
import csv
import io
import json
import os
import signal
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from inference import protocol
from inference.executor import dagger
from inference.executor.executor import FakeCamera, FakeRobot, RunLog, parse_args, run, run_rtc
from scripts.piper_api.foot_pedal import EV_KEY, INPUT_EVENT_STRUCT, KEY_PRESS, KEY_RELEASE, FootPedalTrigger

GRAVITY_COMP_REPO = Path.home() / "workspace" / "midas-piper-control"
if not (GRAVITY_COMP_REPO / "gravity_comp" / "control.py").is_file():
    GRAVITY_COMP_REPO = Path.home() / "midas_hand" / "midas-piper-control"


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def dagger_args(root, *extra, **overrides):
    args = parse_args(["--predictor", "localhost", "--dagger", str(root), *extra])
    args.start_log = Path(root) / "no_starts.jsonl"
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


class ScriptedPedals:
    """poll() returns the scripted role lists in order; held_s() reads a dict (or a callable)."""

    def __init__(self, script=(), held=None):
        self.script = list(script)
        table = {} if held is None else held  # a live dict: tests change it while running
        self.held = held if callable(held) else table.get
        self.polls = 0
        self.stopped = False

    def poll(self):
        self.polls += 1
        return self.script.pop(0) if self.script else []

    def held_s(self, role):
        return self.held(role)

    def describe(self):
        return "scripted"

    def stop(self):
        self.stopped = True


class FakeVideo:
    def __init__(self):
        self.path = None

    def begin(self, path):
        self.path = Path(path)
        self.path.write_bytes(b"")
        self.started = time.monotonic_ns()

    def end(self):
        return [{"frame_idx": 0, "monotonic_ns": self.started, "receipt_monotonic_ns": self.started,
                 "sequence_num": 0},
                {"frame_idx": 1, "monotonic_ns": time.monotonic_ns(), "receipt_monotonic_ns": time.monotonic_ns(),
                 "sequence_num": 1}]


class FakeSampler:
    def __init__(self, value=0.0):
        self.value = value
        self.latest = (time.monotonic_ns(), np.full(13, value))
        self.recording = False
        self.stopped = False

    def begin(self):
        self.recording = True

    def end(self):
        self.recording = False
        return [(time.monotonic_ns(), np.full(13, self.value), np.zeros(13))]

    def stop(self):
        self.stopped = True


class DaggerFakeRobot(FakeRobot):
    """FakeRobot with the Piper log rows, hand writer and return counter a DAgger session reads."""

    def __init__(self):
        super().__init__()
        self.piper_log = SimpleNamespace(rows=[])
        self.hand_writer = Mock(target=None)
        self.applied = 0
        self.returns = 0

    def apply_action(self, action):
        super().apply_action(action)
        self.applied += 1
        self.piper_log.rows.append({"monotonic_ns": time.monotonic_ns(),
                                    **{f"joint_{i}": "0.1" for i in range(1, 7)}, "fw_x": "0.3"})

    def return_to_start(self, args):
        self.returns += 1


def fake_takeover(outcome="success", kind="kinesthetic"):
    def run_takeover(session):
        session.human_interval_start()
        session.event("active")
        time.sleep(0.001)
        return outcome

    return SimpleNamespace(kind=kind, run=Mock(side_effect=run_takeover), describe=lambda: {"fake": True})


def make_session(root, robot, pedals, takeover=None, *, args=None, log=None):
    args = args or dagger_args(root, "--takeover", "kinesthetic")
    return dagger.DaggerSession(args, robot, FakeCamera(), pedals, takeover or fake_takeover(),
                                video=FakeVideo(), follower=FakeSampler(), label_hand=None,
                                log=log if log is not None else RunLog(Path(root) / "runs"),
                                fk=lambda q: np.r_[np.asarray(q[:3]), 0.0, 0.0, 0.0])


class PedalTests(unittest.TestCase):
    def feed(self, pedal, *events):
        pedal._fd = 3
        pedal._buffer = b"".join(INPUT_EVENT_STRUCT.pack(0, 0, EV_KEY, code, value) for code, value in events)

    def test_trigger_tracks_each_key_held_on_its_own(self):
        pedal = FootPedalTrigger("/dev/null", debounce_s=0.0)
        with patch.object(FootPedalTrigger, "_drain_raw_events"):
            self.feed(pedal, (48, KEY_PRESS), (46, KEY_PRESS))
            self.assertEqual([event.code for event in pedal.poll()], [48, 46])
            self.assertIsNotNone(pedal.held_s(48))
            self.assertIsNotNone(pedal.held_s(46))
            self.assertIsNone(pedal.held_s(30))
            self.feed(pedal, (46, KEY_RELEASE))
            pedal.poll()
            self.assertIsNotNone(pedal.held_s(48))
            self.assertIsNone(pedal.held_s(46))
        pedal._fd = None

    def test_roles_map_codes_and_debounce_per_role(self):
        trigger = Mock()
        trigger.poll.side_effect = [[SimpleNamespace(code=48), SimpleNamespace(code=46), SimpleNamespace(code=99)],
                                    [SimpleNamespace(code=48), SimpleNamespace(code=30)]]
        pedals = dagger.DaggerPedals(trigger, dagger.pedal_codes(["b", "c", "a"]), debounce_s=10.0)
        self.assertEqual(pedals.poll(), ["takeover", "success"])  # unknown code ignored
        self.assertEqual(pedals.poll(), ["failure"])  # takeover again inside its debounce
        trigger.held_s.return_value = 1.5
        self.assertEqual(pedals.held_s("failure"), 1.5)
        trigger.held_s.assert_called_with(30)
        with self.assertRaisesRegex(ValueError, "different keys"):
            dagger.pedal_codes(["a", "a", "b"])

    def test_failure_pedal_ends_on_release_and_discards_when_held(self):
        held = {}
        end = dagger.EndPedals(ScriptedPedals(held=held), discard_hold_s=2.0)
        held["failure"] = 0.5
        self.assertIsNone(end.update(["failure"]))
        self.assertTrue(end.pending)
        self.assertIsNone(end.update([]))
        held["failure"] = 2.5
        self.assertEqual(end.update([]), "discard")
        held.clear()
        self.assertEqual(dagger.EndPedals(ScriptedPedals(held=held), 2.0).update(["failure"]), "failure")
        self.assertEqual(dagger.EndPedals(ScriptedPedals(held=held), 2.0).update(["success"]), "success")

    def test_take_rows_keeps_rows_of_the_interval_and_drops_older_ones(self):
        rows = [{"monotonic_ns": t} for t in (1, 2, 3, 4, 5)]
        self.assertEqual([r["monotonic_ns"] for r in dagger.take_rows(rows, 2, 4)], [2, 3, 4])
        self.assertEqual(rows, [{"monotonic_ns": 5}])


class ArgumentTests(unittest.TestCase):
    def test_dagger_options_are_checked(self):
        with tempfile.TemporaryDirectory() as root:
            args = parse_args(["--predictor", "h", "--dagger", root, "--takeover", "kinesthetic"])
            self.assertEqual(args.pedal_keys, ["a", "b", "c"])
            self.assertEqual(args.kinesthetic_assist_frac, 0.0)
            for bad in (["--dagger", root, "--takeover", "teleop"],
                        ["--dagger", root, "--takeover", "kinesthetic", "--no-log"],
                        ["--dagger", root, "--takeover", "kinesthetic", "--fake"],
                        ["--dagger", root, "--takeover", "kinesthetic", "--arm-control", "pose"],
                        ["--dagger", root, "--takeover", "kinesthetic", "--pedal-keys", "a", "a", "c"],
                        ["--dagger", root, "--takeover", "kinesthetic", "--kinesthetic-ramp-in-s", "0.5"],
                        ["--dagger", root, "--takeover", "kinesthetic", "--kinesthetic-assist-frac", "0.7"]):
                with self.subTest(bad=bad), contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit):
                    parse_args(["--predictor", "h", *bad])
            teleop = parse_args(["--predictor", "h", "--dagger", root, "--takeover", "teleop",
                                 "--teleop-calibration", "cal.json", "--leader-port", "/dev/x",
                                 "--leader-config", "leader.yaml"])
            self.assertFalse(teleop.teleop_position_only)
            self.assertIsNone(parse_args(["--predictor", "h"]).dagger)

    def test_minimal_dagger_defaults_and_explicit_overrides(self):
        args = parse_args(["--predictor", "h", "--dagger", "round1"])
        self.assertEqual(args.takeover, "kinesthetic")
        plain = parse_args(["--predictor", "h"])
        dagger_keys = {"dagger", "takeover", "allow_underpowered", "gravity_force"}
        for key, value in vars(plain).items():
            if key not in dagger_keys:
                self.assertEqual(getattr(args, key), value, key)
        self.assertFalse(args.rtc)
        self.assertTrue(args.allow_underpowered)
        self.assertTrue(args.gravity_force)
        self.assertFalse(args.show)
        self.assertIsNone(args.dagger_round)
        args = parse_args(["--predictor", "h", "--dagger=round1", "--async-inference",
                           "--start-home=home_cube_newhw", "--hand-config=custom.yaml", "--hand-port=/dev/custom",
                           "--rtc-max-guidance=2", "--pedal-keys", "c", "b", "a",
                           "--no-allow-underpowered", "--no-gravity-force", "--gravity-comp-repo=/tmp/gc"])
        self.assertFalse(args.rtc)
        self.assertTrue(args.async_inference)
        self.assertEqual(args.start_home, "home_cube_newhw")
        self.assertEqual(args.hand_config, Path("custom.yaml"))
        self.assertEqual(args.hand_port, "/dev/custom")
        self.assertEqual(args.rtc_max_guidance, 2)
        self.assertEqual(args.pedal_keys, ["c", "b", "a"])
        self.assertFalse(args.allow_underpowered)
        self.assertFalse(args.gravity_force)
        self.assertEqual(args.gravity_comp_repo, Path("/tmp/gc"))
        plain = parse_args(["--predictor", "h"])
        self.assertFalse(plain.rtc)
        self.assertFalse(plain.allow_underpowered)
        self.assertFalse(plain.gravity_force)

    def test_gravity_repo_discovery_precedence(self):
        with patch.dict(os.environ, {"TRIMORPH_GRAVITY_COMP_REPO": "/tmp/environment-gc"}):
            self.assertEqual(dagger.gravity_comp_candidates("/tmp/explicit-gc"), [Path("/tmp/explicit-gc")])
            self.assertEqual(dagger.gravity_comp_candidates(), [Path("/tmp/environment-gc")])
        with patch.dict(os.environ, {}, clear=True):
            candidates = dagger.gravity_comp_candidates()
            self.assertEqual(candidates[0].name, "midas-piper-control")
            self.assertIn(Path.home() / "midas_hand" / "midas-piper-control", candidates)

    def test_action_reply_header_carries_policy_provenance(self):
        left, right = socket.socketpair()
        with left, right:
            protocol.send_action(left, 3, np.zeros((2, 19)), infer_ms=1.5, extra={"policy": {"sha256": "abc"}})
            seq, chunk, header = protocol.recv_action(right)
        self.assertEqual((seq, chunk.shape, header["policy"]), (3, (2, 19), {"sha256": "abc"}))
        log = RunLog(tempfile.gettempdir())
        log.note_header(header)
        self.assertEqual(log.policy, {"sha256": "abc"})


class SessionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def episode(self, index):
        path = self.root / "raw" / f"episode_{index:06d}"
        return path, json.loads((path / "metadata.json").read_text())

    def test_sync_policy_takeover_saves_the_rollout_and_starts_the_next(self):
        robot = DaggerFakeRobot()
        # Polls: start (discard, press) | run: discard, cycle 0 top, action 0, action 1 (press = takeover)
        # | next start (discard, press) | cycle 1 top, its 3 actions.
        pedals = ScriptedPedals([[], ["takeover"], [], [], [], ["takeover"], [], ["takeover"]])
        takeover = fake_takeover("success")
        args = dagger_args(self.root, "--takeover", "kinesthetic", hz=1000.0, max_cycles=2)
        session = make_session(self.root, robot, pedals, takeover, args=args)
        chunk = np.zeros((3, 19))
        with patch("inference.executor.executor.protocol.send_obs"), \
             patch("inference.executor.executor.protocol.recv_action",
                   side_effect=[(0, chunk, {"policy": {"sha256": "p1"}}), (1, chunk, {})]), \
             patch("inference.executor.executor.time.sleep"), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            session.wait_for_start(robot, FakeCamera())
            run(args, robot, FakeCamera(), Mock(), session, None, session.log)
            session.close()
        takeover.run.assert_called_once_with(session)
        self.assertEqual(robot.returns, 1)
        self.assertEqual(robot.applied, 1 + 3)  # nothing from chunk 0 after the takeover
        path, metadata = self.episode(0)
        self.assertEqual(metadata["status"], "complete")
        info = metadata["dagger"]
        self.assertEqual((info["outcome"], info["takeover"], info["policy"]), ("success", "kinesthetic",
                                                                                {"sha256": "p1"}))
        self.assertEqual([s["source"] for s in info["segments"]], ["policy", "human"])
        self.assertEqual(len(info["human_intervals"]), 1)
        self.assertLessEqual(info["segments"][1]["start_ns"], info["human_intervals"][0][0])
        for name in ("camera.mp4", "camera_timestamps.csv", "leader_hand.csv", "follower.csv", "piper.csv",
                     "commands.csv", "events.csv", "chunks.npz"):
            self.assertTrue((path / name).is_file(), name)
        piper = read_csv(path / "piper.csv")
        self.assertEqual(len(piper), 1)
        self.assertAlmostEqual(float(piper[0]["x"]), 0.1)  # FK of the measured joints
        self.assertEqual(piper[0]["fw_x"], "0.3")
        self.assertEqual([row["source"] for row in read_csv(path / "commands.csv")], ["policy"])
        self.assertIn("takeover", [row["event"] for row in read_csv(path / "events.csv")])
        self.assertEqual(len(read_csv(path / "leader_hand.csv")), 1)
        _, second = self.episode(1)
        self.assertEqual(second["status"], "interrupted")
        self.assertEqual(second["dagger"]["segments"], [{"source": "policy", "start_ns": second["dagger"]["segments"]
                                                         [0]["start_ns"], "end_ns": second["dagger"]["segments"][0]
                                                         ["end_ns"]}])
        self.assertIn("Saved episode 0: success, complete", output.getvalue())
        self.assertTrue(pedals.stopped)

    def test_rtc_takeover_drops_the_pending_reply_and_restarts_without_overlap(self):
        robot = DaggerFakeRobot()
        # Polls: start (discard, press) | run_rtc: discard, tick 1, tick 2 (press = takeover)
        # | next start (discard, press).
        pedals = ScriptedPedals([[], ["takeover"], [], [], ["takeover"], [], ["takeover"]])
        args = dagger_args(self.root, "--takeover", "kinesthetic", "--rtc", hz=10.0, max_cycles=3,
                           rtc_horizon=1)
        order = []
        takeover = fake_takeover("success")
        takeover.run.side_effect = lambda session: (order.append("takeover"), "success")[1]
        session = make_session(self.root, robot, pedals, takeover, args=args)
        chunks = [np.zeros((4, 19), np.float32) for _ in range(3)]
        replies = iter([(seq, chunk, {"infer_ms": 1.0}) for seq, chunk in enumerate(chunks)])

        def recv(sock):
            order.append("recv")
            return next(replies)

        with patch("inference.executor.executor.protocol.send_obs") as send_obs, \
             patch("inference.executor.executor.protocol.recv_action", side_effect=recv), \
             patch("inference.executor.executor.reply_ready", return_value=False), \
             patch("inference.executor.executor.time.sleep"), \
             contextlib.redirect_stdout(io.StringIO()):
            session.wait_for_start(robot, FakeCamera())
            run_rtc(args, robot, FakeCamera(), Mock(), session, None, session.log)
            session.close()
        # The operator gets control before the reply in flight is drained (unexecuted).
        self.assertEqual(order, ["recv", "takeover", "recv", "recv"])
        self.assertEqual(send_obs.call_args_list[2].kwargs["rtc"]["prev_chunk"].shape, (0, 19))
        self.assertEqual(self.episode(0)[1]["dagger"]["outcome"], "success")

    def test_policy_only_rollouts_end_by_pedal_or_timeout_and_held_failure_discards(self):
        held = {}
        robot = DaggerFakeRobot()
        pedals = ScriptedPedals(held=held)
        session = make_session(self.root, robot, pedals)
        with contextlib.redirect_stdout(io.StringIO()):
            pedals.script = [[], ["takeover"]]
            session.wait_for_start(robot, FakeCamera())
            session.pending = ["success"]
            pedals.script = [[], ["takeover"]]
            session.interrupt(session.args, robot, FakeCamera())  # episode 0: success, then episode 1 starts
            held["failure"] = 0.1
            session.pending = ["failure"]
            pedals.script = [[], [], [], ["takeover"]]
            with patch("inference.executor.dagger.time.sleep", side_effect=lambda s: held.clear()):
                session.interrupt(session.args, robot, FakeCamera())  # episode 1: released = failure
            held["failure"] = 5.0
            session.pending = ["failure"]
            pedals.script = [[], ["takeover"]]
            session.interrupt(session.args, robot, FakeCamera())  # episode 2: held = discarded
            session.args.episode_max_s = 1.0
            session.episode.start_s -= 2.0
            self.assertEqual(session.poll(), ["timeout"])
            pedals.script = [[], ["takeover"]]
            session.interrupt(session.args, robot, FakeCamera())  # episode 2 again: timeout
            session.close()
        self.assertEqual(self.episode(0)[1]["dagger"]["outcome"], "success")
        self.assertEqual(self.episode(1)[1]["dagger"]["outcome"], "failure")
        timeout = self.episode(2)[1]
        self.assertEqual((timeout["dagger"]["outcome"], timeout["dagger"]["reason"]), ("failure", "timeout"))
        self.assertEqual(self.episode(0)[1]["dagger"]["segments"][0]["source"], "policy")
        self.assertEqual(len(self.episode(0)[1]["dagger"]["segments"]), 1)
        self.assertEqual(session.counts["discard"], 1)
        self.assertEqual(sorted(p.name for p in (self.root / "raw").iterdir()),
                         ["episode_000000", "episode_000001", "episode_000002", "episode_000003"])
        self.assertEqual(self.episode(3)[1]["status"], "interrupted")


class TeleopTakeoverTests(unittest.TestCase):
    def test_arm_follows_quest_deltas_from_the_commanded_pose_and_holds_on_release(self):
        from data_collection.collection.config import PIPER_START_JOINTS_DEG
        from inference.executor.joint_control import JointController

        with tempfile.TemporaryDirectory() as root:
            args = dagger_args(root, "--takeover", "teleop", "--teleop-calibration", "cal.json",
                               "--leader-port", "/dev/x", "--leader-config", "leader.yaml",
                               "--takeover-hand-ramp-s", "0.2", "--teleop-hz", "100")
        control = JointController()
        start = np.radians(PIPER_START_JOINTS_DEG["home_cube_newhw"])
        control.reset(start)
        robot = SimpleNamespace(joint_control=control, piper=Mock(), commanded_pose=np.asarray(control.fk(start)),
                                hand_writer=Mock(target=np.zeros(13)))
        robot.eef_pose = lambda: robot.commanded_pose  # perfect tracking
        anchor = robot.commanded_pose.copy()
        tick = {"n": 0}
        quest_position = np.zeros(3)

        def quest_get():
            if 5 <= tick["n"] < 25:
                quest_position[0] += 0.001  # the operator moves 1 mm per tick while engaged
            return time.monotonic(), quest_position.copy(), np.array([0.0, 0.0, 0.0, 1.0]), tick["n"]

        def poll():
            tick["n"] += 1
            return ["success"] if tick["n"] == 40 else []

        pedals = ScriptedPedals(held=lambda role: 0.1 if role == "takeover" and tick["n"] < 30 else None)
        pedals.poll = poll
        quest = SimpleNamespace(get=quest_get)
        leader = SimpleNamespace(latest=(time.monotonic_ns(), np.full(13, 0.2)))
        calibration = {"linear_matrix": np.eye(3).tolist(), "orientation": {"rotation_matrix": np.eye(3).tolist()}}
        session = SimpleNamespace(pedals=pedals, camera=FakeCamera(), view=None, events=[], commands=[],
                                  intervals=[])
        session.event = lambda name, detail="": session.events.append(name)
        session.human_interval_start = lambda: session.intervals.append("start")
        session.human_interval_end = lambda: session.intervals.append("end")
        session.teleop_command = session.commands.append
        takeover = dagger.TeleopTakeover(args, robot, leader, quest, calibration)
        with contextlib.redirect_stdout(io.StringIO()):
            outcome = takeover.run(session)
        self.assertEqual(outcome, "success")
        first = np.asarray(robot.piper.move_joint.call_args_list[0].args[0])
        np.testing.assert_allclose(first, start, atol=1e-6)  # no jump at the takeover
        x = np.array([float(row["x"]) for row in session.commands])
        self.assertAlmostEqual(x[0], anchor[0], places=6)
        self.assertAlmostEqual(x[29] - anchor[0], 0.019, delta=0.002)  # 19 mm of Quest motion while engaged
        np.testing.assert_allclose(x[30:], x[29], atol=1e-9)  # released: the arm holds
        self.assertEqual(session.intervals, ["start", "end"])
        self.assertEqual(session.events[:3], ["teleop", "engage", "disengage"])
        hands = [call.args[0] for call in robot.hand_writer.submit.call_args_list]
        self.assertLess(hands[0][0], 0.2)  # ramps from the policy's last target
        np.testing.assert_allclose(hands[-1], 0.2)
        self.assertTrue(all(row["source"] == "teleop" for row in session.commands))


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def monotonic(self):
        return self.t

    def monotonic_ns(self):
        return int(self.t * 1e9)

    def sleep(self, seconds):
        self.t += max(seconds, 0.0)

    def perf_counter(self):
        return self.t

    def time(self):
        return self.t


class FakeArmApi:
    """The PiperAPI surface HardwareSession and the takeover use, with fresh feedback on every read."""

    def __init__(self, on_read=None):
        self.stamp = 0.0
        self.calls = []
        self.torques = []
        self.command_times = []
        self.on_read = on_read
        self.reads = 0

    def get_joint_state(self):
        self.stamp += 0.001
        self.reads += 1
        if self.on_read is not None:
            self.on_read(self.reads)
        return SimpleNamespace(joints=[0.1] * 6, timestamp=self.stamp, hz=300.0)

    def get_motor_state(self):
        return SimpleNamespace(velocities=[0.0] * 6, torques=[0.0] * 6, currents=[0.0] * 6,
                               timestamp=self.stamp, timestamps=[self.stamp] * 6, hz=300.0)

    def move_mit_torque(self, joint, torque, **kwargs):
        self.torques.append((joint, torque, kwargs))
        self.command_times.append(dagger.time.monotonic())
        return torque

    def set_mit_mode(self):
        self.calls.append("set_mit_mode")
        self.mode_time = dagger.time.monotonic()

    def get_joint_angles(self):
        return [0.1] * 6

    def __getattr__(self, name):  # set_mit_mode, set_position_mode, enable, move_joint
        def call(*args, **kwargs):
            self.calls.append(name)
            return True
        return call


class KinestheticTakeoverTests(unittest.TestCase):
    def setUp(self):
        if not (GRAVITY_COMP_REPO / "gravity_comp" / "control.py").is_file():
            self.skipTest("midas-piper-control gravity_comp not available")
        if str(GRAVITY_COMP_REPO) not in sys.path:
            sys.path.append(str(GRAVITY_COMP_REPO))
        from gravity_comp import control, hw_session

        self.gc = SimpleNamespace(control=control, hw_session=hw_session, root=GRAVITY_COMP_REPO)
        config = control.DragConfig(alpha=np.ones(6), assist_nm=np.zeros(6), coulomb_nm=np.ones(6),
                                    deadband_rad_s=0.02, kd_soft=np.full(6, 0.05), tau_cap_nm=np.full(6, 5.0),
                                    firmware_limits_nm=np.full(6, 8.0), joint_limits=np.tile([-3.0, 3.0], (6, 1)),
                                    ramp_in_s=1.0)
        artifact = SimpleNamespace(data={"firmware": {}}, output_scales=lambda: [1.0] * 6,
                                   torque_limits=lambda: [8.0] * 6)
        self.plan = dagger.KinestheticPlan(gc=self.gc, artifact=artifact,
                                           model=SimpleNamespace(gravity_torques=lambda q: np.full(6, 0.5)),
                                           config=config, info={}, driver="v183", calibration=Path("calib.json"),
                                           calibration_sha256="x")

    def run_takeover(self, api, end_at_s=None):
        with tempfile.TemporaryDirectory() as root:
            args = dagger_args(root, "--takeover", "kinesthetic")
        clock = FakeClock()
        start = clock.t
        writer = Mock()
        writer.lock = contextlib.nullcontext()
        robot = SimpleNamespace(piper=api, hand=Mock(), hand_writer=writer, joint_control=SimpleNamespace(q=1),
                                commanded_pose=np.zeros(6), eef_pose=lambda: np.zeros(6))
        session = SimpleNamespace(camera=FakeCamera(), view=None, events=[], tables={}, intervals=[],
                                  episode=SimpleNamespace(open_interval=None))
        session.pedals = ScriptedPedals()
        session.pedals.poll = lambda: ["success"] if end_at_s is not None and clock.t - start >= end_at_s else []
        session.event = lambda name, detail="": session.events.append(name)
        session.add_table = lambda name, fields, rows: session.tables.__setitem__(name, rows)

        def interval_start():
            session.episode.open_interval = clock.monotonic_ns()
            session.intervals.append(("start", clock.t - start))

        def interval_end():
            if session.episode.open_interval is not None:
                session.episode.open_interval = None
                session.intervals.append(("end", clock.t - start))

        session.human_interval_start, session.human_interval_end = interval_start, interval_end
        takeover = dagger.KinestheticTakeover(args, robot, self.plan)
        with patch("inference.executor.dagger.time", clock), contextlib.redirect_stdout(io.StringIO()):
            try:
                return takeover.run(session), session, robot
            finally:
                self.handler_after = signal.getsignal(signal.SIGINT)

    def test_supported_handoff_then_end_pedal_freezes_and_returns_to_position_control(self):
        api = FakeArmApi()
        before = signal.getsignal(signal.SIGINT)
        outcome, session, robot = self.run_takeover(api, end_at_s=2.0)
        self.assertEqual(outcome, "success")
        self.assertIs(self.handler_after, before)
        robot.hand_writer.pause.assert_called_once()
        robot.hand.disable_torque.assert_called_once()
        self.assertEqual(api.calls[0], "set_mit_mode")
        self.assertEqual(api.command_times[0], api.mode_time)  # no application sleep before support
        self.assertEqual(api.calls[-4:], ["enable", "set_position_mode", "move_joint", "move_joint"])
        states = [row["state"] for row in session.tables["gravity_comp.csv"]]
        self.assertEqual([s for i, s in enumerate(states) if i == 0 or s != states[i - 1]],
                         ["HANDOFF", "ACTIVE", "BRAKE", "FREEZE_HOLD"])
        first_ff, full_ff = api.torques[0][1], api.torques[6 * 300][1]
        self.assertAlmostEqual(first_ff, 0.5)  # full gravity from the first MIT command
        self.assertAlmostEqual(full_ff, 0.5, places=6)
        self.assertEqual(api.torques[0][2]["kp"], self.plan.config.hold_kp)
        self.assertEqual(api.torques[0][2]["p_des"], 0.1)
        handoff_rows = [r for r in session.tables["gravity_comp.csv"] if r["state"] == "HANDOFF"]
        self.assertGreater(handoff_rows[0]["kp_1"], handoff_rows[-1]["kp_1"])
        (start_label, active_s), (end_label, frozen_s) = session.intervals
        self.assertEqual((start_label, end_label), ("start", "end"))
        self.assertAlmostEqual(active_s, 1.0, delta=0.02)  # human data starts after the ramp-in
        self.assertAlmostEqual(frozen_s, 2.0, delta=0.02)
        self.assertEqual(session.events[:4], ["hand_limp", "mit_switch", "supported_handoff", "active"])
        self.assertIsNone(robot.commanded_pose)

    def test_stream_watchdog_names_joint_and_individual_motor_stalls(self):
        for source in ("joint", "motor1", "motor5", "motor6"):
            with self.subTest(source=source):
                watch = dagger.FeedbackWatchdog()
                clocks = {"joint": 1.0, **{f"motor{i}": 1.0 for i in range(1, 7)}}
                self.assertTrue(watch.update(clocks, 0.0))
                clocks = {name: 1.0 if name == source else 2.0 for name in clocks}
                self.assertFalse(watch.update(clocks, 0.06))
                self.assertEqual(watch.stale_sources, [source])
                clocks[source] = 2.0
                self.assertTrue(watch.update(clocks, 0.061))

    def test_stuck_motor_clock_is_rejected_before_mode_switch(self):
        api = FakeArmApi()
        original = api.get_motor_state

        def stuck():
            state = original()
            state.timestamps[4] = 0.001
            return state

        api.get_motor_state = stuck
        with self.assertRaisesRegex(RuntimeError, "before MIT.*motor5"):
            self.run_takeover(api, end_at_s=2.0)
        self.assertNotIn("set_mit_mode", api.calls)
        self.assertEqual(api.torques, [])

    def test_single_motor_clock_freeze_brakes_with_full_gravity_and_names_source(self):
        api = FakeArmApi()
        original = api.get_motor_state

        def stuck_after_entry():
            state = original()
            state.timestamps[4] = min(api.stamp, 0.03)
            return state

        api.get_motor_state = stuck_after_entry
        with self.assertRaisesRegex(RuntimeError, "stale Piper feedback: stale sources: motor5"):
            self.run_takeover(api, end_at_s=2.0)
        self.assertEqual(api.calls[-4:], ["enable", "set_position_mode", "move_joint", "move_joint"])
        self.assertTrue(all(abs(torque - 0.5) < 1e-6 for _, torque, _ in api.torques))

    def test_stationary_joint_with_advancing_clocks_does_not_abort(self):
        api = FakeArmApi()
        original = api.get_joint_state

        def moving():
            state = original()
            state.joints[0] += 0.00001 * api.reads
            return state

        api.get_joint_state = moving
        outcome, session, _ = self.run_takeover(api, end_at_s=2.0)
        self.assertEqual(outcome, "success")
        self.assertTrue(all(not row["stale_sources"] for row in session.tables["gravity_comp.csv"]))

    def test_ctrl_c_freezes_holds_and_switches_to_position_mode_before_stopping(self):
        def interrupt(reads):
            if reads == 300:
                os.kill(os.getpid(), signal.SIGINT)

        api = FakeArmApi(on_read=interrupt)
        before = signal.getsignal(signal.SIGINT)
        with self.assertRaises(KeyboardInterrupt):
            self.run_takeover(api)
        self.assertIs(self.handler_after, before)
        self.assertEqual(api.calls[-4:], ["enable", "set_position_mode", "move_joint", "move_joint"])

    def test_real_calibration_builds_a_capped_config_and_warns_at_sagging_starts(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")
        with tempfile.TemporaryDirectory() as root:
            args = dagger_args(root, "--takeover", "kinesthetic", "--gravity-comp-repo", str(GRAVITY_COMP_REPO),
                               "--allow-underpowered", "--gravity-force")
            with contextlib.redirect_stdout(io.StringIO()):
                plan = dagger.prepare_kinesthetic(args)
                self.assertEqual(plan.config.ramp_in_s, 1.0)
                self.assertTrue(np.all(plan.config.assist_nm == 0))
                from data_collection.collection.config import PIPER_START_JOINTS_DEG as starts
                self.assertIn("J1", dagger.start_sag_warning(plan, starts["home_4c"]))
                self.assertIsNone(dagger.start_sag_warning(plan, starts["home_cube_newhw"]))
                with self.assertRaisesRegex(ValueError, "allow-underpowered"):
                    dagger.prepare_kinesthetic(dagger_args(root, "--takeover", "kinesthetic", "--gravity-force", "--no-allow-underpowered",
                                                           "--gravity-comp-repo", str(GRAVITY_COMP_REPO)))
                with self.assertRaisesRegex(ValueError, "piper-firmware"):
                    dagger.prepare_kinesthetic(dagger_args(root, "--takeover", "kinesthetic", "--gravity-force",
                                                           "--allow-underpowered", "--piper-firmware", "default",
                                                           "--gravity-comp-repo", str(GRAVITY_COMP_REPO)))


# Compact raw-to-training export coverage lives in test_dagger_export.py.


if __name__ == "__main__":
    unittest.main()
