"""Task-specific replay starts are validated and pinned before hardware opens."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from data_collection.collection import config as defaults
from data_collection.collection.task_start import write_task_start
from inference.executor import executor
from inference.executor import dagger
from scripts.kinesthetic import rollout


class ExecutorTaskStartTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.hand_config = self.root / "hand.yaml"
        original = executor.CONTROL_ROOT / defaults.KINESTHETIC_HAND_CONFIG
        self.hand_config.write_bytes(original.read_bytes())
        self.hand_hash = hashlib.sha256(self.hand_config.read_bytes()).hexdigest()
        self.start_log = self.root / "starts.jsonl"
        self.profile_path = self.root / "task.json"
        self.default_log_patch = patch.object(executor, "DEFAULT_START_LOG", self.start_log)
        self.default_log_patch.start()
        self.addCleanup(self.default_log_patch.stop)

    def profile(self, task="screw", second=False):
        ids = list(reversed(range(13)))
        hand = np.array([-.25, .11, -.32, .47, -.55, .66, .77,
                         -.88, .002, .99, -1.11, -.22, 1.33])
        if second:
            hand = hand * .5
        joints = ([-24.123456, 105.234567, -90.345678, 20.456789, -20.567891, -60.678912]
                  if second else [-35.123456, 90.234567, -60.345678, -15.456789, 30.567891, 45.678912])
        return dict(schema_version=1, task=task, captured_utc="2026-11-23T00:00:00+00:00",
                    arm=dict(joints_deg=joints),
                    hand=dict(motor_ids=ids, positions_rad=hand[ids].tolist(), config_sha256=self.hand_hash),
                    source=dict(episode_path="/tmp/raw/episode_000001", arm_monotonic_ns=1_000_000_000,
                                hand_monotonic_ns=1_001_000_000, at_s=0.0), warnings=[])

    def parse(self, profile=None, extra=()):
        argv = ["--predictor", "localhost", "--hand-config", str(self.hand_config), *extra]
        if profile is not None:
            write_task_start(self.profile_path, profile, overwrite=True)
            argv += ["--start-profile", str(self.profile_path)]
        return executor.parse_args(argv)

    def robot(self):
        robot = executor.RobotIO.__new__(executor.RobotIO)
        robot.hand = Mock()
        robot.hand.clip_positions.side_effect = lambda q: q.copy()
        robot.piper = Mock()
        robot.piper.get_eef_pose.return_value = [0.2, -.2, .3, 1, 2, 3]
        robot.joint_control = Mock()
        robot.commanded_pose = None
        return robot

    def move(self, robot, args):
        with patch("scripts.piper_api.replay_trajectory.read_joint_angles", return_value=np.zeros(6)), \
             patch("scripts.teleop.quest_teleop.go_home") as go_home, \
             patch.object(executor.time, "sleep"), redirect_stdout(io.StringIO()):
            robot.move_to_start(args, confirm=False)
        return go_home.call_args.args[1]

    def test_two_tasks_replay_their_exact_arm_and_hand_without_named_fallback(self):
        results = []
        # A task name is not a legacy hand-start key. Clearing that table proves
        # neither task quietly inherits a global median hand configuration.
        with patch.object(defaults, "HAND_START_POSITIONS_RAD", {}):
            for task, second in (("screw", False), ("cube", True)):
                with self.subTest(task=task):
                    profile = self.profile(task, second)
                    args = self.parse(profile)
                    robot = self.robot()
                    target = self.move(robot, args)
                    np.testing.assert_array_equal(target, np.radians(profile["arm"]["joints_deg"]))
                    expected_hand = executor.task_hand_positions(profile)
                    np.testing.assert_array_equal(robot.hand.set_positions_blocking.call_args.args[0], expected_hand)
                    self.assertFalse(robot.hand.set_positions_blocking.call_args.kwargs["clip"])
                    self.assertEqual(executor.start_base(args)[2], task)
                    results.append((target, expected_hand))
        self.assertFalse(np.array_equal(results[0][0], results[1][0]))
        self.assertFalse(np.array_equal(results[0][1], results[1][1]))

    def test_parse_pins_snapshot_and_hash_even_after_profile_file_changes(self):
        original = self.profile()
        args = self.parse(original)
        pinned_hash = args.task_start_sha256
        write_task_start(self.profile_path, self.profile("cube", True), overwrite=True)
        joints, hand, task = executor.start_base(args)
        self.assertEqual(task, "screw")
        self.assertEqual(hand, original)
        np.testing.assert_array_equal(joints, original["arm"]["joints_deg"])
        self.assertEqual(args.task_start_sha256, pinned_hash)
        self.assertNotEqual(pinned_hash, hashlib.sha256(self.profile_path.read_bytes()).hexdigest())
        self.profile_path.unlink()
        np.testing.assert_array_equal(executor.start_base(args)[0], original["arm"]["joints_deg"])

    def test_dagger_uses_pinned_task_start_for_kinesthetic_sag_check(self):
        original = self.profile()
        args = self.parse(original, extra=["--dagger", str(self.root / "dagger"),
                                           "--takeover", "kinesthetic"])
        self.profile_path.unlink()
        np.testing.assert_array_equal(dagger.start_joints_deg(args), original["arm"]["joints_deg"])

    def test_profile_snapshot_and_hash_describe_same_bytes_during_file_replacement(self):
        original = self.profile()
        write_task_start(self.profile_path, original)
        original_bytes = self.profile_path.read_bytes()
        replacement_bytes = json.dumps(self.profile("cube", True)).encode()
        real_loads = json.loads
        replaced = False

        def replace_after_decoding(payload, *args, **kwargs):
            nonlocal replaced
            value = real_loads(payload, *args, **kwargs)
            text = payload.decode() if isinstance(payload, bytes) else payload
            if not replaced and text == original_bytes.decode():
                self.profile_path.write_bytes(replacement_bytes)
                replaced = True
            return value

        with patch.object(executor.json, "loads", side_effect=replace_after_decoding):
            args = executor.parse_args(["--predictor", "localhost", "--hand-config", str(self.hand_config),
                                        "--start-profile", str(self.profile_path)])
        self.assertTrue(replaced)
        self.assertEqual(args.task_start_profile, original)
        self.assertEqual(args.task_start_sha256, hashlib.sha256(original_bytes).hexdigest())
        self.assertEqual(self.profile_path.read_bytes(), replacement_bytes)

    def test_profile_conflicts_are_rejected_for_every_other_start_selector(self):
        write_task_start(self.profile_path, self.profile())
        for selector in (("--start-home", "home_4a"), ("--start-home=home_4a",),
                         ("--start-id", "0"), ("--start-joints-deg", "0", "90", "-90", "0", "0", "0")):
            with self.subTest(selector=selector), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                executor.parse_args(["--predictor", "localhost", "--start-profile", str(self.profile_path), *selector])

    def test_invalid_start_is_rejected_before_any_connection_or_device_creation(self):
        modifications = {
            "hand limit": lambda p: p["hand"]["positions_rad"].__setitem__(p["hand"]["motor_ids"].index(8), -.2),
            "configuration hash": lambda p: p["hand"].update(config_sha256="b" * 64),
            "arm bounds": lambda p: p["arm"]["joints_deg"].__setitem__(5, 200),
            "hand motor identity": lambda p: p["hand"]["motor_ids"].__setitem__(0, 0),
        }
        for name, modify in modifications.items():
            profile = self.profile()
            modify(profile)
            # Deliberately write directly so malformed-schema input reaches
            # the executor rather than being rejected by the profile writer.
            self.profile_path.write_text(json.dumps(profile))
            argv = ["executor", "--predictor", "localhost", "--no-log", "--no-pedal",
                    "--hand-config", str(self.hand_config), "--start-profile", str(self.profile_path)]
            with self.subTest(case=name), patch.object(executor.sys, "argv", argv), \
                 patch.object(executor.socket, "create_connection") as connect, \
                 patch.object(executor, "RobotIO") as robot, patch.object(executor, "OakCamera") as camera, \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                executor.main()
            connect.assert_not_called()
            robot.assert_not_called()
            camera.assert_not_called()

    def test_hand_start_sends_unclipped_profile_values_with_motor_id_mapping(self):
        robot = self.robot()
        profile = self.profile()
        expected = executor.task_hand_positions(profile)
        with patch.object(defaults, "HAND_START_POSITIONS_RAD", {}), redirect_stdout(io.StringIO()):
            robot.move_hand_to_start(profile, enable_torque=True)
        np.testing.assert_array_equal(robot.hand.set_positions.call_args.args[0], expected)
        self.assertFalse(robot.hand.set_positions.call_args.kwargs["clip"])
        robot.hand.enable_torque.assert_called_once()
        np.testing.assert_array_equal(robot.hand.set_positions_blocking.call_args.args[0], expected)
        self.assertFalse(robot.hand.set_positions_blocking.call_args.kwargs["clip"])

    def test_connected_hand_limit_difference_refuses_motion_instead_of_clipping(self):
        robot = self.robot()
        robot.hand.clip_positions.side_effect = lambda q: q + .1
        with self.assertRaisesRegex(ValueError, "refusing to clip"):
            robot.move_hand_to_start(self.profile(), enable_torque=True)
        robot.hand.set_positions.assert_not_called()
        robot.hand.set_positions_blocking.assert_not_called()
        robot.hand.enable_torque.assert_not_called()

    def test_logged_task_start_keeps_profile_and_full_precision_for_start_id(self):
        original = self.profile()
        args = self.parse(original)
        first = self.robot()
        target = self.move(first, args)
        record = json.loads(self.start_log.read_text().splitlines()[0])
        self.assertEqual(record["task_start_profile"], original)
        self.assertEqual(record["task_start_sha256"], args.task_start_sha256)
        repeated = self.parse(extra=["--start-id", "0"])
        self.profile_path.unlink()
        robot = self.robot()
        repeated_target = self.move(robot, repeated)
        np.testing.assert_array_equal(repeated_target, target)
        np.testing.assert_array_equal(robot.hand.set_positions_blocking.call_args.args[0], executor.task_hand_positions(original))
        repeated_record = json.loads(self.start_log.read_text().splitlines()[-1])
        self.assertEqual(repeated_record["task_start_profile"], original)

    def test_parsed_start_id_is_pinned_if_log_is_changed_after_parse(self):
        args = self.parse(self.profile())
        self.move(self.robot(), args)
        repeated = self.parse(extra=["--start-id", "0"])
        before = executor.start_base(repeated)
        record = json.loads(self.start_log.read_text().splitlines()[0])
        record["joints_deg"][0] = -99
        self.start_log.write_text(json.dumps(record) + "\n")
        after = executor.start_base(repeated)
        np.testing.assert_array_equal(after[0], before[0])
        np.testing.assert_array_equal(dagger.start_joints_deg(repeated), before[0])
        self.assertEqual(after[1], before[1])

    def test_embedded_logged_profile_is_schema_validated(self):
        args = self.parse(self.profile())
        self.move(self.robot(), args)
        record = json.loads(self.start_log.read_text().splitlines()[0])
        record["task_start_profile"]["hand"]["motor_ids"][0] = 0
        self.start_log.write_text(json.dumps(record) + "\n")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse(extra=["--start-id", "0"])

    def test_rollout_requires_explicit_task_start_and_uses_collection_hand_config(self):
        with self.assertRaisesRegex(ValueError, "no fixed task start"):
            rollout.main(["--predictor", "localhost", "--fake"])
        write_task_start(self.profile_path, self.profile())
        with patch.object(rollout, "running_drag", return_value=False), patch.object(rollout.os, "execv") as execute:
            rollout.main(["--predictor", "localhost", "--start-profile", str(self.profile_path)])
        launched = execute.call_args.args[1]
        self.assertEqual(launched[launched.index("--hand-config") + 1],
                         str(rollout.ROOT / defaults.KINESTHETIC_HAND_CONFIG))
        self.assertEqual(launched[launched.index("--start-profile") + 1], str(self.profile_path))
        self.assertNotIn("--start-home", launched)

    def test_rollout_preserves_explicit_hand_configuration(self):
        write_task_start(self.profile_path, self.profile())
        with patch.object(rollout, "running_drag", return_value=False), patch.object(rollout.os, "execv") as execute:
            rollout.main(["--predictor", "localhost", "--start-profile", str(self.profile_path),
                          "--hand-config", str(self.hand_config)])
        launched = execute.call_args.args[1]
        self.assertEqual(launched.count("--hand-config"), 1)
        self.assertEqual(launched[launched.index("--hand-config") + 1], str(self.hand_config))


if __name__ == "__main__":
    unittest.main()
