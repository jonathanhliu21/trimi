import contextlib
import hashlib
import io
import json
import subprocess
import shutil
import sys
import tempfile
import time
import unittest
from uuid import uuid4
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from data_collection import postprocess_dataset as post
from data_collection.collect_hand_only import parse_args
from data_collection.collection.config import PIPER_START_JOINTS_DEG, DEFAULT_EXECUTOR_START
from data_collection.collection.shared_state import ArmHealth
from data_collection.collection.tracking_monitor import start_guidance
from data_collection.replay_actions import piper_model
from inference.executor.joint_control import JointController
from scripts.dataset import check_episodes as quality
from scripts.kinesthetic import launch
from data_collection.collection.piper_worker import arm_health_problem
import test_postprocess_dataset as fixtures
import test_check_episodes as quality_fixtures
import test_listener_bench_report as bench


def task_start_fixture(task="screw"):
    return dict(schema_version=1, task=task, captured_utc="2026-11-23T00:00:00+00:00",
                arm=dict(joints_deg=list(PIPER_START_JOINTS_DEG["kinesthetic"])),
                hand=dict(motor_ids=list(range(13)), positions_rad=[0.] * 13,
                          config_sha256="a" * 64),
                source=dict(episode_path="/tmp/raw/episode_000000", arm_monotonic_ns=10**9,
                            hand_monotonic_ns=10**9 + 10**6, at_s=0.), warnings=[])


class FKTests(unittest.TestCase):
    def fixture(self, root, *, firmware=True):
        helper = fixtures.PostprocessTests()
        config, episode, *_ = helper.fixture(root)
        config = replace(config, calibration_json=None, pose_source="piper_fk")
        fk, _ = piper_model()
        q0 = np.radians(PIPER_START_JOINTS_DEG["kinesthetic"])
        joints = np.array([q0 + np.radians([i * .2, 0, 0, 0, 0, 0]) for i in range(4)])
        rows = []
        for i, q in enumerate(joints):
            receive = 1_000_000_000 + i * 20_000_000
            row = dict(sample_idx=i, monotonic_ns=receive + 35_000_000,
                       joint_rx_monotonic_ns=receive, robot_timestamp_s=100 + i*.02,
                       eef_timestamp_s=100 + i*.02,
                       **{f"joint_{j+1}": x for j, x in enumerate(q)})
            if firmware:
                pose = fk(q)
                pose[0] += .02
                pose[3:] = (Rotation.from_rotvec([0, 0, np.pi / 2])
                            * Rotation.from_euler("xyz", pose[3:])).as_euler("xyz")
                row.update(dict(zip(post.base.RAW_EEF_FIELDS, pose)))
            rows.append(row)
        helper.write_csv(episode.piper_path, list(rows[0]), rows)
        return config, episode, joints, rows

    def test_fk_roundtrip_uses_receive_clock_and_preserves_raw(self):
        with tempfile.TemporaryDirectory() as folder:
            config, episode, joints, rows = self.fixture(Path(folder))
            raw = episode.piper_path.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                result = post.process_episode(config, episode)
            self.assertEqual(episode.piper_path.read_bytes(), raw)
            with np.load(result["sync"]) as sync:
                np.testing.assert_array_equal(sync["pose_idx"], [0, 1, 2, 3])
                np.testing.assert_array_equal(sync["pose_monotonic_ns"], [r["joint_rx_monotonic_ns"] for r in rows])
            output = post.base.read_csv_rows(Path(result["piper_csv"]))
            self.assertEqual(int(output[0]["recorded_monotonic_ns"]), rows[0]["monotonic_ns"])
            self.assertEqual(int(output[0]["monotonic_ns"]), rows[0]["joint_rx_monotonic_ns"])
            np.testing.assert_allclose([[float(r[f"joint_{j}"]) for j in range(1, 7)] for r in output], joints)
            comparison = json.loads(Path(result["metadata"]).read_text())["fk_firmware_comparison"]
            self.assertAlmostEqual(comparison["position_mm"]["median"], 20)
            self.assertAlmostEqual(comparison["rotation_deg"]["median"], 90)
            controller = JointController()
            controller.reset(joints[0])
            pose = np.load(result["eef"])[0]
            target, diag = controller.command(np.r_[pose[:3], Rotation.from_rotvec(pose[3:]).as_euler("xyz")], 1/15)
            np.testing.assert_allclose(target, joints[0], atol=2e-5)
            self.assertLess(diag["pos_err_mm"], .01)

    def test_fk_without_firmware_and_stale_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            config, episode, _, rows = self.fixture(Path(folder), firmware=False)
            converted, poses, stamps, comparison = post.convert_piper_fk(episode)
            self.assertEqual(comparison["compared_rows"], 0)
            self.assertEqual(poses.shape, (4, 6))
            self.assertEqual(converted[0]["firmware_x"], "")
            self.assertTrue(np.isfinite(poses).all())

    def test_invalid_receive_clock_or_joints_rejected(self):
        for change in ({"joint_rx_monotonic_ns": ""}, {"joint_rx_monotonic_ns": 1_000_000_000},
                       {"joint_3": "nan"}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                _, episode, _, rows = self.fixture(Path(folder))
                rows[1].update(change)
                fixtures.PostprocessTests().write_csv(episode.piper_path, list(rows[0]), rows)
                with self.assertRaises(ValueError):
                    post.convert_piper_fk(episode)

    def test_fk_rejects_anchor_options(self):
        with tempfile.TemporaryDirectory() as folder:
            config, *_ = self.fixture(Path(folder))
            for config in (replace(config, absolute=True), replace(config, piper_start_pose=(0,)*6),
                           replace(config, calibration_json=Path("calibration.json"))):
                with self.assertRaises(ValueError):
                    post.resolve_pose_source(config)


class StartGuideTests(unittest.TestCase):
    def test_passive_j6_health_requires_explicit_selection(self):
        state = [True] * 5 + [False]
        self.assertIn("DRIVES OFF", arm_health_problem(state, 0))
        self.assertEqual(arm_health_problem(state, 0, disable_j6=True), "")
        self.assertIn("UNEXPECTEDLY ENABLED", arm_health_problem([True]*6, 0, disable_j6=True))
        self.assertIn("NO DRIVER STATE", arm_health_problem([True]*5+[None], 0, disable_j6=True))
        self.assertIn("INCOMPLETE", arm_health_problem([], 0, disable_j6=True))
        with patch("sys.argv", ["collect", "--grav-comp-disable-j6"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_guide_freshness_health_and_end_effector_error(self):
        q = tuple(np.radians(PIPER_START_JOINTS_DEG["kinesthetic"]))
        health = ArmHealth(10**9, True, "", 200, q)
        status, error = start_guidance(10**9, health, "kinesthetic")
        self.assertTrue(status.ok)
        np.testing.assert_allclose(error, 0, atol=1e-12)
        health = replace(health, joints=(*q[:5], q[5]+np.radians(40)))
        status, error = start_guidance(10**9, health, "kinesthetic")
        self.assertTrue(status.ok)  # Forty degrees of wrist roll is within the pose criterion.
        self.assertAlmostEqual(error[0], 0.)
        self.assertAlmostEqual(error[1], 40.)
        health = replace(health, joints=(*q[:5], q[5]+np.radians(60)))
        self.assertFalse(start_guidance(10**9, health, "kinesthetic")[0].ok)
        self.assertFalse(start_guidance(2*10**9, health, "kinesthetic")[0].ok)
        self.assertFalse(start_guidance(10**9, replace(health, ok=False, problem="DRIVES OFF"), "kinesthetic")[0].ok)
        self.assertEqual(DEFAULT_EXECUTOR_START, "home_4c")

    def test_flags_and_measured_first_joints(self):
        with patch("sys.argv", ["collect", "--arm-mode", "gravity-comp"]):
            args = parse_args()
            self.assertIsNone(args.kinesthetic_start)
            self.assertTrue(args.no_start_guide)
        with patch("sys.argv", ["collect", "--arm-mode", "gravity-comp",
                                "--kinesthetic-start", "kinesthetic"]):
            self.assertEqual(parse_args().kinesthetic_start, "kinesthetic")
        with patch("sys.argv", ["collect", "--kinesthetic-start", "kinesthetic"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args()
        with tempfile.TemporaryDirectory() as folder:
            episode = bench.record_bench_episode(Path(folder), [time.time()+i*.005 for i in range(3)])
            meta = json.loads((episode/"metadata.json").read_text())
            self.assertFalse(meta["piper"]["gravity_comp"]["passive_j6"])
            self.assertEqual(meta["piper"]["gravity_comp"]["zero_torque_joints"], [])
            first = quality.read_rows(episode/"piper.csv")[0]
            np.testing.assert_allclose(meta["piper"]["measured_start_joints_rad"],
                                       [float(first[f"joint_{j}"]) for j in range(1, 7)])

    def test_task_profile_snapshot_reaches_config_and_episode_metadata(self):
        from data_collection.collect_hand_only import main, start_episode, stop_episode
        from data_collection.collection.episode import RecordingManager
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, calib = root / "screw.json", root / "calib.json"
            profile = task_start_fixture("screw with arbitrary task label")
            raw = json.dumps(profile).encode()
            path.write_bytes(raw)
            calib.write_text("{}")
            with patch("sys.argv", ["collect", "--arm-mode", "gravity-comp", "--start-profile",
                                    str(path), "--grav-comp-calib", str(calib),
                                    "--dataset-root", str(root), "--no-camera"]), \
                    patch("data_collection.collect_hand_only.run_hand_only", return_value=0) as run:
                self.assertEqual(main(), 0)
            config = run.call_args.args[0]
            self.assertEqual(config.start_profile_snapshot, profile)
            self.assertEqual(config.start_profile_sha256, hashlib.sha256(raw).hexdigest())
            self.assertIsNone(config.kinesthetic_start)
            self.assertFalse(config.no_start_guide)
            path.write_text("{}")  # A later edit cannot change the selected target.
            recorder = RecordingManager()
            with contextlib.redirect_stdout(io.StringIO()):
                session = start_episode(config, recorder)
                stop_episode(recorder, "foot_pedal")
            meta = json.loads(session.paths.metadata_path.read_text())
            guide = meta["piper"]["gravity_comp"]
            self.assertEqual(guide["task"], profile["task"])
            self.assertEqual(guide["guided_start_joints_deg"], profile["arm"]["joints_deg"])
            self.assertEqual(guide["start_profile"], profile)
            self.assertEqual(guide["start_profile_sha256"], hashlib.sha256(raw).hexdigest())

    def test_definition_episode_records_its_role_without_a_target(self):
        from data_collection.collect_hand_only import main, start_episode, stop_episode
        from data_collection.collection.episode import RecordingManager
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            calib = root / "calib.json"
            calib.write_text("{}")
            with patch("sys.argv", ["collect", "--arm-mode", "gravity-comp", "--define-start",
                                    "--grav-comp-calib", str(calib), "--dataset-root", str(root)]), \
                    patch("data_collection.collect_hand_only.run_hand_only", return_value=0) as run:
                self.assertEqual(main(), 0)
            config = run.call_args.args[0]
            self.assertTrue(config.define_start)
            self.assertTrue(config.no_start_guide)
            recorder = RecordingManager()
            with contextlib.redirect_stdout(io.StringIO()):
                session = start_episode(config, recorder)
                stop_episode(recorder, "foot_pedal")
            meta = json.loads(session.paths.metadata_path.read_text())
            self.assertEqual(meta["episode_role"], "start_definition")
            self.assertIn("start_definition", meta["flags"])
            self.assertIsNone(meta["piper"]["gravity_comp"]["guided_start_joints_deg"])
            self.assertFalse(meta["piper"]["gravity_comp"]["start_guide_enabled"])

    def test_profile_monitor_receives_explicit_joints_and_calibrated_gravity(self):
        from data_collection.collect_hand_only import run_hand_only
        from data_collection.collection.config import HandOnlyCollectionConfig, RETARGET_ONLY_MODE
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            calib = root / "calib.json"
            gravity = [0., -9.8, .2]
            calib.write_text(json.dumps(dict(gravity=dict(vector_m_s2=gravity))))
            profile = task_start_fixture("arbitrary task")
            config = HandOnlyCollectionConfig(dataset_root=root,
                hand_config=Path("config/xl330_new_hand_transparent.yaml"), hand_port="/dev/fake",
                hand_measure_rate_hz=50., warmup_time=0., episode_time=1., mode=RETARGET_ONLY_MODE,
                use_piper=True, arm_mode="gravity-comp", grav_comp_calib=calib,
                start_profile_snapshot=profile, show_tracking_monitor=True)
            with patch("data_collection.collect_hand_only.initialize_worker", return_value=[]), \
                    patch("data_collection.collection.tracking_monitor.TrackingMonitor") as monitor, \
                    patch("data_collection.collect_hand_only.run_camera_view", return_value="closed"), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_hand_only(config), 0)
            self.assertEqual(monitor.call_args.kwargs["start_name"], profile["task"])
            self.assertEqual(monitor.call_args.kwargs["start_joints_deg"], profile["arm"]["joints_deg"])
            self.assertEqual(monitor.call_args.kwargs["gravity_vector"], gravity)

    def test_start_selection_requires_gravity_mode_and_exclusive_choice(self):
        for flags in (["--start-profile", "task.json"], ["--define-start"], ["--no-start-guide"],
                      ["--arm-mode", "gravity-comp", "--start-profile", "task.json", "--kinesthetic-start", "kinesthetic"],
                      ["--arm-mode", "gravity-comp", "--define-start", "--start-profile", "task.json"]):
            with self.subTest(flags=flags), patch("sys.argv", ["collect", *flags]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args()


class PiperQualityTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)
        quality_fixtures.make_episode(root, "episode_000000", quest_moves=True)
        ep = root/"raw/episode_000000"
        meta = json.loads((ep/"metadata.json").read_text())
        meta["piper"] = {"mode": "gravity_comp_kinesthetic", "gravity_comp": dict(
            guided_start="quality task", guided_start_joints_deg=list(PIPER_START_JOINTS_DEG["kinesthetic"]),
            start_criterion="end_effector_pose", start_position_tolerance_m=.10,
            start_orientation_tolerance_deg=45.)}
        meta["trimorph"]["hand"] = dict(motor_ids=list(range(13)),
                                       joint_lower_limits={str(i):-1000 for i in range(13)},
                                       joint_upper_limits={str(i):1000 for i in range(13)})
        q = np.radians(PIPER_START_JOINTS_DEG["kinesthetic"])
        rows = [dict(monotonic_ns=10**10+i*5_000_000+100_000_000, joint_rx_monotonic_ns=10**10+i*5_000_000,
                     arm_ok="1", err_code="0", ctrl_mode="CAN_CTRL", arm_status="NORMAL", mode_feedback="MOVE_MIT",
                     torque_1=2, torque_2=-2, **{f"enable_{j}":1 for j in range(1,7)},
                     **{f"joint_{j+1}": v for j,v in enumerate(q)}) for i in range(400)]
        return ep, meta, rows

    def check(self, ep, meta, rows):
        (ep/"metadata.json").write_text(json.dumps(meta))
        quality_fixtures.write_csv(ep/"piper.csv", rows)
        with contextlib.redirect_stdout(io.StringIO()):
            return quality.check_episode(ep, 10)

    def test_good_kinesthetic_episode_passes_and_receive_clock_sync(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(self.check(*self.fixture(folder)), [])

    def test_quality_prefers_recorded_task_target_to_legacy_named_lookup(self):
        with tempfile.TemporaryDirectory() as folder:
            ep, meta, rows = self.fixture(folder)
            target = list(PIPER_START_JOINTS_DEG["kinesthetic"])
            meta["piper"]["gravity_comp"] = dict(guided_start="arbitrary task",
                guided_start_joints_deg=target, start_tolerance_deg=5.)
            self.assertEqual(self.check(ep, meta, rows), [])
            meta["piper"]["gravity_comp"]["guided_start_joints_deg"][4] += 20.
            self.assertIn("arbitrary task", "; ".join(self.check(ep, meta, rows)))

    @staticmethod
    def offset_arm(rows, delta_deg):
        q = np.radians(np.asarray(PIPER_START_JOINTS_DEG["kinesthetic"]) + delta_deg)
        for row in rows:
            row.update({f"joint_{j+1}": value for j, value in enumerate(q)})
        return q

    def test_large_joint_difference_can_pass_end_effector_start_criterion(self):
        with tempfile.TemporaryDirectory() as folder:
            ep, meta, rows = self.fixture(folder)
            self.offset_arm(rows, [0, 0, 0, 0, 0, 40])
            self.assertEqual(self.check(ep, meta, rows), [])
            # Historical recordings retain their explicitly recorded joint criterion.
            guide = meta["piper"]["gravity_comp"]
            guide.pop("start_criterion")
            guide["start_tolerance_deg"] = 5.
            self.assertIn("40.00 deg", "; ".join(self.check(ep, meta, rows)))

    def test_pose_position_and_orientation_fail_independently(self):
        for delta, expected, absent in (
            ([30, 0, 0, 0, 0, 0], "start position", "total orientation"),
            ([0, 0, 0, 0, 0, 60], "total orientation", "start position"),
        ):
            with self.subTest(delta=delta), tempfile.TemporaryDirectory() as folder:
                ep, meta, rows = self.fixture(folder)
                self.offset_arm(rows, delta)
                result = "; ".join(self.check(ep, meta, rows))
                self.assertIn(expected, result)
                self.assertNotIn(absent, result)

    def test_orientation_uses_total_so3_angle_instead_of_euler_components(self):
        from data_collection.collection.arm_geometry import piper_frames
        with tempfile.TemporaryDirectory() as folder:
            ep, meta, rows = self.fixture(folder)
            q = self.offset_arm(rows, [0, 0, 0, 35, -35, 0])
            target = np.radians(meta["piper"]["gravity_comp"]["guided_start_joints_deg"])
            current_frame, target_frame = piper_frames(q)[-1], piper_frames(target)[-1]
            relative = Rotation.from_matrix(target_frame[:3, :3].T @ current_frame[:3, :3])
            self.assertTrue(np.all(np.abs(relative.as_euler("xyz", degrees=True)) < 45.))
            self.assertGreater(np.degrees(relative.magnitude()), 45.)
            result = "; ".join(self.check(ep, meta, rows))
            self.assertIn("total orientation", result)
            self.assertNotIn("start position", result)

    def test_start_check_failure_does_not_suppress_physical_joint_limit_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            ep, meta, rows = self.fixture(folder)
            meta["piper"]["gravity_comp"].pop("guided_start_joints_deg")
            self.offset_arm(rows, [0, 0, 0, 0, 0, 360])
            result = "; ".join(self.check(ep, meta, rows))
            self.assertIn("start check unavailable", result)
            self.assertIn("Piper J6:", result)

    def test_end_effector_start_check_preserves_raw_files(self):
        with tempfile.TemporaryDirectory() as folder:
            ep, meta, rows = self.fixture(folder)
            self.offset_arm(rows, [0, 0, 0, 0, 0, 40])
            self.check(ep, meta, rows)
            paths = [ep / "metadata.json", ep / "piper.csv"]
            before = [path.read_bytes() for path in paths]
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(quality.check_episode(ep, 10), [])
            self.assertIn("flange position error", output.getvalue())
            self.assertIn("total orientation error", output.getvalue())
            self.assertEqual([path.read_bytes() for path in paths], before)

    def test_definition_is_never_reported_as_training_clean(self):
        with tempfile.TemporaryDirectory() as folder:
            ep, meta, rows = self.fixture(folder)
            meta["episode_role"] = "start_definition"
            self.assertIn("not a training episode", "; ".join(self.check(ep, meta, rows)))

    def test_explicit_no_guide_collection_is_not_reported_as_start_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            ep, meta, rows = self.fixture(folder)
            meta["piper"]["gravity_comp"] = dict(start_guide_enabled=False)
            self.assertIn("no recorded task-start target", "; ".join(self.check(ep, meta, rows)))

    def test_passive_j6_quality_uses_recorded_enable_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            ep, meta, rows = self.fixture(folder)
            meta["piper"]["gravity_comp"] = dict(disabled_joints=[6])
            for row in rows:
                row["enable_6"] = 0
            self.assertEqual(self.check(ep, meta, rows), [])
            rows[10]["enable_6"] = 1
            self.assertIn("unhealthy", "; ".join(self.check(ep, meta, rows)))

    def test_zero_gain_passive_j6_still_requires_all_six_drives_enabled(self):
        with tempfile.TemporaryDirectory() as folder:
            ep, meta, rows = self.fixture(folder)
            meta["piper"]["gravity_comp"] = dict(passive_j6=True, disabled_joints=[], zero_torque_joints=[6])
            self.assertEqual(self.check(ep, meta, rows), [])
            rows[10]["enable_6"] = 0
            self.assertIn("unhealthy", "; ".join(self.check(ep, meta, rows)))

    def test_missing_motor_ids_and_invalid_limits_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            ep, meta, rows = self.fixture(folder)
            meta["trimorph"]["hand"]["motor_ids"] = []
            self.assertIn("expected recorded motor IDs", "; ".join(self.check(ep, meta, rows)))
            meta["trimorph"]["hand"]["motor_ids"] = list(range(13))
            for upper in (float("nan"), -1001):
                meta["trimorph"]["hand"]["joint_upper_limits"]["8"] = upper
                self.assertIn("hand motor 8: missing/invalid", "; ".join(self.check(ep, meta, rows)))

    def test_start_mismatch_and_camera_hand_sync_are_flagged(self):
        with tempfile.TemporaryDirectory() as folder:
            ep, meta, rows = self.fixture(folder)
            meta["piper"]["gravity_comp"] = dict(guided_start="kinesthetic", start_tolerance_deg=5)
            rows[0]["joint_1"] += np.radians(8)
            quality_fixtures.write_csv(ep/"camera_timestamps.csv", [
                dict(monotonic_ns=10**10+2_100_000_000+i*33_333_333) for i in range(3)])
            # Video decoding is orthogonal to the real receive-clock distance calculation.
            with patch("cv2.VideoCapture") as capture:
                capture.return_value.get.return_value = 3
                result = "; ".join(self.check(ep, meta, rows))
            self.assertIn("start differs from kinesthetic", result)
            self.assertIn("camera->hand sync distances exceed", result)

    def test_mode_limits_torque_and_hand_are_flagged(self):
        with tempfile.TemporaryDirectory() as folder:
            ep, meta, rows = self.fixture(folder)
            rows[40].update(mode_feedback="MOVE_J", joint_6=9, torque_1=7.2)
            meta["trimorph"]["hand"]["joint_upper_limits"]["8"] = .01
            result = "; ".join(self.check(ep, meta, rows))
            for message in ("not in MIT", "status changed", "J6:", "near 7.2", "hand motor 8"):
                self.assertIn(message, result)

    def test_bad_receive_clock_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            ep, meta, rows = self.fixture(folder)
            rows[10]["joint_rx_monotonic_ns"] = rows[9]["joint_rx_monotonic_ns"]
            self.assertIn("strictly increasing", "; ".join(self.check(ep, meta, rows)))

    def test_torque_duration_excludes_feedback_gap(self):
        with tempfile.TemporaryDirectory() as folder:
            _, meta, rows = self.fixture(folder)
            rows = [rows[0], rows[200]]
            rows[0]["torque_1"] = 7.2
            warnings = []
            with contextlib.redirect_stdout(io.StringIO()):
                quality.piper_checks(rows, meta, warnings)
            self.assertFalse(any("torque near" in warning for warning in warnings))
            self.assertTrue(any("maximum gap" in warning for warning in warnings))


class LauncherTests(unittest.TestCase):
    def mocked_launch(self, root, flags, *, mutate_source=None):
        """Exercise session files and preflight wiring without processes or devices."""
        calib, port, hand = root / "calib.json", root / "hand-port", root / "hand.yaml"
        calib.write_text("{}")
        port.touch()
        hand.write_text("{}")
        checked = []

        def run(command, **kwargs):
            if any(str(arg).endswith("check_start_pose.py") for arg in command):
                checked.append(command)
                profile_arg = command[command.index("--start-profile") + 1]
                self.assertEqual(json.loads(Path(profile_arg).read_text())["task"], "screw")
                if mutate_source is not None:
                    mutate_source.write_text("{}")
                return subprocess.CompletedProcess(command, 0, stdout=json.dumps(dict(ok=True, command_nm=[0.] * 6)))
            self.assertIn("pytest", command)
            return subprocess.CompletedProcess(command, 0, stdout="210 passed\n")

        with patch.object(launch, "running_drag", return_value=[]), \
                patch.object(launch.shutil, "which", return_value="/fake/bin"), \
                patch.object(launch.subprocess, "check_output", side_effect=[
                    '[{"flags":["UP"]}]', "--j5-kd --passive-j6 --passive-j5-j6 --disable-j6"]), \
                patch.object(launch.subprocess, "run", side_effect=run), \
                patch.object(launch, "tmux", return_value="%fake"), \
                patch("data_collection.collection.hand_common.load_hand_config"), \
                patch("builtins.input", return_value="mounted"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launch.main(["--midas-repo", str(root), "--dataset-root", str(root / "dataset"),
                                          "--calib", str(calib), "--hand-config", str(hand), "--hand-port", str(port),
                                          "--channel", "unit-" + uuid4().hex[:8], "--detach", *flags]), 0)
        manifest_path, = (root / "dataset" / "sessions").glob("*/session.json")
        return manifest_path, json.loads(manifest_path.read_text()), checked

    def test_profile_bytes_and_hash_are_preserved_through_preflight_and_session(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "task.json"
            original = json.dumps(task_start_fixture(), indent=2).encode()
            path.write_bytes(original)
            manifest_path, manifest, checked = self.mocked_launch(
                root, ["--start-profile", str(path)], mutate_source=path)
            self.assertEqual(len(checked), 1)
            saved = manifest_path.parent / "task_start.json"
            self.assertEqual(saved.read_bytes(), original)
            self.assertEqual(manifest["start_profile_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(manifest["start_profile"], json.loads(original))
            self.assertEqual(manifest["start_check"]["start_profile_path"], str(saved))
            command = manifest["collector_command"]
            self.assertEqual(command[command.index("--start-profile") + 1], str(saved))

    def test_definition_session_does_not_run_or_claim_a_start_torque_check(self):
        with tempfile.TemporaryDirectory() as folder:
            _, manifest, checked = self.mocked_launch(Path(folder), [])
            self.assertEqual(checked, [])
            self.assertIsNone(manifest["start_check"]["ok"])
            self.assertEqual(manifest["start_check"]["status"], "not_checked")
            self.assertEqual(manifest["episode_role"], "start_definition")
            self.assertIn("--define-start", manifest["collector_command"])
            self.assertNotIn("--kinesthetic-start", manifest["collector_command"])

    def test_default_launcher_defines_start_without_a_fixed_named_target(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(launch.main(["--dry-run", "--midas-repo", folder]), 0)
        self.assertIn("--define-start", output.getvalue())
        self.assertNotIn("--kinesthetic-start", output.getvalue())

    def test_zero_j5_damping_is_forwarded_and_saved_in_the_session_command(self):
        with tempfile.TemporaryDirectory() as folder:
            _, manifest, _ = self.mocked_launch(Path(folder), ["--j5-kd", "0", "--passive-j6"])
            drag = manifest["drag_command"]
            self.assertEqual(float(drag[drag.index("--j5-kd") + 1]), 0.0)
            self.assertIn("--passive-j6", drag)
            self.assertIn("--grav-comp-passive-j6", manifest["collector_command"])

    def test_invalid_j5_damping_is_rejected_before_preflight(self):
        for value in ("-1", "5.1", "nan", "inf", "-inf"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), \
                    patch.object(launch.subprocess, "check_output") as preflight:
                with self.assertRaises(SystemExit) as exc:
                    launch.main(["--dry-run", f"--j5-kd={value}"])
                self.assertEqual(exc.exception.code, 2)
                preflight.assert_not_called()

    def test_passive_j5_j6_is_saved_in_both_session_commands(self):
        with tempfile.TemporaryDirectory() as folder:
            _, manifest, _ = self.mocked_launch(Path(folder), ["--passive-j5-j6"])
            self.assertIn("--passive-j5-j6", manifest["drag_command"])
            self.assertIn("--grav-comp-passive-j5-j6", manifest["collector_command"])
            self.assertNotIn("--disable-j6", manifest["drag_command"])
            self.assertNotIn("--grav-comp-disable-j6", manifest["collector_command"])

    def test_profile_launcher_uses_a_session_snapshot_and_legacy_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "task.json"
            path.write_text(json.dumps(task_start_fixture()))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(launch.main(["--dry-run", "--midas-repo", folder,
                                              "--start-profile", str(path)]), 0)
            self.assertIn("--start-profile", output.getvalue())
            self.assertIn("/task_start.json", output.getvalue())
            self.assertNotIn("--kinesthetic-start", output.getvalue())
            self.assertNotIn("--define-start", output.getvalue())
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(launch.main(["--dry-run", "--midas-repo", folder,
                                              "--start-home", "kinesthetic"]), 0)
            self.assertIn("--kinesthetic-start kinesthetic", output.getvalue())
    def test_passive_flag_selects_matching_controller_and_collector_profiles(self):
        for flag, controller_flag, collector_flag in (
            ("--passive-j6", "--passive-j6", "--grav-comp-passive-j6"),
            ("--disable-j6", "--disable-j6", "--grav-comp-disable-j6"),
            ("--passive-j5-j6", "--passive-j5-j6", "--grav-comp-passive-j5-j6"),
        ):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as folder:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(launch.main(["--dry-run", "--midas-repo", folder, flag]), 0)
                drag, collector = [launch.shlex.split(line.split(": ", 1)[1])
                                   for line in output.getvalue().splitlines()]
                self.assertIn(controller_flag, drag)
                self.assertEqual(drag[drag.index("--j5-kd") + 1], "0.005")
                self.assertIn(collector_flag, collector)
                selected = {"--grav-comp-passive-j6", "--grav-comp-disable-j6",
                            "--grav-comp-passive-j5-j6"}
                self.assertEqual(selected.intersection(collector), {collector_flag})

    def test_passive_and_disabled_launcher_profiles_conflict_before_preflight(self):
        for first, second in (("--passive-j6", "--disable-j6"),
                              ("--passive-j5-j6", "--disable-j6"),
                              ("--passive-j5-j6", "--passive-j6")):
            with self.subTest(flags=(first, second)), contextlib.redirect_stderr(io.StringIO()), \
                    patch.object(launch.subprocess, "check_output") as preflight:
                with self.assertRaises(SystemExit) as exc:
                    launch.main([first, second, "--dry-run"])
                self.assertEqual(exc.exception.code, 2)
                preflight.assert_not_called()

    def test_readiness_ignores_partial_line_and_invalidates_freeze(self):
        self.assertFalse(launch.drag_ready("assist at target - drag aw"))
        self.assertFalse(launch.drag_ready(launch.READY))
        self.assertTrue(launch.drag_ready("\x1b[32m" + launch.READY + "\r\n"))
        self.assertFalse(launch.drag_ready(launch.READY + "\nSIGINT -> freeze\n"))
        self.assertFalse(launch.drag_ready(launch.READY + "\n*** control loop aborted: RuntimeError\n"))
        self.assertFalse(launch.drag_ready(launch.READY + "\n*** DAMP BRAKE: kd braking\n"))
        self.assertFalse(launch.drag_ready("freeze requested: stale feedback\n" + launch.READY + "\n"))

    def test_freeze_before_ready_never_starts_collector(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "session.json").write_text(json.dumps({"drag_pane": "%900", "collector_command": ["never-run"]}))
            (root / "drag.log").write_text("freeze requested: stale feedback\n")
            with patch.object(launch, "pane_alive", return_value=True), patch.object(launch.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "froze or stopped"):
                    launch.collect(root)
                spawn.assert_not_called()

    def test_dead_drag_never_starts_collector(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"session.json").write_text(json.dumps({"drag_pane":"%900", "collector_command":["never-run"]}))
            (root/"drag.log").write_text(launch.READY+"\n")
            with patch.object(launch, "pane_alive", return_value=False), patch.object(launch.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "exited"):
                    launch.collect(root)
                spawn.assert_not_called()

    def test_ready_runs_real_harmless_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            marker=root/"ran"
            cmd=[sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok')", str(marker)]
            (root/"session.json").write_text(json.dumps({"drag_pane":"%900", "collector_command":cmd}))
            (root/"drag.log").write_text(launch.READY+"\n")
            with patch.object(launch, "pane_alive", return_value=True):
                self.assertEqual(launch.collect(root),0)
            self.assertEqual(marker.read_text(),"ok")

    @unittest.skipUnless(shutil.which("tmux"), "tmux unavailable")
    def test_real_tmux_pane_logging_and_readiness_without_hardware(self):
        name = "kinesthetic-test-" + uuid4().hex[:12]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            marker = root / "ran"
            pane = launch.tmux("new-session", "-d", "-P", "-F", "#{pane_id}", "-s", name,
                               "-x", "100", "-y", "30", "bash --noprofile --norc")
            try:
                launch.tmux("set-option", "-t", name, "remain-on-exit", "on")
                launch.tmux("pipe-pane", "-o", "-t", pane, "cat >> " + launch.shlex.quote(str(root / "drag.log")))
                fake_drag = [sys.executable, "-u", "-c",
                             "import time; time.sleep(.1); print('assist at target - drag away.'); time.sleep(3)"]
                cmd = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok')", str(marker)]
                (root/"session.json").write_text(json.dumps({"drag_pane":pane, "collector_command":cmd}))
                launch.tmux("send-keys", "-l", "-t", pane, "exec " + launch.shlex.join(fake_drag))
                launch.tmux("send-keys", "-t", pane, "Enter")
                self.assertEqual(launch.collect(root), 0)
                self.assertEqual(marker.read_text(), "ok")
            finally:
                launch.tmux("kill-session", "-t", name)  # only this test's harmless fake controller


if __name__ == "__main__":
    unittest.main()
