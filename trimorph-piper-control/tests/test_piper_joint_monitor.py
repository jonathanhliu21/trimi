import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from data_collection.collect_hand_only import parse_args, run_hand_only
from data_collection.collection.piper_joint_monitor import (
    GREEN, RED, ROW_TOP, ROW_HEIGHT, PiperJointMonitor, joint_limit_status,
)
from data_collection.collection.piper_worker import run_piper_worker
from data_collection.collection.shared_state import PiperControllerStatus, PiperJointSample, SharedState


class PiperJointMonitorTests(unittest.TestCase):
    def test_both_limits_and_exact_three_degree_boundary(self):
        for angle, ok, label in (
            (-10.1, False, "OUTSIDE"), (-10, False, "NEAR LIMIT"),
            (-7, False, "NEAR LIMIT"), (-6.999, True, "IN RANGE"),
            (0, True, "IN RANGE"), (6.999, True, "IN RANGE"),
            (7, False, "NEAR LIMIT"), (10, False, "NEAR LIMIT"),
            (10.1, False, "OUTSIDE"), (np.nan, False, "INVALID"),
        ):
            with self.subTest(angle=angle):
                result, status = joint_limit_status(angle, -10., 10.)
                self.assertEqual(result, ok)
                self.assertTrue(status.startswith(label))

    def test_missing_stale_and_invalid_feedback_never_show_green(self):
        import cv2

        shared = SharedState()
        now = 10_000_000_000
        shared.set_piper_controller_status(PiperControllerStatus(now, "NORMAL(0x0)", 0))
        with patch.object(cv2, "namedWindow"):
            monitor = PiperJointMonitor(shared, limits_deg=[[-10., 10.]] * 6)
        for joints, sample_ns, render_ns, colors in (
            (None, now, now, [RED] * 6),
            ([0, -7, 7, -10.1, 10.1, np.nan], now, now, [GREEN] + [RED] * 5),
            ([0] * 6, now, now + 500_000_001, [RED] * 6),
        ):
            if joints is not None:
                shared.set_piper_joints(PiperJointSample(sample_ns, tuple(np.radians(joints))))
            image = monitor.render(render_ns)
            for j, color in enumerate(colors):
                self.assertEqual(tuple(image[ROW_TOP + j * ROW_HEIGHT, 16]), color)

    def test_firmware_status_overrides_in_range_joints_even_with_zero_error_bits(self):
        import cv2

        shared = SharedState()
        now = 10_000_000_000
        shared.set_piper_joints(PiperJointSample(now, (0.,) * 6))
        with patch.object(cv2, "namedWindow"):
            monitor = PiperJointMonitor(shared, limits_deg=[[-10., 10.]] * 6)
        for status, err, color in (("TARGET_POS_EXCEEDS_LIMIT(0x4)", 0, RED),
                                   ("SINGULARITY_POINT(0x3)", 0, RED),
                                   ("NORMAL(0x0)", 256, RED), ("NORMAL(0x0)", 0, GREEN)):
            shared.set_piper_controller_status(PiperControllerStatus(now, status, err))
            image = monitor.render(now)
            self.assertEqual(tuple(image[4, 4]), color)
            for j in range(6):
                self.assertEqual(tuple(image[ROW_TOP + j * ROW_HEIGHT, 16]), color)
        # Live joints cannot make missing/stale controller feedback look healthy.
        shared.set_piper_controller_status(PiperControllerStatus(now - 1_000_000_000, "NORMAL", 0))
        self.assertEqual(tuple(monitor.render(now)[4, 4]), RED)

    def test_recording_banner_uses_episode_index(self):
        import cv2

        session = SimpleNamespace(paths=SimpleNamespace(episode_index=7),
                                  start_ns=time.monotonic_ns(), lock=threading.Lock(), piper_rows=[])
        recorder = SimpleNamespace(current=lambda: session)
        with patch.object(cv2, "namedWindow"):
            monitor = PiperJointMonitor(SharedState(), recorder=recorder, limits_deg=[[-10, 10]] * 6)
        with patch.object(cv2, "putText", wraps=cv2.putText) as put_text:
            monitor.render()
        self.assertIn("COLLECTING SAMPLE 000007", [call.args[1] for call in put_text.call_args_list])

    def test_counter_latches_warnings_checks_all_frames_and_excludes_deleted(self):
        import cv2

        shared = SharedState()
        now = 10_000_000_000
        recorder = SimpleNamespace(session=None)
        recorder.current = lambda: recorder.session
        with patch.object(cv2, "namedWindow"):
            monitor = PiperJointMonitor(shared, recorder=recorder, limits_deg=[[-10, 10]] * 6)

        def feedback(status="NORMAL(0x0)"):
            shared.set_piper_joints(PiperJointSample(now, (0.,) * 6))
            shared.set_piper_controller_status(PiperControllerStatus(now, status, 0))

        def start():
            session = SimpleNamespace(paths=SimpleNamespace(episode_index=7), start_ns=now,
                                      lock=threading.Lock(), piper_rows=[], stop_reason="")
            session.piper_rows.append({f"joint_{j}": "0" for j in range(1, 7)})
            recorder.session = session
            feedback()
            monitor.render(now)
            return session

        def stop(session, reason="foot_pedal"):
            session.stop_reason = reason
            recorder.session = None
            monitor.render(now)

        stop(start())
        self.assertEqual((monitor.samples_ok, monitor.samples_collected), (1, 1))
        session = start()
        feedback("NO_SOLUTION(0x2)")
        monitor.render(now)
        feedback()
        monitor.render(now)
        stop(session)
        self.assertEqual((monitor.samples_ok, monitor.samples_collected), (1, 2))
        session = start()
        # Bad frame arrives after the last render, followed by recovery.
        session.piper_rows.extend([
            {f"joint_{j}": str(np.radians(9)) for j in range(1, 7)},
            {f"joint_{j}": "0" for j in range(1, 7)},
        ])
        stop(session)
        self.assertEqual((monitor.samples_ok, monitor.samples_collected), (1, 3))
        stop(start(), "foot_pedal_delete")
        monitor.render(now)
        self.assertEqual((monitor.samples_ok, monitor.samples_collected), (1, 3))
        with patch.object(cv2, "putText", wraps=cv2.putText) as put_text:
            monitor.render(now)
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("1 / 3", labels)
        self.assertIn("IDLE - NOT COLLECTING", labels)

    def test_cli_defaults_and_opt_out(self):
        for flags, monitor, logging in (
            ([], False, False),
            (["--use-piper", "--joint-level"], True, True),
            (["--use-piper", "--piper-joint-monitor"], True, True),
            (["--arm-mode", "gravity-comp"], True, True),
            (["--arm-mode", "gravity-comp", "--no-piper-joint-monitor"], False, False),
        ):
            with self.subTest(flags=flags), patch("sys.argv", ["collect_hand_only.py", *flags]):
                args = parse_args()
                self.assertEqual(args.show_piper_joint_monitor, monitor)
                self.assertEqual(args.piper_joint_level, logging)
        with patch("sys.argv", ["collect_hand_only.py", "--piper-joint-monitor"]), \
                patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args()

    def test_drag_teach_monitor_reads_idle_joints_without_motion_commands(self):
        shared = SharedState()
        config = SimpleNamespace(
            piper_firmware="v183", piper_interface="socketcan", piper_channel="can0",
            piper_installation_pos="unchanged", piper_payload="unchanged",
            piper_rate_hz=250., teacher_mode_poll_hz=20., piper_joint_level=True,
            show_piper_joint_monitor=True,
        )
        with patch("scripts.piper_api.piper_api.PiperAPI") as api_type:
            api = api_type.return_value
            api.is_ok.return_value = True
            api.get_state.return_value = {"teach_status": "IDLE"}
            joints = SimpleNamespace(timestamp=time.time(), joints=[.2] * 6)
            samples = []

            def cached_joints():
                if samples:
                    shared.shutdown.set()
                    # A repeated SDK cache entry must not refresh the display timestamp.
                    samples.append(shared.latest_piper_joints())
                else:
                    samples.append(None)
                return joints

            api.get_joint_state.side_effect = cached_joints
            run_piper_worker(config, shared, SimpleNamespace(current=lambda: None),
                             init_done=threading.Event())
            self.assertEqual(shared.latest_piper_joints().joints, (.2,) * 6)
            self.assertEqual(shared.latest_piper_joints(), samples[1])
            api.enable.assert_not_called()
            api.move_joint.assert_not_called()
            api.move_eef.assert_not_called()
            api.set_mit_mode.assert_not_called()
            api.disconnect.assert_called_once()

    def test_joint_window_runs_with_no_camera_or_other_monitors_and_closes(self):
        from data_collection.collection.config import HandOnlyCollectionConfig
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            config = HandOnlyCollectionConfig(
                dataset_root=Path(folder), hand_config=Path("config/hand.yaml"),
                hand_port="/dev/hand", hand_measure_rate_hz=50., warmup_time=0.,
                episode_time=0., use_piper=True, show_piper_joint_monitor=True)
            with patch("data_collection.collect_hand_only.initialize_worker", return_value=[]), \
                    patch("data_collection.collection.piper_joint_monitor.PiperJointMonitor") as monitor, \
                    patch("data_collection.collection.episode._hand_metadata", return_value={}), \
                    patch("data_collection.collect_hand_only.run_camera_view", return_value="duration_complete") as view:
                self.assertEqual(run_hand_only(config), 0)
            self.assertFalse(view.call_args.kwargs["show_camera"])
            self.assertIsNotNone(view.call_args.kwargs["on_tick"])
            monitor.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
