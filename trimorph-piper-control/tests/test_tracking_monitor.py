import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from data_collection.collection.quest_worker import QUEST_TRACKER_NAME
from data_collection.collection.shared_state import ArmHealth, SharedState, TrackerSample
from data_collection.collection.tracking_monitor import (
    arm_status,
    FAIL_COLOR,
    OK_COLOR,
    StreamStatus,
    TrackingMonitor,
    display_frames,
    project,
    stream_status,
)
from data_collection.collect_hand_only import main, parse_args
from data_collection.collection.config import PIPER_START_JOINTS_DEG

SECOND = 1_000_000_000


def sample(ns, position=(0.0, 0.0, 0.0), lost=False, count=1):
    return TrackerSample(ns, tuple(position), (0.0, 0.0, 0.0, 1.0), lost, count)


class StreamStatusTests(unittest.TestCase):
    def test_green_only_for_fresh_tracked_moving_samples(self):
        now = 10 * SECOND
        moving = np.array([[0, 0, 0], [0.01, 0, 0]])
        still = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.50005]])
        self.assertEqual(stream_status(now, None, None, check_frozen=True).text, "NO DATA")
        self.assertEqual(stream_status(now, sample(now - SECOND), moving, check_frozen=True).text,
                         "NO SIGNAL 1.0s")
        self.assertEqual(stream_status(now, sample(now, lost=True), moving, check_frozen=False).text,
                         "OBJECT LOST")
        self.assertEqual(stream_status(now, sample(now), still, check_frozen=True).text, "POSITION FROZEN")
        # Vicon is not checked for stillness, and an unfilled window is not judged frozen.
        self.assertTrue(stream_status(now, sample(now), still, check_frozen=False).ok)
        self.assertTrue(stream_status(now, sample(now), None, check_frozen=True).ok)
        self.assertTrue(stream_status(now, sample(now - SECOND // 4), moving, check_frozen=True).ok)


class DisplayFrameTests(unittest.TestCase):
    def test_calibration_maps_trackers_to_piper_axes(self):
        quest, vicon = display_frames(None)
        self.assertEqual((quest.label, vicon.label), ("own axes", "own axes"))
        a = np.diag([2.0, 3.0, 4.0])
        b = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        w = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        quest, vicon = display_frames(dict(
            linear_matrix=a.tolist(), orientation=dict(rotation_matrix=b.tolist()),
            vicon_reference={"run": dict(vicon_to_piper_rotation=w.tolist())}))
        np.testing.assert_allclose(quest.linear, a)
        np.testing.assert_allclose(quest.rotation, b)
        np.testing.assert_allclose(vicon.linear, w)
        self.assertEqual((quest.label, vicon.label), ("Piper axes", "Piper axes"))

    def test_projection_puts_up_on_top_and_scales(self):
        center = (100, 100)
        origin, up, x = project([[0, 0, 0], [0, 0, 1], [1, 0, 0]], 0.0, 0.0, 50.0, center)
        np.testing.assert_array_equal(origin, center)
        np.testing.assert_array_equal(up, (100, 50))
        np.testing.assert_array_equal(x, (150, 100))


class TrackingMonitorTests(unittest.TestCase):
    def make_monitor(self, shared, **kwargs):
        import cv2

        with patch.object(cv2, "namedWindow"), patch.object(cv2, "setMouseCallback"):
            return TrackingMonitor(shared, show_quest=True, show_vicon=True, **kwargs)

    def box_color(self, image, index, boxes):
        width = (image.shape[1] - 130 - 12 * (boxes + 2)) // boxes
        return tuple(int(v) for v in image[45, 12 + index * (width + 12) + width - 5])

    def test_status_boxes_turn_red_when_a_stream_drops(self):
        shared = SharedState()
        monitor = self.make_monitor(shared)
        now = 100 * SECOND
        image = monitor.render(now)
        self.assertEqual(self.box_color(image, 0, 2), FAIL_COLOR)  # Quest: no data
        self.assertEqual(self.box_color(image, 1, 2), FAIL_COLOR)  # Vicon: no topic yet

        shared.set_tracker_pose(QUEST_TRACKER_NAME, now, (0, 0, 0), (0, 0, 0, 1))
        shared.set_tracker_pose("/vicon/Yori/Yori", now, (1, 2, 3), (0, 0, 0, 1))
        image = monitor.render(now)
        self.assertEqual(self.box_color(image, 0, 2), OK_COLOR)
        self.assertEqual(self.box_color(image, 1, 2), OK_COLOR)

        shared.set_tracker_pose("/vicon/Yori/Yori", now, (0, 0, 0), (0, 0, 0, 1), lost=True)
        image = monitor.render(now + SECOND)  # Quest silent for 1 s, Vicon lost
        self.assertEqual(self.box_color(image, 0, 2), FAIL_COLOR)
        self.assertEqual(self.box_color(image, 1, 2), FAIL_COLOR)

    def test_frozen_quest_turns_red_after_one_second(self):
        shared = SharedState()
        monitor = self.make_monitor(shared)
        now = 100 * SECOND
        for step in range(25):
            t = now + step * SECOND // 20
            shared.set_tracker_pose(QUEST_TRACKER_NAME, t, (0.0934, 0.8303, 1.3493), (0, 0, 0, 1))
            image = monitor.render(t)
        self.assertEqual(self.box_color(image, 0, 2), FAIL_COLOR)

    def test_trails_are_relative_to_the_first_pose_and_mapped_by_calibration(self):
        shared = SharedState()
        a = np.diag([2.0, 2.0, 2.0])
        monitor = self.make_monitor(shared, calibration=dict(
            linear_matrix=a.tolist(), orientation=dict(rotation_matrix=np.eye(3).tolist())))
        now = 100 * SECOND
        shared.set_tracker_pose(QUEST_TRACKER_NAME, now, (1.0, 1.0, 1.0), (0, 0, 0, 1))
        monitor.render(now)
        shared.set_tracker_pose(QUEST_TRACKER_NAME, now + 1, (1.1, 1.0, 1.0), (0, 0, 0, 1))
        monitor.render(now + 1)
        stream = monitor.streams[QUEST_TRACKER_NAME]
        np.testing.assert_allclose(stream.pose[0], (0.2, 0.0, 0.0), atol=1e-12)
        self.assertEqual(len(stream.trail), 2)
        monitor._on_mouse(monitor.cv2.EVENT_LBUTTONDBLCLK, 0, 0, 0, None)
        self.assertIsNone(stream.pose)
        self.assertEqual(len(stream.trail), 0)


class ArmBoxTests(unittest.TestCase):
    def test_arm_status_text(self):
        now = 100 * SECOND
        self.assertEqual(arm_status(now, None).text, "NO DATA")
        self.assertEqual(arm_status(now, ArmHealth(now, True, "", 199.6)), StreamStatus(True, "OK 200 Hz"))
        self.assertEqual(arm_status(now, ArmHealth(now, False, "DRIVES OFF J2", 200.0)),
                         StreamStatus(False, "DRIVES OFF J2"))
        self.assertFalse(arm_status(now + SECOND, ArmHealth(now, True, "", 200.0)).ok)

    def test_arm_box_alone_follows_the_arm_health(self):
        import cv2

        shared = SharedState()
        with patch.object(cv2, "namedWindow"), patch.object(cv2, "setMouseCallback"):
            monitor = TrackingMonitor(shared, show_quest=False, show_vicon=False, show_arm=True)
        box = TrackingMonitorTests.box_color
        now = 100 * SECOND
        self.assertEqual(box(None, monitor.render(now), 0, 1), FAIL_COLOR)
        shared.set_arm_health(ArmHealth(now, True, "", 200.0))
        self.assertEqual(box(None, monitor.render(now), 0, 1), OK_COLOR)
        shared.set_arm_health(ArmHealth(now, False, "ERR 0x0100", 200.0))
        self.assertEqual(box(None, monitor.render(now), 0, 1), FAIL_COLOR)

    def test_start_readiness_controls_prominent_box_and_banner(self):
        import cv2

        shared = SharedState()
        with patch.object(cv2, "namedWindow"), patch.object(cv2, "setMouseCallback"):
            monitor = TrackingMonitor(shared, show_quest=False, show_vicon=False, show_arm=True,
                                      start_name="kinesthetic", passive_j6=True)
        now = 100 * SECOND
        target = np.radians(PIPER_START_JOINTS_DEG["kinesthetic"])
        far = target + np.radians([-15.08, -8.58, 32.84, -25.86, -119.94, -94.71])
        for joints, ok, problem, render_time, color, label, health in (
            (far, True, "", now, FAIL_COLOR, "NOT AT START kinesthetic", "ARM FEEDBACK: OK 200 Hz"),
            (target, True, "", now, OK_COLOR, "AT START kinesthetic", "ARM FEEDBACK: OK 200 Hz"),
            (target, False, "ERR 0x0100", now, FAIL_COLOR, "NOT AT START kinesthetic",
             "ARM FEEDBACK: ERR 0x0100"),
            (target, True, "", now + SECOND, FAIL_COLOR, "NOT AT START kinesthetic",
             "ARM FEEDBACK: NO SIGNAL 1.0s"),
        ):
            with self.subTest(label=label, health=health):
                shared.set_arm_health(ArmHealth(now, ok, problem, 200., tuple(joints)))
                with patch.object(cv2, "putText", wraps=cv2.putText) as put_text:
                    image = monitor.render(render_time)
                self.assertEqual(tuple(image[14, 30]), color)  # Prominent START status box.
                self.assertEqual(tuple(image[82, 30]), color)  # Filled readiness banner.
                labels = [call.args[1] for call in put_text.call_args_list]
                self.assertIn("START", labels)
                self.assertIn(label + " (J6 passive: align manually)", labels)
                self.assertTrue(any(text.startswith(health) for text in labels))
                if render_time == now:
                    self.assertTrue(any(text.startswith("End effector:") for text in labels))
                    self.assertIn("Ready within 10 cm AND 45 deg total rotation (flange frame)", labels)


class MonitorCliTests(unittest.TestCase):
    def test_monitor_follows_quest_and_mocap_and_can_be_disabled(self):
        with patch("sys.argv", ["collect_hand_only.py"]):
            self.assertTrue(parse_args().show_tracking_monitor)
        with patch("sys.argv", ["collect_hand_only.py", "--no-monitor"]):
            self.assertFalse(parse_args().show_tracking_monitor)
        with patch("sys.argv", ["collect_hand_only.py", "--no-quest", "--no-mocap"]):
            self.assertFalse(parse_args().show_tracking_monitor)
        for argv in (["--no-quest", "--no-mocap", "--monitor"],
                     ["--no-monitor", "--monitor-calibration", "cal.json"]):
            with (self.subTest(argv=argv), patch("sys.argv", ["collect_hand_only.py", *argv]),
                  patch("sys.stderr"), self.assertRaises(SystemExit)):
                parse_args()

    def test_monitor_options_reach_the_config(self):
        with (
            patch("sys.argv", ["collect_hand_only.py", "--monitor-calibration", "cal.json"]),
            patch("data_collection.collect_hand_only.run_hand_only", return_value=0) as run,
        ):
            self.assertEqual(main(), 0)
        config = run.call_args.args[0]
        self.assertTrue(config.show_tracking_monitor)
        self.assertEqual(str(config.monitor_calibration), "cal.json")


if __name__ == "__main__":
    unittest.main()
