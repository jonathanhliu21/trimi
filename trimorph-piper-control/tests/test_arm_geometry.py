"""The displayed arm must match rollout FK and never imply stale feedback is current."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from data_collection.collection.arm_geometry import flange_pose_error, gravity_up_rotation, piper_frames
from data_collection.collection.config import PIPER_START_JOINTS_DEG
from data_collection.collection.shared_state import ArmHealth, SharedState
from data_collection.collection.tracking_monitor import (
    FAIL_COLOR, TrackingMonitor, current_arm_frames, project, start_guidance,
)
from data_collection.replay_actions import piper_model


class ArmGeometryTests(unittest.TestCase):
    def test_flange_position_and_orientation_match_executor_fk(self):
        fk, _ = piper_model()
        poses = [np.zeros(6), np.radians(PIPER_START_JOINTS_DEG["kinesthetic"]),
                 np.radians([-71.61, 127.86, -77.785, -102.66, -74.895, -52.153])]
        poses.extend(np.random.default_rng(42).uniform(-2, 2, size=(12, 6)))
        for q in poses:
            with self.subTest(joints=q):
                frames = piper_frames(q)
                pose = fk(q)
                self.assertEqual(frames.shape, (7, 4, 4))
                np.testing.assert_allclose(frames[0], np.eye(4))
                np.testing.assert_allclose(frames[-1, :3, 3], pose[:3], atol=1e-10)
                np.testing.assert_allclose(frames[-1, :3, :3],
                                           Rotation.from_euler("xyz", pose[3:]).as_matrix(), atol=1e-9)

    def test_j6_rotation_changes_flange_axes_even_with_same_skeleton_endpoint(self):
        q = np.radians(PIPER_START_JOINTS_DEG["kinesthetic"])
        turned = q.copy()
        turned[5] += np.pi / 2
        before, after = piper_frames(q), piper_frames(turned)
        np.testing.assert_allclose(before[-1, :3, 3], after[-1, :3, 3], atol=1e-12)
        self.assertFalse(np.allclose(before[-1, :3, :3], after[-1, :3, :3]))

    def test_invalid_joint_arrays_are_rejected(self):
        for q in ([0] * 5, [0] * 7, [0, 0, 0, 0, 0, np.nan], [np.inf] * 6):
            with self.subTest(joints=q), self.assertRaisesRegex(ValueError, "six finite"):
                piper_frames(q)

    def test_flange_rotation_uses_shortest_angle_and_equivalent_joint_turns(self):
        target = np.zeros(6)
        current = target.copy()
        current[5], target[5] = np.radians((-179, 179))
        distance, angle = flange_pose_error(current, target)
        self.assertAlmostEqual(distance, 0)
        self.assertAlmostEqual(angle, 2)
        current = target.copy()
        current[5] += 2 * np.pi
        distance, angle = flange_pose_error(current, target)
        self.assertAlmostEqual(distance, 0)
        self.assertAlmostEqual(angle, 0)

    def test_gravity_view_is_orthonormal_and_physical_up_is_positive_z(self):
        for gravity in ((-.0863, -9.8052, .2956), (9.81, 0, 0), (0, 0, -9.81)):
            with self.subTest(gravity=gravity):
                rotation = gravity_up_rotation(gravity)
                np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(rotation), 1)
                np.testing.assert_allclose(rotation @ (-np.asarray(gravity) / np.linalg.norm(gravity)),
                                           (0, 0, 1), atol=1e-12)
        for gravity in ((0, 0, 0), (0, np.nan, 1), (1, 2)):
            with self.subTest(gravity=gravity), self.assertRaises(ValueError):
                gravity_up_rotation(gravity)


class MeasuredArmGeometryTests(unittest.TestCase):
    def make_monitor(self, shared, **kwargs):
        import cv2

        with patch.object(cv2, "namedWindow"), patch.object(cv2, "setMouseCallback"):
            return TrackingMonitor(shared, show_quest=False, show_vicon=False,
                                   show_arm=True, **kwargs)

    def test_fresh_disabled_arm_is_drawn_but_stale_or_invalid_geometry_is_hidden(self):
        now = 10**9
        q = tuple(np.radians(PIPER_START_JOINTS_DEG["kinesthetic"]))
        disabled = ArmHealth(now, False, "DRIVES OFF", 200, q)
        frames, _ = current_arm_frames(now, disabled)
        np.testing.assert_allclose(frames, piper_frames(q))
        for stamp, health in ((now + 10**9, disabled), (now, None),
                              (now, ArmHealth(now, True, "", 200)),
                              (now, ArmHealth(now, True, "", 200, (np.nan,) * 6))):
            with self.subTest(stamp=stamp, health=health):
                frames, message = current_arm_frames(stamp, health)
                self.assertIsNone(frames)
                self.assertTrue(message.startswith("CURRENT hidden"))

    def test_start_ghost_stays_visible_while_stale_current_is_not_drawn(self):
        import cv2

        shared = SharedState()
        now = 10**9
        q = tuple(np.radians(PIPER_START_JOINTS_DEG["kinesthetic"]))
        shared.set_arm_health(ArmHealth(now, True, "", 200, q))
        with patch.object(cv2, "namedWindow"), patch.object(cv2, "setMouseCallback"):
            monitor = TrackingMonitor(shared, show_quest=False, show_vicon=False,
                                      show_arm=True, start_name="kinesthetic",
                                      gravity_vector=(0, -9.81, 0))
        with patch.object(monitor, "_draw_arm", wraps=monitor._draw_arm) as draw:
            monitor.render(now)
            self.assertEqual([call.kwargs["prefix"] for call in draw.call_args_list], ["S", "C"])
        with patch.object(monitor, "_draw_arm", wraps=monitor._draw_arm) as draw:
            monitor.render(now + 10**9)
            self.assertEqual([call.kwargs["prefix"] for call in draw.call_args_list], ["S"])

    def test_explicit_task_start_controls_guidance_and_geometry_without_global_mutation(self):
        now = 10**9
        target = np.array([-40., 100., -70., -55., -45., -20.])
        health = ArmHealth(now, True, "", 200, tuple(np.radians(target)))
        legacy = dict(PIPER_START_JOINTS_DEG)
        status, error = start_guidance(now, health, "pick_cube",
                                      start_joints_deg=target)
        self.assertTrue(status.ok)
        np.testing.assert_allclose(error, 0, atol=1e-10)
        shared = SharedState()
        shared.set_arm_health(health)
        monitor = self.make_monitor(shared, start_name="pick_cube", start_joints_deg=target)
        np.testing.assert_allclose(monitor.start_frames, piper_frames(np.radians(target)))
        target[:] = 0  # Caller changes cannot silently move a live target.
        np.testing.assert_allclose(monitor.start_joints_deg, [-40, 100, -70, -55, -45, -20])
        self.assertEqual(PIPER_START_JOINTS_DEG, legacy)
        with self.assertRaisesRegex(ValueError, "six finite"):
            self.make_monitor(shared, start_name="pick_cube", start_joints_deg=[0, 0, np.nan])

    def test_start_criterion_includes_exact_distance_and_rotation_thresholds(self):
        now = 10**9
        target = np.radians(PIPER_START_JOINTS_DEG["kinesthetic"])
        radius = np.linalg.norm(piper_frames(target)[-1, :2, 3])
        angle_for_10_cm = 2 * np.arcsin(.1 / (2 * radius))
        for joint, delta, expected in ((0, angle_for_10_cm, True),
                                        (0, angle_for_10_cm + .0001, False),
                                        (5, np.radians(45), True),
                                        (5, np.radians(45.001), False)):
            with self.subTest(joint=joint, delta=delta):
                current = target.copy()
                current[joint] += delta
                health = ArmHealth(now, True, "", 200, tuple(current))
                status, error = start_guidance(now, health, "kinesthetic")
                self.assertEqual(status.ok, expected)
                self.assertIsNotNone(error)

    def test_orientation_criterion_combines_axes_instead_of_comparing_euler_components(self):
        now = 10**9
        target = np.radians(PIPER_START_JOINTS_DEG["kinesthetic"])
        current = target + np.radians([0, 0, 0, 40, 18, 10])
        relative = piper_frames(target)[-1, :3, :3].T @ piper_frames(current)[-1, :3, :3]
        self.assertTrue(np.all(np.abs(Rotation.from_matrix(relative).as_euler("xyz", degrees=True)) < 45))
        status, (distance, angle) = start_guidance(now, ArmHealth(now, True, "", 200, tuple(current)),
                                                 "kinesthetic")
        self.assertLess(distance, .1)
        self.assertGreater(angle, 45)
        self.assertFalse(status.ok)

    def test_large_joint_difference_with_equivalent_flange_is_accepted(self):
        now = 10**9
        target = np.radians(PIPER_START_JOINTS_DEG["kinesthetic"])
        current = target.copy()
        current[5] += 2 * np.pi  # Physical bounds remain a separate check.
        status, (distance, angle) = start_guidance(now, ArmHealth(now, True, "", 200, tuple(current)),
                                                 "kinesthetic")
        self.assertTrue(status.ok)
        self.assertAlmostEqual(distance, 0)
        self.assertAlmostEqual(angle, 0)

    def test_disabled_arm_shows_flange_error_without_readiness_and_stale_feedback_hides_error(self):
        now = 10**9
        target = tuple(np.radians(PIPER_START_JOINTS_DEG["kinesthetic"]))
        health = ArmHealth(now, False, "DRIVES OFF", 200, target)
        status, error = start_guidance(now, health, "kinesthetic")
        self.assertFalse(status.ok)
        np.testing.assert_allclose(error, (0, 0), atol=1e-10)
        status, error = start_guidance(now + 10**9, health, "kinesthetic")
        self.assertFalse(status.ok)
        self.assertIsNone(error)

    def test_definition_mode_has_no_legacy_ghost_and_never_implies_start_readiness(self):
        import cv2

        now = 10**9
        shared = SharedState()
        shared.set_arm_health(ArmHealth(now, True, "", 200,
                                        tuple(np.radians(PIPER_START_JOINTS_DEG["kinesthetic"]))))
        monitor = self.make_monitor(shared, start_name="kinesthetic", definition_mode=True)
        self.assertIsNone(monitor.start_frames)
        with patch.object(monitor, "_draw_arm", wraps=monitor._draw_arm) as draw, \
                patch.object(cv2, "putText", wraps=cv2.putText) as put_text:
            image = monitor.render(now)
        self.assertEqual([call.kwargs["prefix"] for call in draw.call_args_list], ["C"])
        self.assertEqual(tuple(image[14, 30]), FAIL_COLOR)
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("DEFINE START", labels)
        self.assertTrue(any("NO TASK START SAVED" in label for label in labels))

    def test_initial_fit_fills_scene_and_mouse_zoom_persists_until_double_click(self):
        import cv2

        now = 10**9
        shared = SharedState()
        q = np.radians([-71.61, 127.86, -77.785, -102.66, -74.895, -52.153])
        shared.set_arm_health(ArmHealth(now, False, "DRIVES OFF", 200, tuple(q)))
        monitor = self.make_monitor(shared, start_name="kinesthetic", size=(1120, 820),
                                    gravity_vector=(0, -9.81, 0))
        monitor.render(now)
        points = np.vstack((piper_frames(q)[:, :3, 3], monitor.start_frames[:, :3, 3]))
        pixels = project(points @ monitor.arm_view_rotation.T, monitor.yaw, monitor.pitch,
                         monitor.scale, monitor.arm_center)
        self.assertGreaterEqual(pixels[:, 0].min(), 49)
        self.assertLessEqual(pixels[:, 0].max(), monitor.width - 69)
        self.assertGreaterEqual(pixels[:, 1].min(), 257)
        self.assertLessEqual(pixels[:, 1].max(), monitor.height - 89)
        self.assertGreater(np.ptp(pixels[:, 1]), .70 * (monitor.height - 90 - 258))
        fitted_scale = monitor.scale
        monitor._on_mouse(cv2.EVENT_MOUSEWHEEL, 0, 0, 1, None)
        manual_scale = monitor.scale
        self.assertGreater(manual_scale, fitted_scale)
        monitor.render(now)
        self.assertEqual(monitor.scale, manual_scale)
        monitor._on_mouse(cv2.EVENT_LBUTTONDBLCLK, 0, 0, 0, None)
        monitor.render(now)
        self.assertAlmostEqual(monitor.scale, fitted_scale)


if __name__ == "__main__":
    unittest.main()
