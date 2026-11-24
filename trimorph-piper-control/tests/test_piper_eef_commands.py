"""EEF command normalization: preserve orientation instead of SDK clamping."""
import io
import math
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.piper_api.piper_api import PiperAPI, normalize_eef_pose
from data_collection.replay_episode import move_to_first_pose, replay_piper_commands


class EefCommandTests(unittest.TestCase):
    def assert_same_pose(self, actual, expected):
        np.testing.assert_array_equal(actual[:3], expected[:3])
        self.assertLessEqual(abs(actual[3]), math.pi)
        self.assertLessEqual(abs(actual[4]), math.pi / 2)
        self.assertLessEqual(abs(actual[5]), math.pi)
        error = Rotation.from_euler("xyz", actual[3:]).inv() * Rotation.from_euler("xyz", expected[3:])
        self.assertLess(error.magnitude(), 1e-12)

    def test_reported_warning_pose_is_wrapped_without_changing_rotation(self):
        pose = [.4, -.2, .3, -5.771995822179587, 1.304459083, -6.149964325179586]
        before = pose.copy()
        robot = Mock()
        PiperAPI(robot=robot).move_eef(pose)
        sent = robot.move_p.call_args.args[0]
        self.assert_same_pose(sent, pose)
        np.testing.assert_allclose(sent[3:], [.511189485, 1.304459083, .133220982])
        self.assertEqual(pose, before)

    def test_all_euler_branches_keep_rotation_and_valid_sdk_ranges(self):
        rng = np.random.default_rng(42)
        angles = list(rng.uniform(-8 * math.pi, 8 * math.pi, (100, 3)))
        angles += [[math.pi, math.pi / 2, -math.pi], [-math.pi, -math.pi / 2, math.pi]]
        for rpy in angles:
            pose = [.4, -.2, .3, *rpy]
            self.assert_same_pose(normalize_eef_pose(pose), pose)

    def test_preposition_and_replay_both_normalize_at_sdk_boundary(self):
        from pyAgxArm.utiles.validator import Validator

        clock = SimpleNamespace(now=0.)
        pose = np.array([.4, -.2, .3, -5.771995822179587, 1.304459083, -6.149964325179586])
        stream = (np.array([0., .008]), np.array([pose, pose + [0, 0, 0, -.01, 0, -.01]]))
        source = stream[1].copy()
        robot = Mock()
        robot.move_p.side_effect = lambda target: Validator.clamp_pose6(target, "flange_pose")
        api = PiperAPI(robot=robot)
        args = SimpleNamespace(arm_control="pose", joint_level=False, piper_rate_hz=250.,
                               arm_diag=False, no_preposition=False, preposition_seconds=.1,
                               preposition_speed_percent=10, settle_seconds=0.)

        class Stop:
            def is_set(self):
                return False

            def wait(self, seconds):
                clock.now += seconds
                return False

        output = io.StringIO()
        with patch("data_collection.replay_episode.time.monotonic", side_effect=lambda: clock.now), \
                patch("data_collection.replay_episode.time.sleep", side_effect=lambda seconds: setattr(clock, "now", clock.now + seconds)), \
                redirect_stdout(output):
            move_to_first_pose(None, api, None, stream, 0., args)
            replay_piper_commands(api, stream, 0., .008, clock.now, args, Stop())
        self.assertNotIn("Warning:", output.getvalue())
        self.assertGreaterEqual(robot.move_p.call_count, 4)
        self.assert_same_pose(robot.move_p.call_args_list[0].args[0], pose)
        self.assert_same_pose(robot.move_p.call_args.args[0], stream[1][-1])
        np.testing.assert_array_equal(stream[1], source)


if __name__ == "__main__":
    unittest.main()
