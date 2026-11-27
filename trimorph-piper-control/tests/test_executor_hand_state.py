"""--hand-state: which hand positions the executor sends as the observation state (mock devices only)."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from inference.executor.executor import RobotIO, parse_args


def make_robot(hand_state, measured, target=None, max_rad=1.5):
    """A RobotIO with mocked arm and hand, built without opening any hardware."""
    robot = RobotIO.__new__(RobotIO)
    robot.piper = Mock()
    robot.piper.get_eef_pose.return_value = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
    robot.hand = Mock()
    robot.hand.read_pos.return_value = np.array(measured, dtype=float)
    robot.hand.config = SimpleNamespace(joint_lower_limits_array=np.full(13, -2.9),
                                        joint_upper_limits_array=np.zeros(13))
    robot.hand_writer = None
    robot.state_anchor, robot.commanded_pose, robot.state_note = "measured", None, "measured"
    robot.hand_state, robot.hand_state_max_rad = hand_state, max_rad
    robot.policy_hand_target = None if target is None else np.array(target, dtype=float)
    robot.hand_state_fallback = ()
    return robot


class HandStateTests(unittest.TestCase):
    def test_default_is_measured_and_command_is_accepted(self):
        self.assertEqual(parse_args(['--predictor', 'localhost', '--fake']).hand_state, 'measured')
        args = parse_args(['--predictor', 'localhost', '--fake', '--hand-state', 'command'])
        self.assertEqual((args.hand_state, args.hand_state_max_rad), ('command', 1.5))

    def test_command_with_takeover_is_rejected(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            parse_args(['--predictor', 'localhost', '--fake', '--hand-state', 'command', '--takeover', 'teleop'])
        self.assertIn('--hand-state command', stderr.getvalue())

    def test_measured_mode_ignores_the_command(self):
        robot = make_robot('measured', np.full(13, -2.4), target=np.full(13, -2.8))
        np.testing.assert_allclose(robot.read_state()[6:], -2.4, rtol=0, atol=1e-6)

    def test_command_mode_uses_measured_until_a_command_is_sent(self):
        robot = make_robot('command', np.full(13, -0.1))
        np.testing.assert_allclose(robot.read_state()[6:], -0.1, rtol=0, atol=1e-6)
        self.assertIn('no command yet', robot.state_note)

    def test_command_mode_sends_clipped_target_and_measured_for_far_motors(self):
        measured = np.full(13, -2.4)
        measured[4] = -0.4  # stalled index: 2.5 rad from its command
        robot = make_robot('command', measured, target=np.full(13, -3.1))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            hand = robot.read_state()[6:]
        expected = np.full(13, -2.9)  # target clipped to the lower joint limit
        expected[4] = -0.4
        np.testing.assert_allclose(hand, expected, rtol=0, atol=1e-6)
        self.assertIn('motor 4', out.getvalue())
        self.assertIn('hand command, measured for motors 4', robot.state_note)
        self.assertAlmostEqual(robot.last_state_read['measured_hand_4_rad'], -0.4)
        self.assertAlmostEqual(robot.last_state_read['measured_hand_0_rad'], -2.4)
        # The same fallback is not reported twice in a row.
        with contextlib.redirect_stdout(io.StringIO()) as out:
            robot.read_state()
        self.assertEqual(out.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
