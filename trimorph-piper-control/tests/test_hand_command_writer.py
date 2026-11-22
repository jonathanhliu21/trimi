"""Hardware-free hand writer timing, reset, and failure checks."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from inference.executor.executor import HandCommandWriter, RobotIO, parse_args


class HandCommandWriterTests(unittest.TestCase):
    def make_writer(self, hand):
        with patch('inference.executor.executor.threading.Thread'):
            return HandCommandWriter(hand, 100.0)

    def test_repeats_latest_target_at_100_hz_and_skips_overruns(self):
        hand = Mock()
        writer = self.make_writer(hand)
        now = [0.0]
        sent = []

        def send(target, clip):
            sent.append((now[0], target.copy()))
            if len(sent) == 1:
                writer.submit(np.ones(13))
            if len(sent) == 2:
                now[0] += 0.025  # slow bus: no catch-up burst

        def wait(delay):
            now[0] += delay
            if len(sent) == 4:
                writer.stop.set()

        writer.submit(np.zeros(13))
        hand.set_positions.side_effect = send
        writer.stop.wait = wait
        with patch('inference.executor.executor.time.perf_counter', side_effect=lambda: now[0]):
            writer.run()
        np.testing.assert_allclose([t for t, _ in sent], [0.0, 0.01, 0.045, 0.055])
        np.testing.assert_array_equal(sent[0][1], np.zeros(13))
        for _, target in sent[1:]:
            np.testing.assert_array_equal(target, np.ones(13))

    def test_pause_clears_target_and_failure_reaches_caller(self):
        hand = Mock()
        writer = self.make_writer(hand)
        writer.submit(np.zeros(13))
        writer.pause()
        self.assertIsNone(writer.target)
        writer.submit(np.ones(13))
        hand.set_positions.side_effect = OSError('serial failure')
        writer.run()
        with self.assertRaisesRegex(RuntimeError, 'Hand command writer failed'):
            writer.submit(np.zeros(13))
        writer.close()
        writer.thread.join.assert_called_once()

    def test_rate_validation(self):
        self.assertEqual(parse_args(['--predictor', 'localhost']).hand_command_hz, 100)
        for value in ('0', '-1', 'nan', 'inf'):
            with self.subTest(value=value), patch('sys.stderr'), self.assertRaises(SystemExit):
                parse_args(['--predictor', 'localhost', '--hand-command-hz', value])

    def test_robot_submits_hand_target_and_checks_error_before_arm_motion(self):
        robot = RobotIO.__new__(RobotIO)
        robot.hand_writer = Mock()
        robot.hand = Mock()
        robot.piper = Mock()
        robot.joint_control = None
        action = np.r_[np.zeros(6), np.ones(13)]
        robot.apply_action(action)
        np.testing.assert_array_equal(robot.hand_writer.submit.call_args.args[0], action[6:])
        robot.hand.set_positions.assert_not_called()
        robot.piper.reset_mock()
        robot.hand_writer.check_error.side_effect = RuntimeError('writer failed')
        with self.assertRaisesRegex(RuntimeError, 'writer failed'):
            robot.apply_action(action)
        robot.piper.move_eef.assert_not_called()
