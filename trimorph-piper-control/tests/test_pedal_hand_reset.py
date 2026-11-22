"""Pedal reset torque ordering without robot hardware."""
from contextlib import ExitStack, redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch

import numpy as np

from data_collection.collection.config import HAND_START_POSITIONS_RAD
from inference.executor.executor import HandCommandWriter, RobotIO, parse_args


class PedalHandResetTests(unittest.TestCase):
    def setup_robot(self, extra=()):
        args = parse_args(['--predictor', 'localhost', '--start-home', 'home_m3', *extra])
        robot = RobotIO.__new__(RobotIO)
        robot.hand = Mock()
        robot.hand.clip_positions.side_effect = lambda q: q
        robot.hand.read_pos.return_value = np.zeros(13)
        robot.piper = Mock()
        robot.joint_control = Mock()
        robot.commanded_pose = np.ones(6)
        robot.log_start = Mock()
        with patch('inference.executor.executor.threading.Thread'):
            robot.hand_writer = HandCommandWriter(robot.hand, 100)
        robot.hand_writer.submit(np.ones(13))
        return args, robot

    def environment(self, stack, arm):
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(patch('scripts.piper_api.replay_trajectory.read_joint_angles', return_value=np.zeros(6)))
        stack.enter_context(patch('scripts.teleop.quest_teleop.go_home', side_effect=arm))
        stack.enter_context(patch('inference.executor.executor.time.sleep'))

    def test_pedal_reset_relaxes_hand_then_sets_goal_before_enabling(self):
        args, robot = self.setup_robot()
        events = []

        def disable():
            self.assertIsNone(robot.hand_writer.target)
            events.append('torque_off')

        robot.hand.disable_torque.side_effect = disable
        robot.hand.set_positions.side_effect = lambda *a, **k: events.append('target')
        robot.hand.enable_torque.side_effect = lambda: events.append('torque_on')
        robot.hand.set_positions_blocking.side_effect = lambda *a, **k: events.append('wait_for_hand')
        with ExitStack() as stack:
            self.environment(stack, lambda *a: events.append('arm_return'))
            robot.return_to_start(args)
        self.assertEqual(events, ['torque_off', 'arm_return', 'target', 'torque_on', 'wait_for_hand'])
        np.testing.assert_allclose(robot.hand.set_positions.call_args.args[0], HAND_START_POSITIONS_RAD['home_m3'])
        self.assertIsNone(robot.hand_writer.target)
        self.assertIsNone(robot.joint_control.q)
        self.assertIsNone(robot.commanded_pose)

    def test_failures_do_not_restore_torque_or_continue_reset(self):
        for failure in ('disable', 'arm', 'target'):
            with self.subTest(failure=failure):
                args, robot = self.setup_robot()
                arm = Mock()
                if failure == 'arm':
                    arm.side_effect = OSError('reset failed')
                else:
                    method = robot.hand.disable_torque if failure == 'disable' else robot.hand.set_positions
                    method.side_effect = OSError('reset failed')
                with ExitStack() as stack:
                    self.environment(stack, arm)
                    with self.assertRaisesRegex(OSError, 'reset failed'):
                        robot.return_to_start(args)
                robot.hand.enable_torque.assert_not_called()
                robot.hand.set_positions_blocking.assert_not_called()
                if failure == 'disable':
                    arm.assert_not_called()

    def test_no_hand_start_restores_torque_at_measured_pose(self):
        args, robot = self.setup_robot(['--no-hand-start'])
        events = []
        robot.hand.disable_torque.side_effect = lambda: events.append('off')
        robot.hand.set_positions.side_effect = lambda *a, **k: events.append('hold')
        robot.hand.enable_torque.side_effect = lambda: events.append('on')
        with ExitStack() as stack:
            self.environment(stack, lambda *a: events.append('arm'))
            robot.return_to_start(args)
        self.assertEqual(events, ['off', 'arm', 'hold', 'on'])
        np.testing.assert_array_equal(robot.hand.set_positions.call_args.args[0], np.zeros(13))
        robot.hand.set_positions_blocking.assert_not_called()

    def test_initial_start_keeps_existing_torque_behavior(self):
        args, robot = self.setup_robot()
        with ExitStack() as stack:
            self.environment(stack, lambda *a: None)
            robot.move_to_start(args, confirm=False)
        robot.hand.disable_torque.assert_not_called()
        robot.hand.enable_torque.assert_not_called()
        robot.hand.set_positions_blocking.assert_called_once()
