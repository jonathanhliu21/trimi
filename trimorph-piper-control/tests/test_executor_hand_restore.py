"""RobotIO.restore_hand: reboot errored hand motors and re-apply lost settings at a (re)start (mock hand only)."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from inference.executor.executor import DXL_ADDR_POSITION_P_GAIN, RobotIO


def make_robot(gains, errors=None, answering=None):
    hand = Mock()
    hand.motor_ids = list(range(13))
    hand.config = SimpleNamespace(position_p_gain=2000)
    hand.read_hardware_error_status.return_value = {i: (errors or {}).get(i, 0) for i in range(13)}
    hand.ping.return_value = {i: 1190 for i in (answering if answering is not None else range(13))}
    client = Mock()
    client.dxl.COMM_SUCCESS = 0
    client.packet_handler.read2ByteTxRx.side_effect = lambda port, motor_id, address: (gains[motor_id], 0, 0)
    client.packet_handler.reboot.return_value = (0, 0)
    hand._client.return_value = client
    robot = RobotIO.__new__(RobotIO)
    robot.hand, robot.hand_velocity, robot.hand_acceleration = hand, 3.0, 50.0
    return robot, hand, client


class RestoreHandTests(unittest.TestCase):
    def run_restore(self, robot):
        out = io.StringIO()
        with patch('inference.executor.executor.time.sleep'), contextlib.redirect_stdout(out):
            robot.restore_hand()
        return out.getvalue()

    def test_configured_hand_is_left_alone(self):
        robot, hand, client = make_robot([2000] * 13)
        self.assertEqual(self.run_restore(robot), '')
        hand.configure.assert_not_called()
        client.packet_handler.reboot.assert_not_called()
        self.assertEqual(client.packet_handler.read2ByteTxRx.call_args.args[2], DXL_ADDR_POSITION_P_GAIN)

    def test_power_cycled_hand_gets_its_configuration_back(self):
        robot, hand, client = make_robot([400] * 13)
        out = self.run_restore(robot)
        hand.configure.assert_called_once_with(enable_torque=False)
        hand.set_motion_profile.assert_called_once_with(profile_velocity_rad_s=3.0, profile_acceleration_rad_s2=50.0)
        client.packet_handler.reboot.assert_not_called()
        self.assertIn('power cycle', out)

    def test_overloaded_motor_is_rebooted_and_reconfigured(self):
        gains = [2000] * 13
        gains[10] = 400  # the reboot resets its RAM
        robot, hand, client = make_robot(gains, errors={10: 32})
        out = self.run_restore(robot)
        self.assertEqual([c.args[1] for c in client.packet_handler.reboot.call_args_list], [10])
        hand.configure.assert_called_once_with(enable_torque=False)
        self.assertIn('motor 10 had hardware error 32 (overload)', out)
        self.assertNotIn('power cycle', out)  # its lost gain is explained by the reboot

    def test_motor_that_does_not_come_back_stops_the_run(self):
        robot, hand, client = make_robot([2000] * 13, errors={7: 32}, answering=[i for i in range(13) if i != 7])
        with self.assertRaises(RuntimeError):
            self.run_restore(robot)
        hand.configure.assert_not_called()


if __name__ == '__main__':
    unittest.main()
