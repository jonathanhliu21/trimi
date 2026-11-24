import unittest
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

from data_collection.collection.config import ARM_MODE_GRAVITY_COMP, PIPER_JOINT_FIELDS
from data_collection.collection.piper_worker import (
    arm_health_problem,
    gravity_comp_sample,
    piper_workspace_warning,
    run_piper_worker,
)
from data_collection.collection.shared_state import SharedState
from scripts.piper_api.piper_api import (
    RECEIVE_ONLY_ROBOT_CALLS,
    PiperArmStatus,
    PiperJointState,
    ReceiveOnlyPiperAPI,
)


class PiperWorkspaceWarningTests(unittest.TestCase):
    def test_cartesian_worker_records_measured_pose_without_reading_joints(self):
        shutdown = threading.Event()
        rows = []
        session = SimpleNamespace(elapsed_s=lambda _: 0., append_piper=rows.append)
        config = SimpleNamespace(
            piper_firmware="v183", piper_interface="socketcan", piper_channel="can0",
            piper_installation_pos=None, piper_payload=None,
            piper_rate_hz=250., teacher_mode_poll_hz=20., piper_joint_level=False,
        )
        pose = [.4, -.2, .3, .5, -.6, .7]
        with patch("scripts.piper_api.piper_api.PiperAPI") as api_type:
            api = api_type.return_value
            api.is_ok.return_value = True
            api.get_state.return_value = {"teach_status": "START_RECORDING"}

            def read_pose():
                shutdown.set()
                return SimpleNamespace(pose=pose, timestamp=1., hz=250.)

            api.get_eef_state.side_effect = read_pose
            run_piper_worker(config, SimpleNamespace(shutdown=shutdown),
                             SimpleNamespace(current=lambda: session), init_done=threading.Event())
            api.get_joint_state.assert_not_called()
            api.enable.assert_not_called()
            api.move_eef.assert_not_called()
            api.move_joint.assert_not_called()
            api.disconnect.assert_called_once()
        self.assertEqual(len(rows), 1)
        self.assertEqual([float(rows[0][k]) for k in ("x", "y", "z", "roll", "pitch", "yaw")], pose)
        self.assertFalse(any(k.startswith("joint_") for k in rows[0]))

    def test_normal_state_has_no_warning(self):
        self.assertIsNone(
            piper_workspace_warning(
                {"arm_status": "NORMAL(0x0)", "err_code": 0}
            )
        )

    def test_target_limit_is_reported(self):
        warning = piper_workspace_warning(
            {"arm_status": "TARGET_POS_EXCEEDS_LIMIT(0x4)", "err_code": 0}
        )

        self.assertIsNotNone(warning)
        self.assertIn("workspace/kinematic limit", warning)
        self.assertIn("TARGET_POS_EXCEEDS_LIMIT", warning)

    def test_singularity_and_joint_limit_bits_are_reported(self):
        warning = piper_workspace_warning(
            {
                "arm_status": "SINGULARITY_POINT(0x3)",
                "err_code": (1 << 8) | (1 << 12),
            }
        )

        self.assertIsNotNone(warning)
        self.assertIn("SINGULARITY_POINT", warning)
        self.assertIn("joint_angle_limit=[1, 5]", warning)

    def test_joint_level_worker_records_joints_without_commanding_arm(self):
        shutdown = threading.Event()
        shared = SimpleNamespace(shutdown=shutdown)
        rows = []
        session = SimpleNamespace(
            elapsed_s=lambda _sample_ns: 0.0,
            append_piper=rows.append,
        )
        recorder = SimpleNamespace(current=lambda: session)
        config = SimpleNamespace(
            piper_firmware="v183",
            piper_interface="socketcan",
            piper_channel="can0",
            piper_installation_pos="right",
            piper_payload="half",
            piper_rate_hz=250.0,
            teacher_mode_poll_hz=20.0,
            piper_joint_level=True,
        )
        init_done = threading.Event()
        with patch("scripts.piper_api.piper_api.PiperAPI") as api_type:
            api = api_type.return_value
            api.is_ok.return_value = True
            api.get_fps.return_value = 200.0
            api.get_state.return_value = {
                "teach_status": "START_RECORDING", "arm_status": "NORMAL", "err_code": 0
            }
            api.robot.get_joint_angle_vel_limits.return_value = SimpleNamespace(
                msg=SimpleNamespace(min_angle_limit=-1.0, max_angle_limit=1.0)
            )

            def eef_once():
                shutdown.set()
                return SimpleNamespace(pose=[0.1] * 6, timestamp=1.0, hz=200.0)

            api.get_eef_state.side_effect = eef_once
            api.get_joint_state.return_value = SimpleNamespace(joints=[0.2] * 6)
            run_piper_worker(config, shared, recorder, init_done=init_done)
            api.enable.assert_not_called()
            api.move_eef.assert_not_called()
            api.move_joint.assert_not_called()

        self.assertTrue(init_done.is_set())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["joint_1"], "0.200000000")
        self.assertEqual(rows[0]["joint_6"], "0.200000000")

    def test_joint_level_worker_warns_when_measured_joint_exceeds_limit(self):
        shutdown = threading.Event()
        shared = SimpleNamespace(shutdown=shutdown)
        rows = []
        session = SimpleNamespace(elapsed_s=lambda _sample_ns: 0.0, append_piper=rows.append)
        recorder = SimpleNamespace(current=lambda: session)
        config = SimpleNamespace(
            piper_firmware="v183",
            piper_interface="socketcan",
            piper_channel="can0",
            piper_installation_pos="right",
            piper_payload="half",
            piper_rate_hz=250.0,
            teacher_mode_poll_hz=20.0,
            piper_joint_level=True,
        )
        with (
            patch("scripts.piper_api.piper_api.PiperAPI") as api_type,
            patch("data_collection.collection.piper_worker.print") as printed,
        ):
            api = api_type.return_value
            api.is_ok.return_value = True
            api.get_fps.return_value = 200.0
            api.get_state.return_value = {"teach_status": "START_RECORDING", "arm_status": "NORMAL", "err_code": 0}
            api.robot.get_joint_angle_vel_limits.return_value = SimpleNamespace(
                msg=SimpleNamespace(min_angle_limit=-1.0, max_angle_limit=1.0)
            )

            def eef_once():
                shutdown.set()
                return SimpleNamespace(pose=[0.1] * 6, timestamp=1.0, hz=200.0)

            api.get_eef_state.side_effect = eef_once
            api.get_joint_state.return_value = SimpleNamespace(joints=[0.2] * 4 + [1.01, 0.2])
            run_piper_worker(config, shared, recorder, init_done=threading.Event())

        self.assertEqual(rows[0]["joint_5"], "1.010000000")
        self.assertTrue(any("J5 above maximum" in call.args[0] for call in printed.call_args_list))



def fake_sdk_robot(joint_stamps, *, enabled=True, err_code=0, arm_status="NORMAL",
                   eef_stamp_offset=0.0, on_exhausted=None):
    """A pyAgxArm robot double that serves cached feedback and records every call.

    get_joint_angles serves one frame per entry of joint_stamps (repeats included,
    as the SDK cache repeats until a new frame arrives), then calls on_exhausted.
    """
    robot = Mock()
    robot.is_ok.return_value = True
    robot.get_fps.return_value = 600.0
    stamps = list(joint_stamps)

    def joint_angles():
        if len(stamps) > 1:
            stamp = stamps.pop(0)
        else:
            stamp = stamps[0]
            if on_exhausted is not None:
                on_exhausted()
        robot.last_stamp = stamp
        return SimpleNamespace(msg=[0.1, 1.2, -0.9, 0.0, 0.7, 0.3], timestamp=stamp, hz=200.0)

    robot.get_joint_angles.side_effect = joint_angles
    robot.get_motor_states.side_effect = lambda joint: SimpleNamespace(
        msg=SimpleNamespace(position=0.0, velocity=0.01 * joint, current=0.1 * joint, torque=0.5 * joint),
        timestamp=robot.last_stamp, hz=200.0,
    )
    robot.get_driver_states.side_effect = lambda joint: SimpleNamespace(
        msg=SimpleNamespace(foc_status=SimpleNamespace(driver_enable_status=enabled)),
        timestamp=robot.last_stamp, hz=50.0,
    )
    robot.get_arm_status.side_effect = lambda: SimpleNamespace(
        msg=SimpleNamespace(ctrl_mode="CAN_CTRL", arm_status=arm_status, mode_feedback="MIT",
                            teach_status="OFF", motion_status="ARRIVED", err_code=err_code),
        timestamp=robot.last_stamp, hz=200.0,
    )
    robot.get_flange_pose.side_effect = lambda: SimpleNamespace(
        msg=[0.3, -0.2, 0.25, 3.0, 0.1, 1.5], timestamp=robot.last_stamp + eef_stamp_offset, hz=200.0,
    )
    return robot


def gravity_comp_config(disable_j6=False):
    return SimpleNamespace(
        arm_mode=ARM_MODE_GRAVITY_COMP,
        piper_firmware="v183", piper_interface="socketcan", piper_channel="can0",
        # Deliberately set: the listener must ignore them and send nothing.
        piper_installation_pos="right", piper_payload="half", piper_joint_level=True,
        piper_rate_hz=20000.0, teacher_mode_poll_hz=20.0,
        grav_comp_disable_j6=disable_j6,
    )


class GravityCompListenerTests(unittest.TestCase):
    def run_listener(self, robot, shared, session=None, *, disable_j6=False):
        rows = []
        if session is None:
            session = SimpleNamespace(elapsed_s=lambda _ns: 0.5, append_piper=rows.append)
        init_done = threading.Event()
        with (
            patch("scripts.piper_api.piper_api.create_piper_robot", return_value=robot),
            patch("scripts.piper_api.piper_api.time.sleep"),
            patch("data_collection.collection.piper_worker.print") as printed,
        ):
            run_piper_worker(gravity_comp_config(disable_j6), shared,
                             SimpleNamespace(current=lambda: session), init_done=init_done)
        self.assertTrue(init_done.is_set())
        return rows, printed

    def assert_receive_only(self, robot):
        called = {name.split(".")[0] for name, _args, _kwargs in robot.mock_calls}
        self.assertLessEqual(called, RECEIVE_ONLY_ROBOT_CALLS)
        robot.connect.assert_called_once()
        robot.disconnect.assert_called_once()

    def test_listener_sends_nothing_and_keeps_one_row_per_joint_frame(self):
        shared = SharedState()
        stamp = time.time()
        robot = fake_sdk_robot(
            [stamp, stamp, stamp + 0.005, stamp + 0.005, stamp + 0.005, stamp + 0.010],
            on_exhausted=shared.shutdown.set,
        )
        rows, _printed = self.run_listener(robot, shared)

        self.assert_receive_only(robot)
        for transmitting in ("set_installation_pos", "set_payload", "enable", "disable",
                             "set_motion_mode", "get_joint_angle_vel_limits", "move_mit", "move_j"):
            getattr(robot, transmitting).assert_not_called()
        self.assertEqual([row["sample_idx"] for row in rows], [0, 1, 2])
        self.assertEqual([float(row["robot_timestamp_s"]) for row in rows],
                         [round(stamp + d, 6) for d in (0.0, 0.005, 0.010)])
        row = rows[0]
        self.assertEqual([float(row[name]) for name in PIPER_JOINT_FIELDS], [0.1, 1.2, -0.9, 0.0, 0.7, 0.3])
        self.assertEqual(float(row["vel_2"]), 0.02)
        self.assertEqual(float(row["current_3"]), 0.3)
        self.assertEqual(float(row["torque_6"]), 3.0)
        self.assertEqual([row[f"enable_{joint}"] for joint in range(1, 7)], [1] * 6)
        self.assertEqual((row["ctrl_mode"], row["mode_feedback"], row["err_code"], row["arm_ok"]),
                         ("CAN_CTRL", "MIT", 0, 1))
        self.assertEqual(float(row["x"]), 0.3)
        self.assertLess(abs(row["joint_rx_monotonic_ns"] - row["monotonic_ns"]), 1e9)
        health = shared.latest_arm_health()
        self.assertTrue(health.ok)
        self.assertEqual(health.rate_hz, 200.0)
        feedback = shared.latest_piper_joints()
        self.assertEqual(feedback.joints, tuple(health.joints))
        self.assertEqual(feedback.monotonic_ns, rows[-1]["joint_rx_monotonic_ns"])

    def test_disabled_drive_marks_rows_and_warns(self):
        shared = SharedState()
        robot = fake_sdk_robot([time.time()], enabled=False, err_code=0x0100,
                               on_exhausted=shared.shutdown.set)
        rows, printed = self.run_listener(robot, shared)

        self.assert_receive_only(robot)
        self.assertEqual(rows[0]["arm_ok"], 0)
        self.assertEqual(rows[0]["enable_1"], 0)
        health = shared.latest_arm_health()
        self.assertFalse(health.ok)
        self.assertIn("DRIVES OFF J1,2,3,4,5,6", health.problem)
        self.assertIn("ERR 0x0100", health.problem)
        warnings = [call.args[0] for call in printed.call_args_list if "not OK" in str(call.args[0])]
        self.assertEqual(len(warnings), 1)
        self.assertIn("flagged", warnings[0])

    def test_firmware_target_limit_with_zero_error_bits_is_unhealthy(self):
        shared = SharedState()
        robot = fake_sdk_robot([time.time()], arm_status="TARGET_POS_EXCEEDS_LIMIT(0x4)",
                               on_exhausted=shared.shutdown.set)
        rows, _ = self.run_listener(robot, shared)
        self.assert_receive_only(robot)
        self.assertEqual(rows[0]["err_code"], 0)
        self.assertEqual(rows[0]["arm_ok"], 0)
        self.assertFalse(shared.latest_arm_health().ok)
        self.assertIn("TARGET_POS_EXCEEDS_LIMIT", shared.latest_arm_health().problem)
        self.assertEqual(shared.latest_piper_controller_status().arm_status,
                         "TARGET_POS_EXCEEDS_LIMIT(0x4)")

    def test_passive_j6_keeps_all_feedback_without_transmitting(self):
        shared = SharedState()
        robot = fake_sdk_robot([time.time()], on_exhausted=shared.shutdown.set)
        robot.get_driver_states.side_effect = lambda joint: SimpleNamespace(
            msg=SimpleNamespace(foc_status=SimpleNamespace(driver_enable_status=joint != 6)),
            timestamp=robot.last_stamp, hz=50.)
        rows, _ = self.run_listener(robot, shared, disable_j6=True)
        self.assert_receive_only(robot)
        self.assertEqual(rows[0]["enable_6"], 0)
        self.assertEqual(rows[0]["arm_ok"], 1)
        self.assertEqual(float(rows[0]["joint_6"]), .3)
        self.assertTrue(shared.latest_arm_health().ok)

    def test_stale_firmware_end_pose_is_left_blank(self):
        shared = SharedState()
        robot = fake_sdk_robot([time.time()], eef_stamp_offset=-2.0, on_exhausted=shared.shutdown.set)
        rows, _printed = self.run_listener(robot, shared)
        self.assertEqual([rows[0][name] for name in ("x", "y", "z", "roll", "pitch", "yaw")], [""] * 6)
        self.assertNotEqual(rows[0]["eef_timestamp_s"], "")

    def test_no_feedback_refuses_to_start(self):
        robot = fake_sdk_robot([time.time()])
        robot.is_ok.return_value = False
        with (
            patch("scripts.piper_api.piper_api.create_piper_robot", return_value=robot),
            patch("scripts.piper_api.piper_api.time.sleep"),
            patch("data_collection.collection.piper_worker.print"),
            self.assertRaisesRegex(RuntimeError, "No Piper feedback"),
        ):
            run_piper_worker(gravity_comp_config(), SharedState(),
                             SimpleNamespace(current=lambda: None), init_done=threading.Event())
        self.assert_receive_only(robot)

    def test_sample_leaves_missing_feedback_blank(self):
        joints = PiperJointState(joints=[0.0] * 6, timestamp=1000.0, hz=200.0)
        row, problem = gravity_comp_sample(joints, None, [None] * 6, None, None,
                                           now_ns=5_000_000_000, wall_s=1000.002)
        self.assertEqual(row["vel_1"], "")
        self.assertEqual(row["enable_1"], "")
        self.assertEqual(row["err_code"], "")
        self.assertEqual(row["x"], "")
        self.assertEqual(row["arm_ok"], 0)
        self.assertEqual(row["joint_rx_monotonic_ns"], 5_000_000_000 - 2_000_000)
        self.assertIn("NO ARM STATUS", problem)
        # A receive time that is not on the host clock is not converted.
        row, _ = gravity_comp_sample(joints, None, [True] * 6, None, None, now_ns=1, wall_s=2000.0)
        self.assertEqual(row["joint_rx_monotonic_ns"], "")

    def test_health_problem(self):
        self.assertEqual(arm_health_problem([True] * 6, 0), "")
        self.assertEqual(arm_health_problem([True, False, True, True, None, True], 0),
                         "DRIVES OFF J2 NO DRIVER STATE J5")


class ReceiveOnlyPiperAPITests(unittest.TestCase):
    def test_transmitting_methods_are_not_exposed(self):
        api = ReceiveOnlyPiperAPI(robot=Mock())
        for name in ("enable", "disable", "set_mit_mode", "set_position_mode", "set_speed_percent",
                     "set_installation_pos", "set_payload", "move_joint", "move_eef",
                     "move_mit_torque", "move_mit_torques", "get_firmware", "get_state", "robot"):
            self.assertFalse(hasattr(api, name), name)

    def test_connect_and_reads_only_use_cached_sdk_calls(self):
        robot = fake_sdk_robot([123.0])
        api = ReceiveOnlyPiperAPI(robot=robot)
        with patch("scripts.piper_api.piper_api.time.sleep"):
            self.assertTrue(api.connect())
        self.assertEqual(api.get_joint_state().timestamp, 123.0)
        self.assertEqual(api.get_motor_state().torques, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        self.assertEqual(api.get_driver_enable_status(), [True] * 6)
        self.assertIsInstance(api.get_arm_status(), PiperArmStatus)
        self.assertEqual(api.get_eef_state().pose, [0.3, -0.2, 0.25, 3.0, 0.1, 1.5])
        api.is_ok(), api.get_fps()
        api.disconnect()
        called = {name.split(".")[0] for name, _args, _kwargs in robot.mock_calls}
        self.assertLessEqual(called, RECEIVE_ONLY_ROBOT_CALLS)
        robot.set_installation_pos.assert_not_called()
        robot.set_payload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
