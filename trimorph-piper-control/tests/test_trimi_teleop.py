import unittest

import numpy as np

from scripts.teleop.trimi_teleop import check_pair, out_of_limits, parse_args, teleop_loop


def calibration(**changes):
    base = {
        "motor_ids": [0, 1],
        "joint_signs": {0: 1.0, 1: -1.0},
        "home_offsets": {0: 1.0, 1: 2.0},
        "homing": {"thumb": [[0, "Thumb joint 0", -0.03, 1]], "fingers": [[1, "Index PIP", 0.03, -1]]},
    }
    base.update(changes)
    return base


class PairCheckTests(unittest.TestCase):
    def test_hands_homed_from_the_same_table_pass(self):
        check_pair(calibration(), calibration(home_offsets=[1.5, 2.5]))

    def test_mismatched_calibrations_are_refused(self):
        with self.assertRaisesRegex(ValueError, "Motor IDs differ"):
            check_pair(calibration(), calibration(motor_ids=[0, 2], home_offsets=[0.0, 0.0]))
        with self.assertRaisesRegex(ValueError, "homing tables"):
            check_pair(calibration(), calibration(home_offsets=[0, 0], homing={"thumb": [], "fingers": []}))
        with self.assertRaisesRegex(ValueError, "joint signs"):
            check_pair(calibration(), calibration(home_offsets=[0, 0], joint_signs=[1.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "never homed"):
            check_pair(calibration(), calibration(home_offsets=[1.0, 2.0]))

    def test_joints_far_outside_their_limits_are_flagged(self):
        lower, upper = np.array([-3.14, -2.9]), np.array([3.14, 0.02])
        self.assertEqual(out_of_limits([0.0, 0.1], lower, upper), [])
        self.assertEqual(out_of_limits([3.14 + 2 * np.pi - 6.0, -2.9], lower, upper), [])
        self.assertEqual(out_of_limits([3.5 + 2.0, 1.0], lower, upper), [0, 1])
        self.assertEqual(out_of_limits([np.nan, 0.0], lower, upper), [0])


class FakeLeader:
    def __init__(self, readings):
        self.readings = [np.asarray(r, dtype=float) for r in readings]
        self.index = 0

    def read_pos(self):
        value = self.readings[min(self.index, len(self.readings) - 1)]
        self.index += 1
        return value


class FakeFollower:
    def __init__(self, lower, upper):
        self.lower, self.upper = np.asarray(lower, float), np.asarray(upper, float)
        self.goals = []

    def clip_positions(self, positions):
        return np.clip(positions, self.lower, self.upper)

    def set_positions(self, positions, clip=True):
        self.goals.append(self.clip_positions(positions) if clip else np.asarray(positions))

    def read_pos(self):
        return self.goals[-1]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(seconds, 0.0)


class TeleopLoopTests(unittest.TestCase):
    def run_loop(self, readings, cycles, **kwargs):
        clock = Clock()
        follower = FakeFollower([-1.0, -1.0], [1.0, 0.5])
        leader = FakeLeader(readings)
        done = teleop_loop(leader, follower, [0.0, 0.0], rate_hz=10.0,
                           keep_running=lambda: len(follower.goals) < cycles,
                           clock=clock, sleep=clock.sleep, **kwargs)
        return done, np.array(follower.goals)

    def test_follower_eases_onto_the_leader_then_copies_it_within_its_limits(self):
        done, goals = self.run_loop([[0.4, 0.8]] * 6, 6, ramp_s=0.4)
        self.assertEqual(done, 6)
        # 10 Hz: blend 0, 0.25, 0.5, 0.75, then the leader's angles (joint 1 clipped to 0.5)
        np.testing.assert_allclose(goals[:, 0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.4])
        np.testing.assert_allclose(goals[4:, 1], [0.5, 0.5])

    def test_no_ramp_copies_from_the_first_cycle(self):
        _, goals = self.run_loop([[0.3, -0.2], [0.35, -0.25]], 2, ramp_s=0.0)
        np.testing.assert_allclose(goals, [[0.3, -0.2], [0.35, -0.25]])

    def test_a_jump_in_the_leader_reading_stops_it(self):
        with self.assertRaisesRegex(RuntimeError, "jumped"):
            self.run_loop([[0.0, 0.0], [0.0, 1.5]], 5, ramp_s=0.0, max_jump_rad=1.0)
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            self.run_loop([[0.0, np.nan]], 5, ramp_s=0.0)

    def test_arguments_are_checked(self):
        args = parse_args(["--leader-port", "/dev/a", "--follower-port", "/dev/b"])
        self.assertEqual(args.rate_hz, 100.0)
        with self.assertRaises(SystemExit):
            parse_args(["--rate-hz", "0"])
        with self.assertRaises(SystemExit):
            parse_args(["--ramp-s", "-1"])


if __name__ == "__main__":
    unittest.main()
