import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.teleop.analyze_teleop_log import analyze, load_run
from scripts.teleop.trimi_teleop import (
    LOG_FIELDS,
    ContactFeedback,
    check_pair,
    out_of_limits,
    parse_args,
    save_log,
    teleop_loop,
)


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
    """Reaches its goal at once, except joints held back by an object at `block`."""

    def __init__(self, lower, upper, block=None):
        self.lower, self.upper = np.asarray(lower, float), np.asarray(upper, float)
        self.block = np.full(len(self.lower), np.inf) if block is None else np.asarray(block, float)
        self.goals = []

    def clip_positions(self, positions):
        return np.clip(positions, self.lower, self.upper)

    def set_positions(self, positions, clip=True):
        self.goals.append(self.clip_positions(positions) if clip else np.asarray(positions))

    def read_pos(self):
        return np.minimum(self.goals[-1], self.block)

    def read_pos_vel_cur(self):
        position = self.read_pos()
        blocked = position < self.goals[-1]
        return position, np.zeros_like(position), np.where(blocked, -100.0, 20.0)


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

    def test_a_blocked_follower_joint_pushes_back_on_its_leader_joint(self):
        clock = Clock()
        follower = FakeFollower([-1.0, -1.0], [1.0, 1.0], block=[0.2, np.inf])
        leader = FakeLeader([[0.1, 0.1]] * 10 + [[0.5, 0.1]] * 40)
        sent, log = [], {}
        teleop_loop(leader, follower, [0.1, 0.1], rate_hz=100.0, ramp_s=0.0,
                    contact=ContactFeedback([3, 7], [3, 7]),
                    send_feedback=lambda angles, feedback: sent.append(feedback.copy()),
                    current_signs=[-1.0, 1.0], log=log,
                    keep_running=lambda: len(follower.goals) < 50, clock=clock, sleep=clock.sleep)
        # joint 0 is held 0.3 rad short of its goal: feedback is the error beyond the 0.03 rad deadband
        self.assertAlmostEqual(sent[-1][0], 0.27)
        self.assertEqual(sent[-1][1], 0.0)
        self.assertTrue(all(f[0] == 0.0 for f in sent[:10]))
        self.assertEqual({len(log[name]) for name in LOG_FIELDS}, {50})
        self.assertEqual(log["current"][-1][0], 100.0)  # joint-signed

    def test_contact_waits_until_the_joint_has_reached_its_goal(self):
        contact = ContactFeedback([3, 7], [7])
        zeros = np.zeros(2)
        for step in range(20):  # 0.2 s stuck far from the goal right after the ramp
            feedback = contact.update([0.5, 0.5], [0.0, 0.0], zeros, zeros, zeros, step * 0.01)
        self.assertEqual(list(feedback), [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, r"\[9\]"):
            ContactFeedback([3, 7], [9])

    def test_arguments_are_checked(self):
        args = parse_args(["--leader-port", "/dev/a", "--follower-port", "/dev/b"])
        self.assertEqual(args.rate_hz, 100.0)
        with self.assertRaises(SystemExit):
            parse_args(["--rate-hz", "0"])
        with self.assertRaises(SystemExit):
            parse_args(["--ramp-s", "-1"])
        with self.assertRaises(SystemExit):
            parse_args(["--feedback-motors", "3"])


class TransparencyAnalysisTests(unittest.TestCase):
    def test_floor_slope_and_clearing_error_are_recovered(self):
        rng = np.random.default_rng(0)
        t = np.arange(0, 20, 0.01)
        v = 2.0 * np.sin(t)
        q = np.cumsum(v) * 0.01
        free_current = 30 * np.tanh(v / 0.05) + 10 * v + rng.normal(0, 3, len(t))
        error = np.linspace(0.06, 0.3, len(t))
        contact_current = 400 * error + rng.normal(0, 3, len(t))
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, goal, position, velocity, current in (
                ("free", q, q, v, free_current),
                ("contact", 0.5 + error, np.full(len(t), 0.5), np.zeros(len(t)), contact_current),
            ):
                log = {"t": list(t), "leader": list(goal[:, None]), "goal": list(goal[:, None]),
                       "position": list(position[:, None]), "velocity": list(velocity[:, None]),
                       "current": list(current[:, None]), "feedback": list(np.zeros((len(t), 1)))}
                path = Path(tmp) / f"{name}.npz"
                save_log(path, log, [5], ramp_s=1.0)
                paths.append(path)
            [report] = analyze(load_run(paths[0]), load_run(paths[1]))
        self.assertLess(report.floor, 12.0)  # the noise, not the 30 mA of friction
        self.assertGreater(report.free_raw_p99, 40.0)
        self.assertAlmostEqual(report.ma_per_rad, 400.0, delta=20.0)
        self.assertLess(report.clear_error_rad, 0.04)
        self.assertGreater(report.contacts, 1000)


if __name__ == "__main__":
    unittest.main()
