import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.teleop.analyze_teleop_log import analyze, load_run
from scripts.teleop.trimi_teleop import (
    LOG_FIELDS,
    ContactDetector,
    WallContact,
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

    def read_pos_vel(self):
        return self.read_pos(), np.zeros_like(self.readings[0])

    def read_pos_vel_cur(self):
        return self.read_pos(), np.zeros_like(self.readings[0]), np.zeros_like(self.readings[0])


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
        goal = self.goals[-1] if self.goals else np.zeros(len(self.lower))
        return np.minimum(goal, self.block)

    def read_pos_vel_cur(self):
        position = self.read_pos()
        blocked = position < (self.goals[-1] if self.goals else position)
        return position, np.zeros_like(position), np.where(blocked, -300.0, 20.0)


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

    def test_a_blocked_follower_joint_puts_up_a_wall_on_its_leader_joint(self):
        clock = Clock()
        follower = FakeFollower([-1.0, -1.0], [1.0, 1.0], block=[0.2, np.inf])
        leader = FakeLeader([[0.1, 0.1]] * 10 + [[0.5, 0.1]] * 40)
        found, log = [], {}
        wall = WallContact([3, 7], [3, 7])
        teleop_loop(leader, follower, [0.1, 0.1], rate_hz=100.0, ramp_s=0.0, contact=wall,
                    send_feedback=lambda angles, velocity, contact, dt: found.append(contact.copy()),
                    current_signs=[-1.0, 1.0], log=log,
                    keep_running=lambda: len(follower.goals) < 50, clock=clock, sleep=clock.sleep)
        self.assertEqual(found[-1].tolist(), [1.0, 0.0])  # joint 0 held back going up
        self.assertAlmostEqual(wall.walls[0], 0.2)  # where the follower stopped
        self.assertTrue(all(f[0] == 0.0 for f in found[:10]))
        self.assertEqual({len(log[name]) for name in LOG_FIELDS}, {50})
        self.assertEqual(log["current"][-1][0], 300.0)  # joint-signed
        seen = []
        clock2 = Clock()
        follower2 = FakeFollower([-1.0, -1.0], [1.0, 1.0])
        teleop_loop(FakeLeader([[0.1, 0.1]]), follower2, [0.1, 0.1], rate_hz=100.0, ramp_s=0.0,
                    monitor=lambda t, current, error, contact: seen.append((t, current[0])),
                    keep_running=lambda: len(follower2.goals) < 3, clock=clock2, sleep=clock2.sleep)
        self.assertEqual([round(t, 2) for t, _ in seen], [0.0, 0.01, 0.02])

    def test_contact_waits_until_the_joint_has_reached_its_goal(self):
        contact = ContactDetector([3, 7], [7])
        zeros = np.zeros(2)
        for step in range(20):  # 0.2 s stuck far from the goal right after the ramp
            found = contact.update([0.5, 0.5], [0.0, 0.0], zeros, [300.0, 300.0], step * 0.01)
        self.assertEqual(list(found), [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, r"\[9\]"):
            ContactDetector([3, 7], [9])

    def test_grasp_only_joints_ignore_opening(self):
        zeros = np.zeros(2)

        def press(contact, error_sequence):
            out = []
            for step, error in enumerate(error_sequence):
                goal = np.array([error, error])
                out.append(contact.update(goal, zeros, zeros, zeros, step * 0.01))
            return np.array(out)

        # both joints flex into an object (negative error), then the leader backs off
        sequence = [0.0] * 3 + [-0.2] * 15 + [-0.01] * 3
        found = press(ContactDetector([4, 6], [4, 6], {4: -1}), sequence)
        self.assertEqual(found[17].tolist(), [-1.0, -1.0])
        self.assertEqual(found[18].tolist(), [0.0, 0.0])
        # lagging while opening (positive error) is never contact on a grasp-only joint
        opening = press(ContactDetector([4, 6], [4, 6], {4: -1}), [0.0] * 3 + [0.2] * 15)
        self.assertEqual(opening[:, 0].tolist(), [0.0] * 18)
        self.assertEqual(opening[-1, 1], 1.0)
        with self.assertRaisesRegex(ValueError, "must be"):
            ContactDetector([4], [4], {4: 2})

    def test_a_squeeze_is_caught_while_the_leader_keeps_moving(self):
        from scripts.teleop.trimi_teleop import BlockedContact

        grasp = BlockedContact(-1)
        # follower stopped by the object, leader still closing at 2 rad/s: contact after 20 ms
        out = [grasp.update(-0.06 - 0.02 * k, 0.0, 0.0, 0.01 * k) for k in range(4)]
        self.assertEqual(out, [0.0, 0.0, -1.0, -1.0])
        # closing fast in free air: the follower lags, but moves, so no contact
        free = BlockedContact(-1)
        self.assertEqual([free.update(-0.2, -3.0, -300.0, 0.01 * k) for k in range(10)], [0.0] * 10)
        # a stall shorter than 20 ms (the follower speeding up) is not contact either
        brief = BlockedContact(-1)
        speeds = [0.0, 0.3, 1.5, 3.0, 3.0]
        self.assertEqual([brief.update(-0.1, v, -300.0, 0.01 * k) for k, v in enumerate(speeds)], [0.0] * 5)
        # two-way: a side swap restarts the wait
        both = BlockedContact(0)
        errors = [0.1, 0.1, -0.1, -0.1, -0.1]
        self.assertEqual([both.update(e, 0.0, 0.0, 0.01 * k) for k, e in enumerate(errors)], [0.0] * 4 + [-1.0])

    def test_by_default_the_current_does_not_count(self):
        from scripts.teleop.trimi_teleop import BlockedContact

        contact = BlockedContact(-1)
        out = [contact.update(-0.2, 0.0, 0.0, 0.01 * k) for k in range(10)]
        self.assertEqual(out[-1], -1.0)  # held back without any current: still contact

    def test_with_a_current_threshold_contact_needs_both(self):
        from scripts.teleop.trimi_teleop import BlockedContact

        def run(error, current):
            contact = BlockedContact(-1, enter_current_ma=120.0)
            return [contact.update(error, 0.0, current, 0.01 * k) for k in range(10)]

        self.assertEqual(run(-0.2, -50.0), [0.0] * 10)  # lagging without pushing: a slow follower
        self.assertEqual(run(-0.02, -300.0), [0.0] * 10)  # pushing without lagging: friction, acceleration
        self.assertEqual(run(-0.2, 300.0), [0.0] * 10)  # pushing the other way
        self.assertEqual(run(-0.2, -300.0)[-1], -1.0)
        with self.assertRaisesRegex(ValueError, "release"):
            BlockedContact(-1, enter_rad=0.02, exit_rad=0.03)

    def test_the_detector_resets_below_half_the_contact_error(self):
        from scripts.teleop.trimi_teleop import BlockedContact

        contact = BlockedContact(-1)  # contact at 0.04 rad, reset below 0.02
        errors = [-0.041, -0.043, -0.046, -0.03, -0.019, -0.03]
        out = [contact.update(e, 0.0, 0.0, 0.01 * k) for k, e in enumerate(errors)]
        self.assertEqual(out, [0.0, 0.0, -1.0, -1.0, 0.0, 0.0])  # 0.03 alone does not re-enter

    def test_a_joint_can_have_its_own_contact_error_and_hold_time(self):
        contact = ContactDetector([0, 4], [0, 4], {4: -1}, enter_rad=0.02,
                                  motor_overrides={0: {"error_rad": 0.06, "hold_ms": 40}})
        self.assertEqual((contact.detectors[0].enter_rad, contact.detectors[0].persistence_s), (0.06, 0.04))
        self.assertEqual(contact.detectors[1].enter_rad, 0.02)  # the rest keep the common setting
        zeros = np.zeros(2)
        contact.update([0.0, 0.0], zeros, zeros, zeros, 0.0)  # arm
        found = []
        for step in range(1, 8):  # the leader runs 0.05 rad ahead of a still follower on both joints
            found.append(contact.update([0.05, -0.05], zeros, zeros, zeros, 0.01 * step))
        self.assertEqual(found[-1].tolist(), [0.0, -1.0])  # 0.05 is contact on the index PIP, not on motor 0
        with self.assertRaisesRegex(ValueError, "detect no contact"):
            ContactDetector([0, 4], [4], motor_overrides={0: {"error_rad": 0.06}})

    def test_a_gap_that_stays_put_is_not_contact(self):
        from scripts.teleop.trimi_teleop import BlockedContact

        # the follower stops 0.05 rad short of the leader and stays there: it falls short, nothing holds it
        steady = BlockedContact(-1)
        self.assertEqual([steady.update(-0.05, 0.0, 0.0, 0.01 * k) for k in range(30)], [0.0] * 30)
        # the same gap opening as the leader closes on a stopped follower is contact
        closing = BlockedContact(-1)
        out = [closing.update(-0.042 - 0.003 * k, 0.0, 0.0, 0.01 * k) for k in range(5)]
        self.assertEqual(out[-1], -1.0)
        # a slow follower still tracking at 0.4 rad/s is not held back
        slow = BlockedContact(-1)
        self.assertEqual([slow.update(-0.05 - 0.01 * k, -0.4, 0.0, 0.01 * k) for k in range(10)], [0.0] * 10)

    def test_a_follower_at_its_own_homing_stop_is_not_in_contact(self):
        from scripts.teleop.trimi_teleop import homing_stop_readings

        calibration = {"motor_ids": [2, 4], "joint_signs": {2: -1.0, 4: -1.0},
                       "homing": {"thumb": [[2, "Thumb joint 2", 2.1246, -1]], "fingers": [[4, "Index PIP", 0.03, -1]]}}
        stops = homing_stop_readings(calibration)
        self.assertAlmostEqual(stops[2][0], 2.1246)
        self.assertEqual(stops[2][1], 1)
        self.assertAlmostEqual(stops[4][0], 0.03)
        contact = ContactDetector([2, 4], [2, 4], stops=stops)
        zeros = np.zeros(2)
        for step in range(3):
            contact.update([2.1246, -0.5], [2.1246, -0.5], zeros, zeros, 0.01 * step)  # arm both
        # the leader's thumb sits 4 deg past the follower's stop, and the index PIP is held by an object
        for step in range(3, 10):
            found = contact.update([2.1946, -0.7], [2.1246, -0.5], zeros, zeros, 0.01 * step)
        self.assertEqual(found.tolist(), [0.0, -1.0])

    def test_arguments_are_checked(self):
        args = parse_args(["--no-tuning", "--leader-port", "/dev/a", "--follower-port", "/dev/b"])
        self.assertEqual(args.rate_hz, 100.0)
        self.assertIsNone(args.tuning)
        for bad in (["--rate-hz", "0"], ["--ramp-s", "-1"], ["--contact-error-rad", "0"],
                    ["--force-feedback", "--grasp-only", "1=2"], ["--force-feedback", "--grasp-only", "x"],
                    ["--follower-p-gain", "-5"]):
            with self.assertRaises(SystemExit):
                parse_args(["--no-tuning"] + bad)
        self.assertEqual(parse_args(["--no-tuning", "--force-feedback", "--grasp-only"]).grasp_only, {})
        self.assertEqual(parse_args(["--no-tuning", "--force-feedback", "--grasp-only", "1=-1", "2=+1"]).grasp_only,
                         {1: -1, 2: 1})
        self.assertEqual(parse_args(["--no-tuning", "--plot"]).plot, "error")
        self.assertEqual(parse_args(["--no-tuning", "--plot", "current"]).plot, "current")
        self.assertIsNone(parse_args(["--no-tuning"]).contact_current_ma)
        self.assertIsNone(parse_args(["--no-tuning", "--contact-current-ma", "off"]).contact_current_ma)
        self.assertEqual(parse_args(["--no-tuning", "--contact-current-ma", "150"]).contact_current_ma, 150.0)
        with self.assertRaises(SystemExit):
            parse_args(["--no-tuning", "--contact-current-ma", "-5"])
        self.assertEqual(parse_args(["--no-tuning", "--wall-motor-gains", "0:p=600,i=0", "3:d=100"]).wall_motor_gains,
                         {0: {"p": 600, "i": 0}, 3: {"d": 100}})
        for bad in ("0:q=5", "0:p=-1", "x:p=5", "0:p=20000"):
            with self.assertRaises(SystemExit):
                parse_args(["--no-tuning", "--wall-motor-gains", bad])
        self.assertEqual(parse_args(["--no-tuning", "--contact-motor-overrides", "0:error_rad=0.06,hold_ms=60"])
                         .contact_motor_overrides, {0: {"error_rad": 0.06, "hold_ms": 60.0}})
        for bad in ("0:error_rad=0", "0:hold_ms=-1", "0:p=5", "0:error_rad=x"):
            with self.assertRaises(SystemExit):
                parse_args(["--no-tuning", "--contact-motor-overrides", bad])


class TuningFileTests(unittest.TestCase):
    def test_the_file_sets_defaults_and_flags_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning.yaml"
            path.write_text("wall_current_ma: 600\ncontact-current-ma: 90\nforce_feedback: true\n"
                            "grasp_only: {1: -1}\nfollower_p_gain: 1200\nwall_motor_gains: {0: {p: 600, i: 0}}\n")
            args = parse_args(["--tuning", str(path), "--contact-current-ma", "150"])
            self.assertEqual(args.wall_current_ma, 600)
            self.assertEqual(args.contact_current_ma, 150.0)  # the flag wins
            self.assertTrue(args.force_feedback)
            self.assertEqual(args.grasp_only, {1: -1})
            self.assertEqual(args.follower_p_gain, 1200)
            self.assertEqual(args.wall_motor_gains, {0: {"p": 600, "i": 0}})
            self.assertEqual(args.tuning, path)

    def test_saved_settings_load_back_the_same(self):
        from scripts.teleop.trimi_teleop import TUNING_KEYS, save_tuning

        with tempfile.TemporaryDirectory() as tmp:
            saved = Path(tmp) / "saved.yaml"
            first = parse_args(["--no-tuning", "--force-feedback", "--grasp-only", "1=-1",
                                "--wall-current-ma", "800", "--plot", "current", "--follower-d-gain", "300"])
            save_tuning(saved, first)
            again = parse_args(["--tuning", str(saved)])
            for key in TUNING_KEYS:
                self.assertEqual(getattr(again, key), getattr(first, key), key)

    def test_unknown_or_missing_files_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("wall_current: 600\n")  # misspelt
            with self.assertRaises(SystemExit):
                parse_args(["--tuning", str(path)])
            with self.assertRaises(SystemExit):
                parse_args(["--tuning", str(Path(tmp) / "missing.yaml")])

    def test_the_checked_in_file_loads(self):
        from scripts.teleop.trimi_teleop import DEFAULT_TUNING

        self.assertTrue(DEFAULT_TUNING.is_file())
        args = parse_args([])  # the file is meant to be edited: only check that it is valid
        self.assertEqual(args.tuning, DEFAULT_TUNING)


class FakeClient:
    def __init__(self):
        self.calls = []

    def write_desired_pos(self, ids, raw):
        self.calls.append(("goal", tuple(ids), tuple(np.round(raw, 6))))

    def sync_write(self, ids, values, address, size):
        self.calls.append(("write", address, tuple(ids), tuple(values)))

    def set_torque_enabled(self, ids, enabled, **kwargs):
        self.calls.append(("torque", bool(enabled), tuple(ids)))


class FakeLeaderHand:
    class config:
        joint_signs_array = np.array([1.0, -1.0, 1.0])
        home_offsets_array = np.array([1.0, 2.0, 3.0])
        current_unit_ma = 1.0
        goal_current_limit = 1700
        max_current_limit_ma = 1750
        position_p_gain, position_i_gain, position_d_gain = 2000, 100, 400

    def __init__(self):
        self.dxl_client = FakeClient()

    def set_gains(self, motor_ids=None, p=None, i=None, d=None):
        self.dxl_client.calls.append(("gains", tuple(motor_ids), p, i, d))

    def set_motion_profile(self, profile_velocity_rad_s=None, profile_acceleration_rad_s2=None, motor_ids=None):
        self.dxl_client.calls.append(("profile", tuple(motor_ids), profile_velocity_rad_s))


class WallTests(unittest.TestCase):
    def engaged_wall(self):
        from scripts.teleop.trimi_teleop import WallContact

        wall = WallContact([4, 6], [4, 6], {4: -1})
        zeros = np.zeros(2)
        for k in range(3):  # arm: both joints on their goal
            wall.update([-0.5, 0.0], [-0.5, 0.0], zeros, zeros, 0.01 * k,
                        leader_positions=[-0.5, 0.0], leader_currents=zeros)
        k = 3
        # the follower's index PIP stops at -0.5 while the leader closes to -0.6
        while not wall.engaged[0]:
            wall.update([-0.6, 0.0], [-0.5, 0.0], zeros, zeros, 0.01 * k,
                        leader_positions=[-0.6, 0.0], leader_currents=zeros)
            k += 1
        return wall, k

    def test_the_leader_is_held_where_the_follower_stopped(self):
        wall, k = self.engaged_wall()
        self.assertLess(k, 8)  # within the 20 ms check
        np.testing.assert_allclose(wall.walls, [-0.5, np.nan])
        # held there: the follower's error is gone, the wall stays while the hand presses in
        zeros = np.zeros(2)
        for j in range(5):
            out = wall.update([-0.505, 0.0], [-0.5, 0.0], zeros, zeros, 0.01 * (k + j),
                              leader_positions=[-0.505, 0.0], leader_currents=[80.0, 0.0])
        self.assertEqual(out.tolist(), [-1.0, 0.0])

    def test_the_wall_stays_where_contact_began_unless_it_follows(self):
        zeros = np.zeros(2)
        for follow, expected in ((False, -0.5), (True, -0.52)):
            wall, k = self.engaged_wall()
            wall.follow = follow
            # pressing harder: the follower's own flex lets it creep 0.02 rad further in (within pass_rad)
            out = wall.update([-0.54, 0.0], [-0.52, 0.0], zeros, zeros, 0.01 * k,
                              leader_positions=[-0.54, 0.0], leader_currents=[200.0, 0.0])
            self.assertEqual(out[0], -1.0)  # still held: the leader is not out of the wall
            self.assertAlmostEqual(wall.walls[0], expected)

    def test_the_wall_lets_go_when_the_hand_opens_or_the_object_goes(self):
        zeros = np.zeros(2)
        cases = (
            ([-0.485, 0.0], [0.0, 0.0], zeros),  # the leader 0.015 rad out of the wall
            ([-0.5, 0.0], [-60.0, 0.0], zeros),  # its motor pulls the finger in: the hand is opening
            ([-0.5, 0.0], [0.0, 0.0], np.array([-1.0, 0.0])),  # the follower moves on: the object went
        )
        for leader_positions, leader_currents, velocities in cases:
            wall, k = self.engaged_wall()
            later = 0.01 * k + wall.settle_s  # past the settling time
            out = wall.update([-0.5, 0.0], [-0.5, 0.0], velocities, zeros, later,
                              leader_positions=leader_positions, leader_currents=leader_currents)
            self.assertEqual(out.tolist(), [0.0, 0.0])
            self.assertTrue(np.isnan(wall.walls[0]))

    def test_right_after_contact_the_wall_holds_while_the_leader_comes_back(self):
        zeros = np.zeros(2)
        wall, k = self.engaged_wall()
        now = 0.01 * k
        # brought back, the leader overshoots out of the wall and its motor pulls it in again
        out = wall.update([-0.5, 0.0], [-0.5, 0.0], zeros, zeros, now + 0.05,
                          leader_positions=[-0.485, 0.0], leader_currents=[-60.0, 0.0])
        self.assertEqual(out[0], -1.0)  # still settling: held
        out = wall.update([-0.5, 0.0], [-0.5, 0.0], np.array([-1.0, 0.0]), zeros, now + 0.06,
                          leader_positions=[-0.5, 0.0], leader_currents=zeros)
        self.assertEqual(out[0], 0.0)  # the follower moving on still ends it at once

    def test_a_wall_with_nothing_behind_it_lets_go_when_the_follower_goes_past(self):
        zeros = np.zeros(2)
        wall, k = self.engaged_wall()  # wall at -0.5
        # pressed on, the leader bends in and the follower slowly follows past the wall: no object there
        out = wall.update([-0.54, 0.0], [-0.535, 0.0], np.array([-0.1, 0.0]), zeros, 0.01 * k,
                          leader_positions=[-0.54, 0.0], leader_currents=[300.0, 0.0])
        self.assertEqual(out[0], 0.0)

    def finger_wall(self):
        """Index PIP 4, MCP 5, ABAD 6 and middle PIP 7 walled by finger; the index PIP stops on an object."""

        from scripts.teleop.trimi_teleop import FINGER_GROUPS, WallContact

        wall = WallContact([4, 5, 6, 7], [4, 5, 6, 7], {4: -1, 5: -1, 7: -1}, groups=FINGER_GROUPS)
        zeros = np.zeros(4)
        free = [-0.5, -0.3, 0.05, -0.4]
        for k in range(3):
            wall.update(free, free, zeros, zeros, 0.01 * k, leader_positions=free, leader_currents=zeros)
        k = 3
        # the follower PIP stops at -0.5; MCP and ABAD trail the leader by a little, the middle finger is free
        leader = [-0.6, -0.31, 0.05, -0.41]
        follower = [-0.5, -0.305, 0.052, -0.41]
        while not wall.engaged[0]:
            wall.update(leader, follower, zeros, zeros, 0.01 * k, leader_positions=leader, leader_currents=zeros)
            k += 1
        return wall, k, follower

    def test_contact_on_one_joint_walls_its_whole_finger(self):
        wall, k, follower = self.finger_wall()
        out = wall.update([-0.5, -0.305, 0.052, -0.41], follower, np.zeros(4), np.zeros(4), 0.01 * k,
                          leader_positions=[-0.5, -0.305, 0.052, -0.41], leader_currents=np.zeros(4))
        self.assertEqual(out.tolist(), [-1.0, 0.0, 0.0, 0.0])  # only the PIP is in contact
        self.assertEqual(wall.engaged.tolist(), [True, True, True, False])  # the index finger is held
        np.testing.assert_allclose(wall.walls[:3], follower[:3])  # each at its follower's position
        self.assertTrue(np.isnan(wall.walls[3]))

    def test_the_hand_opening_at_any_joint_of_the_finger_lets_it_all_go(self):
        zeros = np.zeros(4)
        held = [-0.5, -0.305, 0.052, -0.41]
        for leader_positions, leader_currents, opened in (
                ([-0.5, -0.29, 0.052, -0.41], zeros, True),  # the MCP (not in contact) comes 0.015 rad out
                ([-0.5, -0.305, 0.052, -0.41], [0.0, -60.0, 0.0, 0.0], True),  # or its motor pulls it in
                ([-0.5, -0.305, 0.08, -0.41], [0.0, 0.0, 300.0, 0.0], False)):  # ABAD has no opening side
            wall, k, follower = self.finger_wall()
            later = 0.01 * k + wall.settle_s
            out = wall.update(held, follower, zeros, zeros, later,
                              leader_positions=leader_positions, leader_currents=leader_currents)
            self.assertEqual(wall.engaged[:3].tolist(), [not opened] * 3)
            self.assertEqual(out[0], 0.0 if opened else -1.0)
            if opened:
                self.assertTrue(np.isnan(wall.walls[:3]).all())

    def test_joints_left_out_of_contact_detection_are_still_walled_with_their_finger(self):
        from scripts.teleop.trimi_teleop import FINGER_GROUPS, WallContact

        wall = WallContact([4, 5, 6], [4, 5, 6], {4: -1, 5: -1}, groups=FINGER_GROUPS, contact_motor_ids=[4, 5],
                           enter_rad=0.02)
        zeros = np.zeros(3)
        for k in range(3):
            wall.update([-0.5, -0.3, 0.0], [-0.5, -0.3, 0.0], zeros, zeros, 0.01 * k,
                        leader_positions=[-0.5, -0.3, 0.0], leader_currents=zeros)
        # the ABAD (6) is held back 0.1 rad: not watched, no wall
        for k in range(3, 10):
            out = wall.update([-0.5, -0.3, 0.1], [-0.5, -0.3, 0.0], zeros, zeros, 0.01 * k,
                              leader_positions=[-0.5, -0.3, 0.1], leader_currents=zeros)
        self.assertFalse(wall.engaged.any())
        # the PIP stops on an object: the whole finger is walled, ABAD included
        for k in range(10, 17):
            out = wall.update([-0.6, -0.3, 0.1], [-0.5, -0.3, 0.0], zeros, zeros, 0.01 * k,
                              leader_positions=[-0.6, -0.3, 0.1], leader_currents=zeros)
        self.assertEqual(out.tolist(), [-1.0, 0.0, 0.0])
        self.assertEqual(wall.engaged.tolist(), [True, True, True])
        with self.assertRaisesRegex(ValueError, "not among the wall motors"):
            WallContact([4, 5, 6], [4, 5], contact_motor_ids=[6])

    def test_a_finger_wall_comes_down_when_its_last_contact_ends(self):
        wall, k, follower = self.finger_wall()
        zeros = np.zeros(4)
        # the object goes: the follower PIP moves on past its wall
        moving = np.array([-1.0, 0.0, 0.0, 0.0])
        out = wall.update([-0.6, -0.305, 0.052, -0.41], [-0.55, -0.305, 0.052, -0.41], moving, zeros, 0.01 * k,
                          leader_positions=[-0.6, -0.305, 0.052, -0.41], leader_currents=zeros)
        self.assertEqual(out.tolist(), [0.0] * 4)
        self.assertFalse(wall.engaged.any())

    def test_leader_goal_is_written_before_the_torque_goes_on(self):
        from midas_hand_api.actuators import control_table as ct

        from scripts.teleop.trimi_teleop import LeaderWall

        hand = FakeLeaderHand()
        calls = hand.dxl_client.calls
        wall = LeaderWall(hand, [3, 5, 7], [5, 7], current_ma=700.0, gains={"p": 3000},
                          motor_gains={7: {"p": 600, "i": 0}})
        wall.start()
        self.assertEqual(calls[0], ("torque", False, (5, 7)))
        self.assertIn(("write", ct.ADDR_OPERATING_MODE, (5, 7), (5, 5)), calls)
        self.assertIn(("write", ct.ADDR_GOAL_CURRENT, (5, 7), (700, 700)), calls)
        gains = [c for c in calls if c[0] == "gains"]
        self.assertEqual(gains, [("gains", (5, 7), 3000, None, None), ("gains", (7,), 600, 0, None)])  # then motor 7's
        with self.assertRaisesRegex(ValueError, "no wall"):
            LeaderWall(hand, [3, 5, 7], [5, 7], current_ma=700.0, motor_gains={3: {"p": 600}})
        calls.clear()
        nan = float("nan")
        wall.update([False, True, False], [nan, -0.5, nan])  # motor 5: sign -1, offset 2.0 -> raw 2.5
        self.assertEqual(calls, [("goal", (5,), (2.5,)), ("write", ct.ADDR_TORQUE_ENABLE, (5,), (1,))])
        calls.clear()
        wall.update([False, True, False], [nan, -0.52, nan])  # the wall moves with the follower
        self.assertEqual(calls, [("goal", (5,), (2.52,))])
        calls.clear()
        wall.update([False, False, False], [nan, nan, nan])
        self.assertEqual(calls, [("write", ct.ADDR_TORQUE_ENABLE, (5,), (0,))])
        calls.clear()
        wall.shutdown()
        self.assertEqual(calls[0], ("torque", False, (5, 7)))
        self.assertIn(("write", ct.ADDR_GOAL_CURRENT, (5, 7), (1700, 1700)), calls)
        self.assertIn(("gains", (5, 7), 2000, 100, 400), calls)


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
