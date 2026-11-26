"""Logging regression checks: mock devices only, no serial/CAN/camera access."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from inference.executor.executor import (
    FakeCamera, FakeRobot, HandCommandWriter, RunLog, capture, filter_action, parse_args,
)
from inference.predictor.predictor import policy_provenance


class ExecutorLoggingTests(unittest.TestCase):
    def test_save_keeps_all_hand_stages_clipping_observations_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(['--predictor', 'localhost', '--fake'])
            log = RunLog(directory)
            log.configure(args)
            log.hand_limits = (np.full(13, -1.0), np.full(13, 1.0))
            robot, camera = FakeRobot(), FakeCamera()
            camera.last_policy_frame_received_ns = 123
            capture(robot, camera, 7, log)
            state = np.zeros(19)
            state[6:] = .25
            chunk = np.zeros((3, 19))
            chunk[:, 6:] = 1.5
            log.chunk(7, state, chunk)
            # Mutating the caller's arrays must not corrupt archived policy actions.
            chunk[:] = 99
            policy = log.chunks[-1][3][2]
            with patch('inference.executor.executor.HAND_ACTION_SCALE', 2.0):
                scaled, filtered = filter_action(policy, state, np.full(19, .5), state[6:])
            log.command(7, 2, scaled, filtered)
            log.note_header({'policy': {'sha256': 'checkpoint-digest', 'hand_conditioned': True}})
            log.save(args)
            meta = json.loads((log.directory / 'meta.json').read_text())
            self.assertTrue(meta['policy']['hand_conditioned'])
            self.assertIn('hand_action_ema_alpha', meta)
            self.assertIn('hand_action_scale', meta)
            self.assertIn('hand_joint_action_scales', meta)
            self.assertEqual(len(meta['effective_hand_action_scales']), 13)
            self.assertEqual(meta['log_schema_version'], 2)
            self.assertTrue(meta['source_sha256'])
            with (log.directory / 'hand_commands.csv').open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual((row['cycle'], row['index']), ('7', '2'))
            self.assertEqual(float(row['policy_7_rad']), 1.5)
            self.assertEqual(float(row['reference_7_rad']), .25)
            self.assertEqual(float(row['scaled_7_rad']), 2.75)
            self.assertEqual(float(row['filtered_7_rad']), 1.5)
            self.assertEqual(float(row['clipped_7_rad']), 1.0)
            self.assertEqual(row['clipped_motor_ids'], ' '.join(map(str, range(13))))
            with (log.directory / 'observations.csv').open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['cycle'], '7')
            self.assertEqual(row['frame_received_monotonic_ns'], '123')
            self.assertEqual(row['measured_x'], '')  # Fake state is not real arm feedback.
            with np.load(log.directory / 'chunks.npz') as chunks:
                np.testing.assert_allclose(chunks['actions'][2, 6:], 1.5)

    def test_successful_hand_writes_logged_without_additional_default_reads(self):
        hand = Mock()
        hand.clip_positions.side_effect = lambda values: np.clip(values, -1, 1)
        with patch('inference.executor.executor.threading.Thread'):
            writer = HandCommandWriter(hand, 100)
        writer.run_log = RunLog('/unused')
        writer.submit(np.full(13, 2.0))
        hand.set_positions.side_effect = lambda *a, **k: writer.stop.set()
        writer.run()
        self.assertIsNone(writer.error)
        row = writer.run_log.hand_writes[0]
        self.assertGreaterEqual(row['finished_monotonic_ns'], row['started_monotonic_ns'])
        self.assertEqual(row['target_8_rad'], 1.0)
        # The existing write retains its target and clip=True behavior.
        np.testing.assert_allclose(hand.set_positions.call_args.args[0], 2.0)
        self.assertTrue(hand.set_positions.call_args.kwargs['clip'])
        hand.read_pos.assert_not_called()
        hand.read_pos_vel_cur.assert_not_called()

    def test_optional_telemetry_failure_is_logged_without_stopping_writes(self):
        hand = Mock()
        hand.clip_positions.side_effect = lambda values: values
        with patch('inference.executor.executor.threading.Thread'):
            writer = HandCommandWriter(hand, 100)
        writer.run_log = RunLog('/unused')
        writer.telemetry_period = .1
        writer.submit(np.zeros(13))
        hand.set_positions.side_effect = lambda *a, **k: writer.stop.set()
        hand.read_pos_vel_cur.side_effect = OSError('diagnostic timeout')
        writer.run()
        self.assertIsNone(writer.error)
        row = writer.run_log.hand_telemetry[0]
        self.assertIn('diagnostic timeout', row['error'])
        self.assertNotIn('position_0_rad', row)

    def test_optional_telemetry_keeps_motor_order_units_and_read_interval(self):
        hand = Mock()
        hand.clip_positions.side_effect = lambda values: values
        hand.read_pos_vel_cur.return_value = (np.arange(13) / 10, np.arange(13) / 20, np.arange(13) * 10)
        with patch('inference.executor.executor.threading.Thread'):
            writer = HandCommandWriter(hand, 100)
        writer.run_log = RunLog('/unused')
        writer.telemetry_period = .1
        writer.submit(np.zeros(13))
        hand.set_positions.side_effect = lambda *a, **k: writer.stop.set()
        writer.run()
        row = writer.run_log.hand_telemetry[0]
        self.assertEqual(row['error'], '')
        self.assertEqual(row['position_7_rad'], .7)
        self.assertEqual(row['velocity_9_rad_s'], .45)
        self.assertEqual(row['current_12_ma'], 120)
        self.assertGreaterEqual(row['finished_monotonic_ns'], row['started_monotonic_ns'])

    def test_bind_robot_uses_cached_config_and_enables_requested_telemetry(self):
        args = parse_args(['--predictor', 'localhost', '--hand-telemetry-hz', '10'])
        hand = Mock()
        hand.motor_ids = list(range(13))
        hand.config = SimpleNamespace(joint_lower_limits_array=np.full(13, -1.),
                                      joint_upper_limits_array=np.full(13, 1.),
                                      home_offsets_array=np.arange(13), joint_signs_array=np.ones(13))
        with patch('inference.executor.executor.threading.Thread'):
            writer = HandCommandWriter(hand, 100)
        log = RunLog('/unused')
        log.configure(args, SimpleNamespace(hand=hand, hand_writer=writer))
        self.assertIs(writer.run_log, log)
        self.assertEqual(writer.telemetry_period, .1)
        self.assertEqual(log.hand_configuration['home_offsets_rad'][7], 7)
        hand.read_pos.assert_not_called()
        hand.read_pos_vel_cur.assert_not_called()
        hand.set_positions.assert_not_called()

    def test_telemetry_requires_explicit_rate_and_logging(self):
        self.assertEqual(parse_args(['--predictor', 'localhost']).hand_telemetry_hz, 0)
        self.assertEqual(parse_args(['--predictor', 'localhost', '--hand-telemetry-hz', '10']).hand_telemetry_hz, 10)
        for extra in (['--hand-telemetry-hz', '-1'], ['--hand-telemetry-hz', 'nan'],
                      ['--hand-telemetry-hz', '51'], ['--hand-telemetry-hz', '10', '--no-log'],
                      ['--hand-telemetry-hz', '10', '--fake']):
            with self.subTest(extra=extra), patch('sys.stderr'), self.assertRaises(SystemExit):
                parse_args(['--predictor', 'localhost', *extra])

    def test_predictor_provenance_exposes_actual_observation_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'policy.ckpt'
            checkpoint.write_bytes(b'not loaded; hash only')
            args = SimpleNamespace(checkpoint=str(checkpoint), no_ema=False, action_mode='chunk')
            policy = SimpleNamespace(policy=SimpleNamespace(num_inference_steps=16), num_obs_steps=1,
                                     hand_pose_key='hand_pose_abs', shape_meta={'obs': {
                                         'camera_0': {'shape': [3, 240, 320], 'type': 'rgb'},
                                         'hand_pose_abs': {'shape': [13], 'type': 'low_dim'}}})
            provenance = policy_provenance(args, policy)
            self.assertTrue(provenance['hand_conditioned'])
            self.assertEqual(provenance['observations']['hand_pose_abs']['shape'], [13])
            self.assertEqual(len(provenance['sha256']), 64)


if __name__ == '__main__':
    unittest.main()
