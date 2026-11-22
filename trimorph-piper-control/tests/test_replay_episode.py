import csv
import io
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from dataclasses import make_dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from data_collection.collection.thumb_retargeting import ThumbRetargetModel
from data_collection.replay_episode import (
    ACTION_SCALE,
    CURRENT_TELEMETRY_HZ,
    JOINT_ACTION_SCALES,
    REPLAY_HZ,
    ReplayCurrentRecorder,
    LocalIKReplay,
    advance_current_sample_schedule,
    build_replay_config,
    build_replay_hand_config,
    config_values_for_motor_ids,
    decode_piper_error_code,
    load_hand_stream,
    load_piper_stream,
    log_piper_diagnostics,
    move_to_first_pose,
    replay_interval,
    replay_piper_commands,
    resolve_leader_hand_path,
    sample_stream,
    scale_hand_stream,
    main,
    parse_args,
    output_current_csv,
    offset_piper_z,
    offset_piper_position,
    prepare_hand_stream_for_replay,
)
from scripts.trimorph_api_test.gen_graphs_from_csvs import _load_telemetry


class ReplayEpisodeTests(unittest.TestCase):
    def local_ik_args(self, **overrides):
        values = dict(arm_control="joint", joint_level=False, piper_rate_hz=60., ik_rate_hz=15.,
                      max_joint_speed_deg_s=(45., 45., 90., 60., 45., 60.), ik_damping_sigma=.03,
                      arm_diag=False, playback_speed=.5, no_preposition=False,
                      preposition_seconds=.1, preposition_speed_percent=20, settle_seconds=0.)
        return SimpleNamespace(**(values | overrides))

    def test_local_ik_mode_defaults_and_invalid_options(self):
        with patch('sys.argv', ['replay_episode.py', '--arm-control', 'joint']):
            args = parse_args()
        self.assertEqual(args.piper_rate_hz, 60.)
        self.assertEqual(args.ik_rate_hz, 15.)
        for flags, message in ((['--joint-level'], 'omit --joint-level'),
                               (['--ik-rate-hz', '61'], '--ik-rate-hz'),
                               (['--ik-rate-hz', 'nan'], '--ik-rate-hz'),
                               (['--ik-damping-sigma', '-1'], '--ik-damping-sigma'),
                               (['--max-joint-speed-deg-s', *(['nan']*6)], '--max-joint-speed-deg-s')):
            with self.subTest(flags=flags), patch('sys.argv', ['replay_episode.py', '--arm-control', 'joint', *flags]):
                with self.assertRaisesRegex(ValueError, message):
                    main()

    def test_local_ik_interpolates_and_respects_bounds_and_wall_time_speed_caps(self):
        api = MagicMock()
        seed = np.radians([-35., 136., -80., 5., -62., 95.])
        api.get_joint_angles.return_value = seed.tolist()
        args = self.local_ik_args()
        control = LocalIKReplay(api, args)
        pose = control.controller.fk(seed).copy()
        pose[:3] += [.05, 0., .1]
        previous = seed
        for wall_time in np.arange(16)/60:
            control.command(api, pose, wall_time)
            sent = np.array(api.move_joint.call_args.args[0])
            self.assertTrue(np.all(abs(sent-previous) <= np.radians(args.max_joint_speed_deg_s)/60 + 1e-8))
            self.assertTrue(np.all(sent >= control.controller.lower-1e-9))
            self.assertTrue(np.all(sent <= control.controller.upper+1e-9))
            self.assertLess(sent[4], 0.)  # preserve the demonstrated wrist side
            previous = sent
        self.assertGreater(np.linalg.norm(previous-seed), .001)
        self.assertLess(np.linalg.norm(control.controller.fk(previous)[:3]-pose[:3]),
                        np.linalg.norm(control.controller.fk(seed)[:3]-pose[:3]))
        self.assertEqual(api.move_joint.call_args.kwargs, {'mode': 'smooth'})
        api.move_eef.assert_not_called()

    def test_local_ik_replay_and_preposition_never_send_cartesian_commands(self):
        clock = SimpleNamespace(now=0.)
        class FakeStop:
            def is_set(self):
                return False
            def wait(self, seconds):
                clock.now += seconds
                return False
        api = MagicMock()
        seed = np.radians([-35., 136., -80., 5., -62., 95.])
        api.get_joint_angles.return_value = seed.tolist()
        from inference.executor.joint_control import JointController
        fk = JointController().fk
        stream = (np.array([0., .1]), np.array([fk(seed), fk(seed+np.radians([1., 0., 0., 0., 0., 0.]))]))
        args = self.local_ik_args()
        with patch('data_collection.replay_episode.time.monotonic', side_effect=lambda: clock.now):
            replay_piper_commands(api, stream, 0., .1, 0., args, FakeStop())
        self.assertAlmostEqual(clock.now, .2)  # half-speed playback, caps use wall time
        self.assertGreater(api.move_joint.call_count, 5)
        api.move_eef.assert_not_called()
        api.reset_mock()
        hand = MagicMock()
        def sleep(seconds):
            clock.now += seconds
        with (patch('data_collection.replay_episode.time.monotonic', side_effect=lambda: clock.now),
              patch('data_collection.replay_episode.time.sleep', side_effect=sleep)):
            move_to_first_pose(hand, api, (stream[0], np.zeros((2, 13))), stream, 0., args)
        api.get_joint_angles.assert_called_once()
        self.assertGreater(api.move_joint.call_count, 0)
        hand.set_positions.assert_called()
        api.move_eef.assert_not_called()

    def test_local_ik_requires_valid_seed_inside_policy_margins(self):
        api = MagicMock()
        for measured, exception in ((None, RuntimeError), ([0.]*5, ValueError),
                                    ([float('nan')]*6, ValueError),
                                    (np.radians([-35.,136.,-80.,5.,-70.,95.]), ValueError)):
            with self.subTest(measured=measured):
                api.get_joint_angles.return_value = measured
                with self.assertRaises(exception):
                    LocalIKReplay(api, self.local_ik_args())
        api.move_joint.assert_not_called()
        api.move_eef.assert_not_called()

    def test_local_ik_main_replays_processed_eef_without_recorded_joints(self):
        import json
        from inference.executor.joint_control import JointController
        seed = np.radians([-35., 136., -80., 5., -62., 95.])
        pose = JointController().fk(seed)
        api = MagicMock()
        api.get_joint_angles.return_value = seed.tolist()
        api.is_ok.return_value = True
        api.enable.return_value = True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw' / 'episode_000002'
            processed = root / 'processed' / raw.name
            raw.mkdir(parents=True)
            processed.mkdir(parents=True)
            (raw / 'leader_hand.csv').write_text('elapsed_s,pos_0\n0,0\n.03,0\n')
            (processed / 'metadata.json').write_text(json.dumps({'pose_frame': 'piper_base'}))
            with (processed / 'piper.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['elapsed_s', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])
                writer.writerow([0., *pose])
                writer.writerow([.03, *pose])
            with (patch('sys.argv', ['replay_episode.py', '2', '--dataset-root', directory,
                                     '--arm-only', '--no-preposition', '--arm-control', 'joint']),
                  patch('scripts.piper_api.piper_api.PiperAPI', return_value=api)):
                self.assertEqual(main(), 0)
        api.set_position_mode.assert_called_once()
        self.assertGreater(api.move_joint.call_count, 0)
        api.move_eef.assert_not_called()
        api.get_eef_pose.assert_not_called()
        api.disconnect.assert_called_once()

    def test_cartesian_feedback_wrap_interpolates_continuously(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'piper.csv'
            path.write_text('elapsed_s,x,y,z,roll,pitch,yaw\n'
                            '0,.4,-.2,.3,3.13,.2,3.12\n'
                            '1,.42,-.2,.3,-3.13,.3,-3.12\n')
            before = path.read_bytes()
            times, poses = load_piper_stream(path)
            target = sample_stream(times, poses, .5)
            np.testing.assert_allclose(target, [.41, -.2, .3, np.pi, .25, np.pi])
            self.assertEqual(path.read_bytes(), before)

    def test_combined_y_z_offsets_preserve_other_coordinates_and_source(self):
        times = np.array([0., 1.])
        poses = np.array([[.4, -.3, .06, 1., 2., 3.], [.5, -.2, .08, 2., 3., 4.]])
        original = poses.copy()
        for y_mm, z_mm in ((-5.4, 0.), (-5.4, 15.5), (5.4, -15.5)):
            with self.subTest(y_mm=y_mm, z_mm=z_mm):
                shifted_times, shifted = offset_piper_position((times, poses), y_mm=y_mm, z_mm=z_mm)
                np.testing.assert_allclose(shifted[:, 1], original[:, 1]+y_mm/1000)
                np.testing.assert_allclose(shifted[:, 2], original[:, 2]+z_mm/1000)
                np.testing.assert_array_equal(shifted[:, [0, 3, 4, 5]], original[:, [0, 3, 4, 5]])
                np.testing.assert_array_equal(poses, original)
                np.testing.assert_array_equal(shifted_times, times)

    def test_replay_y_offset_cli_and_invalid_modes(self):
        with patch('sys.argv', ['replay_episode.py']):
            self.assertEqual(parse_args().piper_y_offset_mm, 0.)
        with patch('sys.argv', ['replay_episode.py', '--piper-y-offset-mm', '-5.4']):
            self.assertEqual(parse_args().piper_y_offset_mm, -5.4)
        for flags in (['--piper-y-offset-mm', 'nan'], ['--piper-y-offset-mm', 'inf'],
                      ['--piper-y-offset-mm', '-5.4', '--hand-only'],
                      ['--piper-y-offset-mm', '-5.4', '--joint-level', '--read-piper-raw']):
            with self.subTest(flags=flags), patch('sys.argv', ['replay_episode.py', *flags]), self.assertRaisesRegex(ValueError, '--piper-y-offset-mm'):
                main()

    def test_replay_z_offset_shifts_only_z_and_preserves_episode(self):
        times = np.array([0., 1.])
        poses = np.array([[.4, -.3, .06, 1., 2., 3.], [.5, -.2, .08, 2., 3., 4.]])
        original = poses.copy()
        for offset_mm in (15.5, -10., 0.):
            with self.subTest(offset_mm=offset_mm):
                shifted_times, shifted = offset_piper_z((times, poses), offset_mm)
                np.testing.assert_allclose(shifted[:, 2], original[:, 2] + offset_mm / 1000)
                np.testing.assert_array_equal(shifted[:, [0, 1, 3, 4, 5]], original[:, [0, 1, 3, 4, 5]])
                np.testing.assert_array_equal(poses, original)
                np.testing.assert_array_equal(shifted_times, times)

    def test_replay_z_offset_cli_and_invalid_modes(self):
        with patch('sys.argv', ['replay_episode.py']):
            self.assertEqual(parse_args().piper_z_offset_mm, 0.)
        with patch('sys.argv', ['replay_episode.py', '--piper-z-offset-mm', '15.5']):
            self.assertEqual(parse_args().piper_z_offset_mm, 15.5)
        for flags in (['--piper-z-offset-mm', 'nan'], ['--piper-z-offset-mm', 'inf'],
                      ['--piper-z-offset-mm', '15.5', '--hand-only'],
                      ['--piper-z-offset-mm', '15.5', '--joint-level', '--read-piper-raw']):
            with self.subTest(flags=flags), patch('sys.argv', ['replay_episode.py', *flags]), self.assertRaisesRegex(ValueError, '--piper-z-offset-mm'):
                main()

    def test_position_log_includes_measured_arm_and_hand_with_timestamps(self):
        hand = MagicMock()
        hand.read_pos_vel_cur.return_value = (np.array([0.2]), np.array([0.3]), np.array([12.0]))
        piper = MagicMock()
        piper.get_joint_state.return_value = SimpleNamespace(joints=[0.1] * 6, timestamp=10.0)
        piper.get_eef_state.return_value = SimpleNamespace(pose=[0.4, -0.2, 0.1, 3.0, 0.0, 1.0], timestamp=10.01)
        recorder = ReplayCurrentRecorder((0,), piper=piper)
        recorder.sample(hand, np.array([0.25]), elapsed_s=0.0)
        piper.get_joint_state.return_value = None
        piper.get_eef_state.return_value = None
        recorder.sample(hand, np.array([0.26]), elapsed_s=0.1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'positions.csv'
            output_current_csv(recorder, output)
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(float(rows[0]['piper_joint_6_rad']), 0.1)
        self.assertEqual(float(rows[0]['piper_x_m']), 0.4)
        self.assertEqual(float(rows[0]['piper_yaw_rad']), 1.0)
        self.assertEqual(float(rows[0]['motor_0_actual_position_rad']), 0.2)
        self.assertEqual(float(rows[0]['motor_0_target_position_rad']), 0.25)
        self.assertEqual(float(rows[0]['piper_joint_timestamp_s']), 10.0)
        self.assertGreater(int(rows[0]['sample_monotonic_ns']), 0)
        self.assertEqual(rows[1]['piper_joint_6_rad'], '')
        self.assertEqual(rows[1]['piper_x_m'], '')

    def test_output_positions_alias(self):
        with patch('sys.argv', ['replay_episode.py', '--output-positions', 'positions.csv']):
            self.assertEqual(parse_args().output_current, Path('positions.csv'))

    def test_loop_cli(self):
        for flags, expected in (([], 1), (["--loop"], None), (["--loop", "3"], 3)):
            with self.subTest(flags=flags), patch("sys.argv", ["replay_episode.py", *flags]):
                self.assertEqual(parse_args().loop, expected)
        for value in ("0", "-1", "1.5", "abc"):
            with self.subTest(value=value), patch("sys.argv", ["replay_episode.py", "--loop", value]), \
                    redirect_stdout(io.StringIO()), patch("sys.stderr", new_callable=io.StringIO), \
                    self.assertRaises(SystemExit):
                parse_args()

    def test_loop_replays_hand_and_cleans_up_on_interrupt(self):
        cases = (
            ([], 1, 0, False),
            (["--loop", "3"], 3, 0, False),
            (["--loop"], 3, 130, False),
            (["--loop"], 3, 130, True),
            (["--loop", "5"], 3, 130, True),
        )
        for flags, plays, exit_code, interrupt_during_replay in cases:
            with self.subTest(flags=flags, interrupt_during_replay=interrupt_during_replay), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                episode = Path(directory) / "raw" / "episode_000000"
                episode.mkdir(parents=True)
                (episode / "leader_hand.csv").write_text("elapsed_s,pos_0\n0,0\n0.2,0.5\n")
                hand = MagicMock()
                hand.motor_ids = (0,)
                hand.read_pos.return_value = np.array([0.0])
                hand.clip_positions.side_effect = lambda target: target
                hand.read_pos_vel_cur.return_value = (np.zeros(1), np.zeros(1), np.zeros(1))
                constructor = MagicMock(return_value=hand)
                stack.enter_context(patch.dict("sys.modules", {"midas_hand_api": SimpleNamespace(MidasHand=constructor)}))
                stack.enter_context(patch("data_collection.replay_episode.build_replay_hand_config"))
                stack.enter_context(patch("data_collection.replay_episode.verify_connected_hand"))
                preposition = stack.enter_context(patch("data_collection.replay_episode.move_to_first_pose"))
                if exit_code == 130 and not interrupt_during_replay:
                    preposition.side_effect = [None, None, None, KeyboardInterrupt]
                if interrupt_during_replay:
                    original_sample = ReplayCurrentRecorder.sample

                    def sample_then_interrupt(recorder, *args, **kwargs):
                        original_sample(recorder, *args, **kwargs)
                        if preposition.call_count == plays:
                            raise KeyboardInterrupt

                    stack.enter_context(patch.object(
                        ReplayCurrentRecorder, "sample", autospec=True,
                        side_effect=sample_then_interrupt,
                    ))
                videos = [MagicMock() for _ in range(plays)]
                mocaps = [MagicMock() for _ in range(plays)]
                video_factory = stack.enter_context(patch("data_collection.replay_episode.ReplayVideoRecorder", side_effect=videos))
                mocap_factory = stack.enter_context(patch("data_collection.replay_episode.ReplayMocapRecorder", side_effect=mocaps))
                clock = iter(index * 0.05 for index in range(1000))
                stack.enter_context(patch("data_collection.replay_episode.time.monotonic", side_effect=lambda: next(clock)))
                stack.enter_context(patch("data_collection.replay_episode.time.sleep"))
                output = Path(directory) / "current.csv"
                stack.enter_context(patch("sys.argv", [
                    "replay_episode.py", "--dataset-root", directory, "--hand-only",
                    "--output-current", str(output), "--output-video", str(Path(directory) / "video.mp4"),
                    "--use-mocap", "--mocap-output", str(Path(directory) / "mocap.csv"), *flags,
                ]))
                stack.enter_context(redirect_stdout(io.StringIO()))
                self.assertEqual(main(), exit_code)
                self.assertEqual(preposition.call_count, plays + (exit_code == 130 and not interrupt_during_replay))
                self.assertEqual(video_factory.call_count, plays)
                self.assertEqual(mocap_factory.call_count, plays)
                for recorder in videos + mocaps:
                    recorder.start.assert_called_once()
                    recorder.stop.assert_called_once()
                constructor.assert_called_once()
                hand.shutdown.assert_called_once()
                paths = sorted(Path(directory).glob("current*.csv"))
                self.assertEqual(len(paths), plays)
                if flags:
                    self.assertEqual(paths[0].name, "current_loop_0001.csv")
                for path in paths:
                    with path.open() as stream:
                        times = [float(row["elapsed_s"]) for row in csv.DictReader(stream)]
                    self.assertTrue(times)
                    self.assertTrue(all(b > a for a, b in zip(times, times[1:])))
                    # The graph tool must accept both complete and interrupted CSVs.
                    _, graph_times, graph_motors = _load_telemetry(path)
                    np.testing.assert_allclose(graph_times, times)
                    self.assertEqual(set(graph_motors), {0})
                    if interrupt_during_replay and path == paths[-1]:
                        self.assertLess(times[-1], 0.2)
                    else:
                        self.assertAlmostEqual(times[-1], 0.2)

    def test_loop_restarts_arm_replay_and_prepositions_each_time(self):
        with ExitStack() as stack:
            stack.enter_context(patch("sys.argv", ["replay_episode.py", "--arm-only", "--loop", "3"]))
            stack.enter_context(patch("data_collection.replay_episode.resolve_episode", return_value=Path("episode_000000")))
            stack.enter_context(patch("data_collection.replay_episode.resolve_piper_source", return_value=(Path("piper.csv"), "piper_base")))
            stream = (np.array([0.0, 0.2]), np.zeros((2, 6)))
            stack.enter_context(patch("data_collection.replay_episode.load_piper_stream", return_value=stream))
            api = MagicMock()
            constructor = MagicMock(return_value=api)
            stack.enter_context(patch.dict("sys.modules", {"scripts.piper_api.piper_api": SimpleNamespace(PiperAPI=constructor)}))
            preposition = stack.enter_context(patch("data_collection.replay_episode.move_to_first_pose"))
            replay = stack.enter_context(patch("data_collection.replay_episode.replay_piper_commands"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(main(), 0)
            self.assertEqual(preposition.call_count, 3)
            self.assertEqual(replay.call_count, 3)
            starts = [call.args[4] for call in replay.call_args_list]
            self.assertTrue(all(b > a for a, b in zip(starts, starts[1:])))
            constructor.assert_called_once()
            api.disconnect.assert_called_once()

    def test_piper_replay_rate_is_independent_of_hand_rate(self):
        with patch("sys.argv", ["replay_episode.py"]):
            args = parse_args()
        self.assertEqual(args.rate_hz, REPLAY_HZ)
        self.assertEqual(args.piper_rate_hz, 250.0)
        with patch(
            "sys.argv",
            ["replay_episode.py", "--rate-hz", "60", "--arm-rate-hz", "125"],
        ):
            args = parse_args()
        self.assertEqual(args.rate_hz, 60.0)
        self.assertEqual(args.piper_rate_hz, 125.0)

    def test_piper_replay_interpolates_on_its_own_250_hz_clock(self):
        clock = SimpleNamespace(now=0.0)

        class FakeStop:
            def is_set(self):
                return False

            def wait(self, seconds):
                clock.now += seconds
                return False

        class FakeAPI:
            def __init__(self):
                self.targets = []

            def move_eef(self, target):
                self.targets.append(target.copy())

        api = FakeAPI()
        args = SimpleNamespace(
            piper_rate_hz=250.0,
            joint_level=False,
            arm_diag=False,
        )
        stream = (
            np.asarray([0.0, 0.008]),
            np.asarray([[0.0] * 6, [8.0] * 6]),
        )
        with patch("data_collection.replay_episode.time.monotonic", side_effect=lambda: clock.now):
            replay_piper_commands(api, stream, 0.0, 0.008, 0.0, args, FakeStop())

        np.testing.assert_allclose(api.targets, [[0.0] * 6, [4.0] * 6, [8.0] * 6])

    def test_piper_replay_playback_speed_stretches_the_episode_clock(self):
        clock = SimpleNamespace(now=0.0)

        class FakeStop:
            def is_set(self):
                return False

            def wait(self, seconds):
                clock.now += seconds
                return False

        class FakeAPI:
            def __init__(self):
                self.targets = []

            def move_eef(self, target):
                self.targets.append(target.copy())

        api = FakeAPI()
        args = SimpleNamespace(piper_rate_hz=250.0, joint_level=False, arm_diag=False, playback_speed=0.5)
        stream = (np.asarray([0.0, 0.008]), np.asarray([[0.0] * 6, [8.0] * 6]))
        with patch("data_collection.replay_episode.time.monotonic", side_effect=lambda: clock.now):
            replay_piper_commands(api, stream, 0.0, 0.008, 0.0, args, FakeStop())

        # Half speed: the 8 ms episode takes 16 ms, so 250 Hz ticks land every 2 ms of episode time.
        np.testing.assert_allclose(api.targets, [[0.0] * 6, [2.0] * 6, [4.0] * 6, [6.0] * 6, [8.0] * 6])
        self.assertAlmostEqual(clock.now, 0.016)

    def test_playback_speed_cli_defaults_to_recorded_speed_and_is_validated(self):
        with patch("sys.argv", ["replay_episode.py"]):
            self.assertEqual(parse_args().playback_speed, 1.0)
        for value in ("0", "-0.5", "nan", "inf", "-inf"):
            with self.subTest(value=value), patch("sys.argv", ["replay_episode.py", f"--playback-speed={value}"]), \
                    self.assertRaisesRegex(ValueError, "--playback-speed"):
                main()

    def test_replay_speed_above_one_passes_validation(self):
        for option in ("--replay-speed", "--playback-speed"):
            with self.subTest(option=option), patch("sys.argv", ["replay_episode.py", option, "2"]), \
                    patch("data_collection.replay_episode.resolve_episode", side_effect=RuntimeError("validated speed")), \
                    self.assertRaisesRegex(RuntimeError, "validated speed"):
                main()

    def test_action_scale_is_a_module_constant(self):
        self.assertEqual(ACTION_SCALE, 1.0)
        self.assertEqual(set(JOINT_ACTION_SCALES), set(range(13)))
        self.assertEqual(JOINT_ACTION_SCALES[0], 1.1)
        self.assertEqual(JOINT_ACTION_SCALES[1], 1.1)
        self.assertEqual(JOINT_ACTION_SCALES[4], 1.1)
        self.assertEqual(JOINT_ACTION_SCALES[5], 1.1)
        self.assertEqual(JOINT_ACTION_SCALES[7], 1.1)
        self.assertEqual(JOINT_ACTION_SCALES[8], 1.1)

    def test_replay_uses_shared_runtime_config(self):
        with patch(
            "sys.argv",
            [
                "replay_episode.py",
                "--dataset-root",
                "dataset",
                "--hand-config",
                "midas.yaml",
                "--hand-port",
                "/dev/midas",
            ],
        ):
            config = build_replay_config(parse_args(), (4, 5, 12))

        self.assertEqual(config.dataset_root, Path("dataset"))
        self.assertEqual(config.hand_config, Path("midas.yaml"))
        self.assertEqual(config.hand_port, "/dev/midas")
        self.assertEqual(config.active_motors, [4, 5, 12])
        self.assertEqual(config.follower_joint_command_scales[4], 0.6)
        self.assertEqual(config.follower_joint_command_offsets[12], 0.262)

    def test_hand_only_offsets_are_resolved_in_recorded_motor_order(self):
        values = config_values_for_motor_ids(
            {4: 0.1, 12: 0.262}, (12, 5, 4), default=0.0
        )

        np.testing.assert_allclose(values, [0.262, 0.0, 0.1])

    def test_current_sampling_is_downsampled_to_ten_hz(self):
        replay_times = np.arange(31, dtype=np.float64) / 30.0
        next_sample_s = 0.0
        sampled_times = []
        for elapsed_s in replay_times:
            due, next_sample_s = advance_current_sample_schedule(
                float(elapsed_s),
                next_sample_s,
                1.0 / CURRENT_TELEMETRY_HZ,
            )
            if due:
                sampled_times.append(elapsed_s)

        np.testing.assert_allclose(sampled_times, np.arange(11) / 10.0)

    def test_output_current_cli_option_is_opt_in_and_requires_a_path(self):
        with patch("sys.argv", ["replay_episode.py"]):
            self.assertIsNone(parse_args().output_current)
        with patch(
            "sys.argv", ["replay_episode.py", "--output-current", "telemetry.csv"]
        ):
            self.assertEqual(parse_args().output_current, Path("telemetry.csv"))

    def test_output_video_cli_option_is_opt_in_and_requires_a_path(self):
        with patch("sys.argv", ["replay_episode.py"]):
            self.assertIsNone(parse_args().output_video)
        with patch(
            "sys.argv", ["replay_episode.py", "--output-video", "replay.mp4"]
        ):
            self.assertEqual(parse_args().output_video, Path("replay.mp4"))

    def test_arm_diagnostics_cli_is_opt_in(self):
        with patch("sys.argv", ["replay_episode.py"]):
            args = parse_args()
            self.assertFalse(args.arm_diag)
            self.assertEqual(args.arm_diag_interval_s, 0.5)
        with patch(
            "sys.argv",
            [
                "replay_episode.py",
                "--arm-diag",
                "--arm-diag-interval-s",
                "0.25",
            ],
        ):
            args = parse_args()
            self.assertTrue(args.arm_diag)
            self.assertEqual(args.arm_diag_interval_s, 0.25)

    def test_piper_speed_settings_match_trajectory_replay(self):
        with patch("sys.argv", ["replay_episode.py"]):
            args = parse_args()
        self.assertEqual(args.preposition_speed_percent, 20)
        self.assertEqual(args.replay_speed_percent, 100)
        with patch(
            "sys.argv",
            ["replay_episode.py", "--preposition-speed-percent", "30",
             "--replay-speed-percent", "60"],
        ):
            args = parse_args()
        self.assertEqual(args.preposition_speed_percent, 30)
        self.assertEqual(args.replay_speed_percent, 60)

    def test_piper_diagnostics_report_limit_and_tracking_errors(self):
        class FakeRobot:
            @staticmethod
            def get_joints_enable_status_list():
                return [True] * 6

        class FakeAPI:
            robot = FakeRobot()

            @staticmethod
            def get_state():
                return {
                    "is_ok": "True",
                    "receive_fps": "200.0",
                    "ctrl_mode": "TEACHING_MODE(0x2)",
                    "arm_status": "TARGET_POS_EXCEEDS_LIMIT(0x4)",
                    "mode_feedback": "MOVE_P(0x0)",
                    "teach_status": "STOP_RECORDING(0x2)",
                    "motion_status": "REACH_TARGET_POS_FAILED(0x1)",
                    "err_code": (1 << 10) | (1 << 4),
                    "x": "0.0",
                    "y": "0.0",
                    "z": "0.0",
                    "roll": "0.0",
                    "pitch": "0.0",
                    "yaw": "0.0",
                }

        output = io.StringIO()
        with redirect_stdout(output):
            log_piper_diagnostics(FakeAPI(), "test", np.ones(6))

        text = output.getvalue()
        self.assertIn("control_mode_not_CAN", text)
        self.assertIn("joint_angle_limit=[3]", text)
        self.assertIn("joint_communication=[5]", text)
        self.assertIn("position_error_m=1.732051", text)

    def test_piper_error_code_decodes_joint_bits(self):
        self.assertEqual(
            decode_piper_error_code((1 << 8) | (1 << 13) | (1 << 1)),
            ([1, 6], [2]),
        )

    def test_current_output_is_compatible_with_graph_generator(self):
        class FakeHand:
            def __init__(self):
                self.sample = 0

            def read_pos_vel_cur(self):
                self.sample += 1
                return (
                    np.asarray([0.1, 0.2]) * self.sample,
                    np.asarray([1.0, 2.0]),
                    np.asarray([10.0, 20.0]) * self.sample,
                )

        recorder = ReplayCurrentRecorder((4, 9))
        hand = FakeHand()
        recorder.sample(hand, np.asarray([0.0, 0.1]), elapsed_s=0.0)
        recorder.sample(hand, np.asarray([0.2, 0.5]), elapsed_s=0.1)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "nested" / "replay.csv"
            output_current_csv(recorder, output)
            trials, times, motors = _load_telemetry(output)

        np.testing.assert_array_equal(trials, [1, 1])
        np.testing.assert_allclose(times, [0.0, 0.1])
        np.testing.assert_allclose(motors[4]["current"], [10.0, 20.0])
        np.testing.assert_allclose(motors[9]["target"], [0.1, 0.5])
        np.testing.assert_allclose(motors[9]["target_velocity"], [0.0, 4.0])
        np.testing.assert_allclose(motors[9]["actual_velocity"], [2.0, 2.0])

    def test_use_follower_cli_option(self):
        with patch("sys.argv", ["replay_episode.py", "--use-follower"]):
            self.assertTrue(parse_args().use_follower)

    def test_follower_config_is_subset_to_recorded_motor_ids(self):
        ReplayConfig = make_dataclass("ReplayConfig", [("operating_mode", int)])
        loaded = ReplayConfig(operating_mode=5)
        path = Path("midas.yaml")
        with patch(
            "data_collection.replay_episode.load_follower_config", return_value=loaded
        ) as loader:
            actual = build_replay_hand_config(
                path,
                "/dev/midas",
                (0, 1, 2, 3),
                use_follower=True,
            )

        self.assertIsNot(actual, loaded)
        self.assertEqual(actual.operating_mode, 3)
        self.assertEqual(loaded.operating_mode, 5)
        loader.assert_called_once_with(path, "/dev/midas", (0, 1, 2, 3))

    def test_follower_scale_is_applied_to_all_hand_positions(self):
        times = np.asarray([0.0, 1.0])
        positions = np.asarray([[1.0, -2.0], [3.0, -4.0]])
        scaled_times, scaled_positions = scale_hand_stream(
            (times, positions), 0.25
        )
        np.testing.assert_array_equal(scaled_times, times)
        np.testing.assert_allclose(
            scaled_positions, [[0.25, -0.5], [0.75, -1.0]]
        )
        np.testing.assert_array_equal(positions, [[1.0, -2.0], [3.0, -4.0]])

    def test_joint_specific_scales_are_applied_by_column(self):
        times = np.asarray([0.0, 1.0])
        positions = np.asarray([[1.0, -2.0], [3.0, -4.0]])

        _, scaled_positions = scale_hand_stream(
            (times, positions), np.asarray([0.5, 2.0])
        )

        np.testing.assert_allclose(scaled_positions, [[0.5, -4.0], [1.5, -8.0]])

    def test_midas_replay_retargets_only_joints_two_and_three(self):
        times = np.asarray([0.0, 1.0])
        positions = np.asarray(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]]
        )
        model = ThumbRetargetModel(np.eye(2) * 2.0, np.ones(2))

        _, targets = prepare_hand_stream_for_replay(
            (times, positions),
            (0, 1, 2, 3, 4),
            use_follower=True,
            command_scale=0.5,
            joint_command_scales=np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
            thumb_model=model,
        )

        np.testing.assert_allclose(
            targets,
            [[0.05, 0.2, 7.0, 9.0, 1.25], [0.1, 0.3, 9.0, 11.0, 1.5]],
        )

    def test_trimorph_replay_keeps_raw_joint_positions(self):
        times = np.asarray([0.0, 1.0])
        positions = np.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        stream = (times, positions)

        result = prepare_hand_stream_for_replay(
            stream,
            (0, 1, 2, 3),
            use_follower=False,
            command_scale=100.0,
            joint_command_scales=np.full(4, 100.0),
            thumb_model=ThumbRetargetModel(np.eye(2) * 100.0, np.ones(2)),
        )

        np.testing.assert_array_equal(result[0], times)
        np.testing.assert_array_equal(result[1], positions)

    def test_midas_replay_applies_joint_offsets(self):
        _, targets = prepare_hand_stream_for_replay(
            (np.asarray([0.0]), np.asarray([[0.0, 1.0]])),
            (12, 4),
            use_follower=True,
            command_scale=1.0,
            joint_command_scales=np.asarray([1.0, 0.4]),
            joint_command_offsets=np.asarray([0.262, 0.0]),
            thumb_model=None,
        )

        np.testing.assert_allclose(targets, [[0.262, 0.4]])

    def test_action_scale_multiplies_all_final_replay_targets(self):
        times = np.asarray([0.0])
        positions = np.asarray([[1.0, 2.0]])

        raw_times, raw_targets = prepare_hand_stream_for_replay(
            (times, positions),
            (4, 12),
            use_follower=False,
            command_scale=1.0,
            joint_command_scales=np.ones(2),
            thumb_model=None,
            action_scale=1.5,
        )
        _, follower_targets = prepare_hand_stream_for_replay(
            (times, positions),
            (4, 12),
            use_follower=True,
            command_scale=1.0,
            joint_command_scales=np.asarray([0.5, 1.0]),
            joint_command_offsets=np.asarray([0.1, 0.2]),
            thumb_model=None,
            action_scale=2.0,
        )

        np.testing.assert_array_equal(raw_times, times)
        np.testing.assert_allclose(raw_targets, [[1.5, 3.0]])
        np.testing.assert_allclose(follower_targets, [[1.2, 4.4]])

    def test_joint_action_scales_follow_motor_ids_not_column_order(self):
        _, targets = prepare_hand_stream_for_replay(
            (np.asarray([0.0]), np.asarray([[1.0, 2.0, 3.0, 4.0]])),
            (2, 1, 4, 0),
            use_follower=False,
            command_scale=1.0,
            joint_command_scales=np.ones(4),
            thumb_model=None,
            action_scale=2.0,
            joint_action_scales={2: 1.0, 1: 1.5, 4: 1.3, 0: 1.5},
        )
        np.testing.assert_allclose(targets, [[2.0, 6.0, 7.8, 12.0]])

    def test_joint_action_scales_apply_to_every_recorded_motor_id(self):
        motor_ids = tuple(range(12, -1, -1))
        _, targets = prepare_hand_stream_for_replay(
            (np.asarray([0.0]), np.ones((1, 13))),
            motor_ids,
            use_follower=False,
            command_scale=1.0,
            joint_command_scales=np.ones(13),
            thumb_model=None,
            joint_action_scales=JOINT_ACTION_SCALES,
        )
        np.testing.assert_allclose(
            targets[0], [JOINT_ACTION_SCALES[motor_id] for motor_id in motor_ids]
        )

    def test_joint_action_scales_cover_finger_two(self):
        _, targets = prepare_hand_stream_for_replay(
            (np.asarray([0.0]), np.asarray([[1.0, 2.0, 3.0, 4.0]])),
            (6, 7, 8, 9),
            use_follower=False,
            command_scale=1.0,
            joint_command_scales=np.ones(4),
            thumb_model=None,
            action_scale=2.0,
            joint_action_scales={6: 1.0, 7: 1.5, 8: 1.5, 9: 1.0},
        )

        np.testing.assert_allclose(targets, [[2.0, 6.0, 9.0, 8.0]])

    def test_joint_action_scales_cover_finger_one(self):
        _, targets = prepare_hand_stream_for_replay(
            (np.asarray([0.0]), np.asarray([[1.0, 2.0, 3.0, 4.0]])),
            (3, 4, 5, 6),
            use_follower=False,
            command_scale=1.0,
            joint_command_scales=np.ones(4),
            thumb_model=None,
            action_scale=2.0,
            joint_action_scales={3: 1.0, 4: 1.5, 5: 1.5, 6: 1.0},
        )

        np.testing.assert_allclose(targets, [[2.0, 6.0, 9.0, 8.0]])

    def test_legacy_proprioception_filename_is_still_supported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            episode_dir = Path(temp_dir)
            legacy_path = episode_dir / "proprioception.csv"
            legacy_path.touch()
            self.assertEqual(resolve_leader_hand_path(episode_dir), legacy_path)

    def test_hand_stream_is_sorted_and_interpolated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "leader_hand.csv"
            fields = ["elapsed_s", *(f"pos_{motor_id}" for motor_id in range(13))]
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                for elapsed_s, value in ((1.0, 2.0), (0.0, 0.0)):
                    row = {"elapsed_s": elapsed_s}
                    row.update({f"pos_{motor_id}": value for motor_id in range(13)})
                    writer.writerow(row)
            stream = load_hand_stream(path)
            np.testing.assert_allclose(sample_stream(*stream, 0.25), np.full(13, 0.5))

    def test_hand_stream_uses_only_recorded_motor_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "leader_hand.csv"
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(
                    output, fieldnames=["elapsed_s", "pos_4", "pos_5"]
                )
                writer.writeheader()
                writer.writerow({"elapsed_s": 0.0, "pos_4": 1.0, "pos_5": 2.0})
            times, values = load_hand_stream(path)
            np.testing.assert_allclose(times, [0.0])
            np.testing.assert_allclose(values, [[1.0, 2.0]])

    def test_replay_interval_uses_stream_overlap(self):
        hand = (np.asarray([0.1, 2.0]), np.zeros((2, 13)))
        arm = (np.asarray([0.2, 3.0]), np.zeros((2, 6)))
        self.assertEqual(replay_interval(hand, arm), (0.2, 2.0))

    def test_empty_piper_stream_is_allowed_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "piper.csv"
            path.write_text("", encoding="utf-8")
            self.assertIsNone(load_piper_stream(path, allow_empty=True))
            with self.assertRaisesRegex(ValueError, "No samples"):
                load_piper_stream(path)

    def test_joint_level_piper_stream_requires_joint_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "piper.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=["elapsed_s", *(f"joint_{i}" for i in range(1, 7))],
                )
                writer.writeheader()
                writer.writerow(
                    {"elapsed_s": 0.0, **{f"joint_{i}": float(i) for i in range(1, 7)}}
                )
            stream = load_piper_stream(path, joint_level=True)
            self.assertIsNotNone(stream)
            np.testing.assert_allclose(stream[1], [[1, 2, 3, 4, 5, 6]])
            with self.assertRaisesRegex(ValueError, "Missing columns"):
                load_piper_stream(path)

    @patch("data_collection.replay_episode.time.sleep")
    @patch("scripts.piper_api.replay_trajectory.preposition_joints_linearly")
    def test_joint_episode_preposition_uses_linear_ramp(self, ramp, _sleep):
        from unittest.mock import Mock

        api = Mock()
        args = SimpleNamespace(
            no_preposition=False,
            joint_level=True,
            preposition_seconds=3.0,
            max_joint_step_rad=0.02,
            settle_seconds=0.0,
            preposition_speed_percent=30,
        )
        piper_stream = (np.asarray([0.0]), np.asarray([[0.5] * 6]))
        move_to_first_pose(None, api, None, piper_stream, 0.0, args)
        ramp.assert_called_once()
        np.testing.assert_allclose(ramp.call_args.args[1], [0.5] * 6)
        api.set_speed_percent.assert_called_once_with(30)
        api.move_eef.assert_not_called()


if __name__ == "__main__":
    unittest.main()
