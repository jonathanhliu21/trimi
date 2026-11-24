"""Collector CLI and recorded policy for the end-effector start check."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data_collection.collect_hand_only import main, parse_args, start_episode, stop_episode
from data_collection.collection.episode import RecordingManager


class KinestheticStartThresholdTests(unittest.TestCase):
    def test_default_pose_thresholds_use_metres_and_total_rotation_degrees(self):
        with patch("sys.argv", ["collect", "--arm-mode", "gravity-comp"]):
            args = parse_args()
        self.assertEqual(args.kinesthetic_start_position_tolerance_m, .10)
        self.assertEqual(args.kinesthetic_start_orientation_tolerance_deg, 45.)

    def test_selected_pose_thresholds_are_saved_with_the_start_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calib = root / "calibration.json"
            calib.write_text("{}")
            argv = ["collect", "--arm-mode", "gravity-comp", "--kinesthetic-start", "kinesthetic",
                    "--grav-comp-calib", str(calib), "--dataset-root", str(root),
                    "--kinesthetic-start-position-tolerance-m", ".07",
                    "--kinesthetic-start-orientation-tolerance-deg", "33"]
            with patch("sys.argv", argv), patch("data_collection.collect_hand_only.run_hand_only", return_value=0) as run:
                self.assertEqual(main(), 0)
            config = run.call_args.args[0]
            self.assertEqual(config.kinesthetic_start_position_tolerance_m, .07)
            self.assertEqual(config.kinesthetic_start_orientation_tolerance_deg, 33.)
            recorder = RecordingManager()
            with contextlib.redirect_stdout(io.StringIO()):
                episode = start_episode(config, recorder)
                stop_episode(recorder, "foot_pedal")
            meta = json.loads(episode.paths.metadata_path.read_text())
            guide = meta["piper"]["gravity_comp"]
            self.assertEqual(guide["start_criterion"], "end_effector_pose")
            self.assertEqual(guide["start_position_tolerance_m"], .07)
            self.assertEqual(guide["start_orientation_tolerance_deg"], 33.)
            self.assertNotIn("start_tolerance_deg", guide)
            self.assertEqual(len(guide["guided_start_joints_deg"]), 6)

    def test_invalid_thresholds_are_rejected_before_starting_workers(self):
        for flag, values in (
            ("--kinesthetic-start-position-tolerance-m", ("0", "-1", "nan", "inf")),
            ("--kinesthetic-start-orientation-tolerance-deg", ("0", "-1", "nan", "inf", "181")),
        ):
            for value in values:
                with self.subTest(flag=flag, value=value), patch("sys.argv", ["collect", flag, value]), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args()


if __name__ == "__main__":
    unittest.main()
