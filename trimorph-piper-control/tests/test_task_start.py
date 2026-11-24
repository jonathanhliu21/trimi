"""Task-start capture uses saved CAN receipts and measured motor IDs only."""
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data_collection.collection.task_start import (
    capture_task_start, load_task_start, task_start_sha256,
    validate_task_start, write_task_start,
)
from scripts.kinesthetic.capture_task_start import main


BASE_NS = 2_000_000_000


class TaskStartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.episode = self.root / "raw" / "episode_000001"
        self.episode.mkdir(parents=True)
        self.metadata = {
            "started_monotonic_ns": BASE_NS,
            "piper": {"gravity_comp": {"calibration_sha256": "b" * 64}},
            "trimorph": {"hand": {
                "motor_ids": list(reversed(range(13))),
                "config_sha256": "a" * 64,
                "joint_lower_limits": {str(i): -1.0 for i in range(13)},
                "joint_upper_limits": {str(i): 1.0 for i in range(13)},
            }},
        }
        self.arm_rows = [
            dict(joint_rx_monotonic_ns=BASE_NS + ms * 1_000_000,
                 monotonic_ns=BASE_NS + (ms + 100) * 1_000_000,
                 **{f"joint_{j}": math.radians(q + sample) for j, q in enumerate([0, 90, -90, 0, 0, 0], 1)})
            for sample, ms in enumerate([5, 15, 25])
        ]
        # Deliberately reverse the header/metadata motor order. The profile must
        # still pair motor IDs and their measured columns correctly.
        self.hand_rows = [
            dict(monotonic_ns=BASE_NS + ms * 1_000_000,
                 **{f"pos_{i}": i / 10 + sample for i in reversed(range(13))})
            for sample, ms in enumerate([7, 27])
        ]
        self.save_episode()

    def save_episode(self):
        (self.episode / "metadata.json").write_text(json.dumps(self.metadata))
        for name, rows in (("piper.csv", self.arm_rows), ("leader_hand.csv", self.hand_rows)):
            with (self.episode / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def capture(self, **kwargs):
        return capture_task_start("screw", self.episode, **kwargs)

    def test_round_trip_preserves_raw_pose_motor_mapping_and_provenance(self):
        self.arm_rows[2]["joint_6"] = math.radians(400.0)
        self.save_episode()
        profile = self.capture(at_s=0.025)
        self.assertAlmostEqual(profile["arm"]["joints_deg"][5], 400)
        self.assertEqual(profile["hand"]["motor_ids"], list(range(13)))
        self.assertEqual(profile["hand"]["positions_rad"], [i / 10 + 1 for i in range(13)])
        self.assertEqual(profile["source"]["arm_monotonic_ns"], BASE_NS + 25_000_000)
        self.assertEqual(profile["source"]["hand_monotonic_ns"], BASE_NS + 27_000_000)
        self.assertEqual(profile["source"]["calibration_sha256"], "b" * 64)
        self.assertEqual(profile["source"]["sync_error_ms"], 2)
        self.assertTrue(any("motor 12" in warning for warning in profile["warnings"]))
        path = write_task_start(self.root / "starts" / "screw.json", profile)
        loaded = load_task_start(path)
        self.assertEqual(loaded, profile)
        self.assertEqual(task_start_sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())
        loaded["arm"]["joints_deg"][0] = 999
        self.assertEqual(load_task_start(path), profile)

    def test_selects_can_receipt_then_nearest_hand_without_interpolation(self):
        profile = self.capture(at_s=0.014)
        self.assertAlmostEqual(profile["arm"]["joints_deg"][0], 1.0)
        self.assertEqual(profile["source"]["arm_monotonic_ns"], BASE_NS + 15_000_000)
        self.assertEqual(profile["source"]["hand_monotonic_ns"], BASE_NS + 7_000_000)
        self.assertEqual(profile["hand"]["positions_rad"][12], 1.2)
        self.assertEqual(profile["source"]["at_s"], 0.014)

    def test_default_time_uses_episode_start(self):
        profile = self.capture()
        self.assertEqual(profile["source"]["arm_monotonic_ns"], BASE_NS + 5_000_000)
        self.assertAlmostEqual(profile["arm"]["joints_deg"][0], 0.0)

    def test_existing_profile_requires_explicit_overwrite(self):
        path = self.root / "profile.json"
        first = self.capture()
        write_task_start(path, first)
        later = self.capture(at_s=0.025)
        with self.assertRaises(FileExistsError):
            write_task_start(path, later)
        self.assertEqual(load_task_start(path), first)
        write_task_start(path, later, overwrite=True)
        self.assertEqual(load_task_start(path), later)

    def test_refuses_poor_arm_hand_sync_and_missing_stream(self):
        self.hand_rows = [dict(self.hand_rows[0], monotonic_ns=BASE_NS + 100_000_000)]
        self.save_episode()
        with self.assertRaisesRegex(ValueError, "50 ms"):
            self.capture()
        (self.episode / "leader_hand.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            self.capture()

    def test_refuses_out_of_coverage_and_invalid_requested_time(self):
        for value in (-0.1, float("nan"), float("inf"), True, 20, 1e308, 10 ** 1000):
            with self.subTest(at_s=value), self.assertRaises(ValueError):
                self.capture(at_s=value)

    def test_refuses_missing_can_clock_instead_of_using_poll_clock(self):
        for row in self.arm_rows:
            del row["joint_rx_monotonic_ns"]
        self.save_episode()
        with self.assertRaisesRegex(ValueError, "columns"):
            self.capture()

    def test_refuses_nonincreasing_source_clocks(self):
        self.arm_rows[1]["joint_rx_monotonic_ns"] = self.arm_rows[0]["joint_rx_monotonic_ns"]
        self.save_episode()
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.capture()

    def test_refuses_missing_or_nonfinite_selected_measurements(self):
        for field, bad_value in (("joint_4", ""), ("joint_4", "nan")):
            with self.subTest(field=field, value=bad_value):
                self.arm_rows[0][field] = bad_value
                self.save_episode()
                with self.assertRaisesRegex(ValueError, "invalid joint_4"):
                    self.capture()
        self.arm_rows[0]["joint_4"] = 0
        self.hand_rows[0]["pos_8"] = "inf"
        self.save_episode()
        with self.assertRaisesRegex(ValueError, "invalid pos_8"):
            self.capture()

    def test_refuses_missing_motor_identity_or_configuration_hash(self):
        self.metadata["trimorph"]["hand"]["motor_ids"][0] = 0
        self.save_episode()
        with self.assertRaisesRegex(ValueError, "motor IDs"):
            self.capture()
        self.metadata["trimorph"]["hand"]["motor_ids"] = list(range(13))
        del self.metadata["trimorph"]["hand"]["config_sha256"]
        self.save_episode()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.capture()

    def test_limit_warnings_preserve_unclipped_hand_and_arm_values(self):
        self.hand_rows[0]["pos_8"] = -7.25
        self.arm_rows[0]["joint_5"] = math.radians(90)
        self.save_episode()
        with patch("data_collection.collection.task_start._arm_limit_warnings", return_value=["Arm J5 outside SDK limits."]):
            profile = self.capture()
        self.assertEqual(profile["hand"]["positions_rad"][8], -7.25)
        self.assertAlmostEqual(profile["arm"]["joints_deg"][4], 90)
        self.assertTrue(any("motor 8" in warning for warning in profile["warnings"]))
        self.assertIn("Arm J5 outside SDK limits.", profile["warnings"])

    def test_schema_rejects_bad_shape_nonfinite_duplicate_ids_or_task(self):
        valid = self.capture()
        mutations = [
            lambda x: x.update(schema_version=2),
            lambda x: x.update(task=" "),
            lambda x: x["arm"].update(joints_deg=[0] * 5),
            lambda x: x["arm"]["joints_deg"].__setitem__(0, float("nan")),
            lambda x: x["hand"]["motor_ids"].__setitem__(0, 1),
            lambda x: x["hand"].update(positions_rad=[0] * 12),
            lambda x: x["source"].update(hand_monotonic_ns=BASE_NS + 500_000_000),
        ]
        for mutate in mutations:
            profile = json.loads(json.dumps(valid))
            mutate(profile)
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                validate_task_start(profile)
        snapshot = validate_task_start(valid)
        snapshot["hand"]["positions_rad"][0] = 99
        self.assertNotEqual(snapshot, valid)

    def test_cli_writes_and_reports_measured_snapshot(self):
        output = self.root / "screw.json"
        with patch("builtins.print") as printer:
            result = main(["--task", "screw", "--from-episode", str(self.episode),
                           "--at-s", "0.025", "--output", str(output)])
        self.assertEqual(result, 0)
        self.assertEqual(load_task_start(output)["task"], "screw")
        self.assertTrue(any("SHA-256" in str(call) for call in printer.call_args_list))


if __name__ == "__main__":
    unittest.main()
