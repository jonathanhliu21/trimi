import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from data_collection.collect_hand_only import start_episode, stop_episode
from data_collection.collection.config import ARM_MODE_GRAVITY_COMP, HandOnlyCollectionConfig
from data_collection.collection.episode import RecordingManager
from data_collection.collection.piper_worker import run_piper_worker
from data_collection.collection.shared_state import SharedState
from scripts.grav_comp.listener_bench_report import checks, main, summarize
from test_piper_worker import fake_sdk_robot


def record_bench_episode(root: Path, stamps, **robot_kwargs) -> Path:
    """Run the real listener on a fake arm into a real episode folder."""
    config = HandOnlyCollectionConfig(
        dataset_root=root,
        hand_config=Path("config/xl330_new_hand_transparent.yaml"),
        hand_port="/dev/ttyUSB0",
        hand_measure_rate_hz=50.0,
        warmup_time=0.0,
        episode_time=30.0,
        use_piper=True,
        arm_mode=ARM_MODE_GRAVITY_COMP,
        piper_rate_hz=20000.0,
        piper_installation_pos="unchanged",
        piper_payload="unchanged",
    )
    shared = SharedState()
    recorder = RecordingManager()
    with patch("data_collection.collect_hand_only.print"):
        session = start_episode(config, recorder)
    robot = fake_sdk_robot(stamps, on_exhausted=shared.shutdown.set, **robot_kwargs)
    with (
        patch("scripts.piper_api.piper_api.create_piper_robot", return_value=robot),
        patch("scripts.piper_api.piper_api.time.sleep"),
        patch("data_collection.collection.piper_worker.print"),
    ):
        run_piper_worker(config, shared, recorder, init_done=threading.Event())
    with patch("data_collection.collect_hand_only.print"):
        stop_episode(recorder, "duration_complete")
    return session.paths.episode_dir


class ListenerBenchReportTests(unittest.TestCase):
    def test_report_on_a_recorded_200_hz_episode(self):
        t0 = time.time()
        # 200 Hz with one 30 ms gap.
        stamps = [t0 + 0.005 * i for i in range(100)] + [t0 + 0.525 + 0.005 * i for i in range(100)]
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = summarize(record_bench_episode(Path(temp_dir), stamps))
        self.assertEqual(summary["piper_mode"], "gravity_comp_kinesthetic")
        self.assertEqual(summary["rows"], 200)
        self.assertEqual(summary["duplicate_frames"], 0)
        self.assertAlmostEqual(summary["joint_rate_hz"], 199 / 1.02, places=1)
        self.assertAlmostEqual(summary["max_interval_ms"], 30.0, places=1)
        self.assertEqual(summary["gaps_over_20ms"], 1)
        self.assertEqual(summary["arm_ok_fraction"], 1.0)
        self.assertEqual(summary["mode_feedback"], ["MIT"])
        self.assertEqual(summary["eef_fraction"], 1.0)
        results = checks(summary)
        self.assertTrue(all(ok for ok, _ in results), results)
        self.assertIn("streams in MIT mode", results[-1][1])

    def test_disabled_drives_and_missing_end_pose_are_reported(self):
        t0 = time.time()
        stamps = [t0 + 0.005 * i for i in range(50)]
        with tempfile.TemporaryDirectory() as temp_dir:
            episode = record_bench_episode(Path(temp_dir), stamps, enabled=False, eef_stamp_offset=-5.0)
            summary = summarize(episode)
            with patch("builtins.print"):
                self.assertEqual(main([str(episode)]), 1)
        self.assertEqual(summary["arm_ok_fraction"], 0.0)
        self.assertEqual(summary["flags"], ["arm_not_ok"])
        results = dict((text.split(" ")[0], ok) for ok, text in checks(summary))
        self.assertFalse(results["drives"])
        self.assertIn("does NOT stream", checks(summary)[-1][1])


if __name__ == "__main__":
    unittest.main()
