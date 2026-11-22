"""Cross-repository regression: measured Piper poses -> saved episode -> training.

Run with the sibling trimorph-piper-control checkout, or set TRIMORPH_CONTROL_ROOT.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from shared.env.midas.midas_image_dataset import MidasImageDataset
from shared.env.midas.se3_utils import relative_to_absolute

CONTROL_ROOT = Path(os.environ.get(
    "TRIMORPH_CONTROL_ROOT",
    Path(__file__).resolve().parents[2] / "trimorph-piper-control",
))


@unittest.skipUnless((CONTROL_ROOT / "data_collection").is_dir(), "Requires control checkout")
class PiperTrainingPipelineTests(unittest.TestCase):
    def test_saved_cartesian_episode_without_trackers_loads_for_training(self):
        with patch.object(sys, "path", [str(CONTROL_ROOT), *sys.path]):
            from data_collection.collection.config import HandOnlyCollectionConfig, RAW_EEF_FIELDS
            from data_collection.collection.episode import EpisodeBuffer, episode_paths
            from data_collection import postprocess_dataset as post

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = HandOnlyCollectionConfig(
                dataset_root=root, hand_config=root / "hand.yaml", hand_port="unused",
                hand_measure_rate_hz=50., warmup_time=0., episode_time=1.,
                use_piper=True, use_quest=False, use_mocap=False, add_camera=True,
            )
            paths = episode_paths(root, 0)
            paths.episode_dir.mkdir(parents=True)
            session = EpisodeBuffer(paths, config)
            # Nontrivial simultaneous rotation distinguishes Euler angles from rotvecs.
            poses = np.array([[.4 + .01*i, -.2 + .003*i, .3, .5 + .02*i, -.6, .7]
                              for i in range(8)])
            hands = np.arange(8*13, dtype=float).reshape(8, 13) / 100
            writer = cv2.VideoWriter(str(paths.camera_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     30., (32, 24))
            self.assertTrue(writer.isOpened())
            try:
                for i, (pose, hand) in enumerate(zip(poses, hands)):
                    ns = session.start_ns + i * 33_333_333
                    session.append_piper(dict(monotonic_ns=ns, elapsed_s=i/30,
                                              **dict(zip(RAW_EEF_FIELDS, pose))))
                    session.append_leader_hand(dict(monotonic_ns=ns,
                                                   **{f"pos_{j}": v for j, v in enumerate(hand)}))
                    session.append_camera(dict(monotonic_ns=ns, frame_idx=i))
                    writer.write(np.full((24, 32, 3), i*20, dtype=np.uint8))
            finally:
                writer.release()
            with patch("data_collection.collection.episode._hand_metadata", return_value={}):
                session.flush()
            self.assertFalse(paths.quest_path.exists())
            self.assertFalse(paths.mocap_path.exists())
            before = paths.piper_path.read_bytes()
            processing = post.PostprocessConfig(root, root / "raw", False, {"complete"}, .01, 1,
                                                pose_source="piper")
            summary = post.process_episode(processing, post.base.discover_raw_episodes(processing)[0])
            self.assertEqual(paths.piper_path.read_bytes(), before)
            expected = np.c_[poses[:, :3], Rotation.from_euler("xyz", poses[:, 3:]).as_rotvec()]
            np.testing.assert_allclose(np.load(summary["eef"]), expected, atol=1e-7)
            metadata = json.loads(Path(summary["metadata"]).read_text())
            self.assertEqual(metadata["source"], "piper.csv")
            self.assertEqual(metadata["pose_frame"], "piper_base")
            with np.load(summary["sync"]) as sync:
                np.testing.assert_array_equal(sync["frame_idx"], np.arange(1, 7))
                np.testing.assert_array_equal(sync["pose_idx"], np.arange(1, 7))
                np.testing.assert_allclose(sync["start_pose"], expected[0], atol=1e-7)
            dataset = MidasImageDataset(
                shape_meta={"obs": {"camera_0": {"shape": [3, 24, 32], "type": "rgb"}},
                            "action": {"shape": [19]}},
                dataset_path=str(root), horizon=3, num_obs_steps=1, val_ratio=0.,
                encoder_window=0, sync_downsample_stride=1,
            )
            batch = dataset[0]
            self.assertEqual(tuple(batch["action"].shape), (3, 19))
            self.assertEqual(tuple(batch["obs"]["camera_0"].shape), (1, 3, 24, 32))
            np.testing.assert_allclose(batch["action"][0], 0., atol=1e-7)
            # Recover the original camera-aligned Cartesian poses and hand targets.
            absolute = relative_to_absolute(batch["action"].numpy(), np.r_[expected[1], hands[1]])
            np.testing.assert_allclose(absolute, np.c_[expected[1:4], hands[1:4]], atol=2e-7)
            normalizer = dataset.get_normalizer()
            self.assertTrue(np.isfinite(normalizer["action"].normalize(batch["action"]).numpy()).all())


if __name__ == "__main__":
    unittest.main()
