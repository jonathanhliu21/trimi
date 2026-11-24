"""DAgger training support: several dataset roots, per-source sampling, warm start.

Run with: python -m unittest discover -s tests -v
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import cv2
import dill
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import RandomSampler, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_shortcut_policy import make_batch, make_policy  # noqa: E402

from shared.env.midas.midas_image_dataset import (  # noqa: E402
    MidasImageDataset,
    _source_window_weights,
)
from shared.models.common.normalizer import LinearNormalizer  # noqa: E402
from shared.utils.sampler import get_val_mask  # noqa: E402

SHAPE_META = {"obs": {"camera_0": {"shape": [3, 24, 32], "type": "rgb"}},
              "action": {"shape": [19]}}
# Hand-joint step per sync row: each root moves its hand at its own speed, so a
# window's relative hand action reveals which root it came from.
SLOPES = {"root_a": 0.01, "root_b": 0.05}


def write_root(root, n_episodes, steps, slope):
    """A synthetic MIDAS root in the postprocessor's layout."""
    for sub in ("camera/mp4_files", "processed_encoder_t", "absolute_action_t", "syncs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    t = np.arange(steps, dtype=np.float32)
    for episode in range(n_episodes):
        name = f"{episode:06d}"
        writer = cv2.VideoWriter(str(root / "camera/mp4_files" / f"{name}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 30., (32, 24))
        assert writer.isOpened()
        for i in range(steps):
            writer.write(np.full((24, 32, 3), (40 * episode + 10 * i) % 250, dtype=np.uint8))
        writer.release()
        hand = episode + slope * t[:, None] * np.ones((1, 13), dtype=np.float32)
        eef = np.c_[.4 + .001 * t, -.2 + .002 * t, np.full_like(t, .3),
                    .01 * t, np.full_like(t, -.02), np.full_like(t, .03)]
        np.save(root / "processed_encoder_t" / f"{name}.npy", hand.astype(np.float32))
        np.save(root / "absolute_action_t" / f"{name}.npy", eef.astype(np.float32))
        index = np.arange(steps, dtype=np.int64)
        np.savez(root / "syncs" / f"{name}.npz", frame_idx=index, encoder_idx=index, pose_idx=index)


def make_dataset(dataset_path, **kwargs):
    options = dict(shape_meta=SHAPE_META, dataset_path=dataset_path, horizon=4,
                   pad_before=0, pad_after=0, num_obs_steps=1, val_ratio=.25, seed=42,
                   encoder_window=0, sync_downsample_stride=1)
    options.update(kwargs)
    return MidasImageDataset(**options)


class ToySourceDataset(torch.utils.data.Dataset):
    """Toy two-source dataset for the workspace: per-window 'source' ids and weights."""

    def __init__(self, dataset_path=None):
        self.batch = make_batch(10)
        self.sources = np.array([0] * 6 + [1] * 4)
        self.source_names = ["toy_a", "toy_b"]
        self.sample_weights = np.where(self.sources == 0, .5 / 6, .5 / 4)

    def __len__(self):
        return 10

    def __getitem__(self, index):
        return {"obs": {k: v[index] for k, v in self.batch["obs"].items()},
                "action": self.batch["action"][index],
                "source": torch.tensor(int(self.sources[index]))}

    def get_normalizer(self):
        # Deliberately unlike the checkpoint's normalizer (make_policy's).
        normalizer = LinearNormalizer()
        normalizer.fit({"camera_0": torch.rand(8, 3, 4, 4),
                        "action": 10 * torch.randn(8, 3),
                        "hand_pose_abs": 10 * torch.randn(8, 2)})
        return normalizer

    def get_validation_dataset(self):
        return self


class ToyRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy):
        return {}


class DaggerDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        base = Path(cls.directory.name)
        cls.root_a, cls.root_b = base / "root_a", base / "root_b"
        write_root(cls.root_a, n_episodes=8, steps=10, slope=SLOPES["root_a"])
        write_root(cls.root_b, n_episodes=4, steps=10, slope=SLOPES["root_b"])

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_single_root_string_and_list_load_the_same(self):
        single = make_dataset(str(self.root_a))
        listed = make_dataset([str(self.root_a)])
        self.assertEqual(single.replay_buffer.n_episodes, 8)
        # Same validation episodes as get_val_mask over all episodes (pre-change behavior).
        np.testing.assert_array_equal(single.val_mask, get_val_mask(8, .25, 42))
        np.testing.assert_array_equal(listed.val_mask, single.val_mask)
        np.testing.assert_array_equal(listed.sampler.indices, single.sampler.indices)
        np.testing.assert_array_equal(listed.replay_buffer["action"][:], single.replay_buffer["action"][:])
        for dataset in (single, listed):
            self.assertEqual(dataset.n_sources, 1)
            self.assertEqual(dataset.source_names, ["root_a"])
            self.assertIsNone(dataset.sample_weights)
            self.assertNotIn("source", dataset[0])
            self.assertNotIn("source", dataset.get_validation_dataset()[0])

    def test_two_roots_concatenate_in_order_with_per_source_validation(self):
        dataset = make_dataset([str(self.root_a), str(self.root_b)])
        single_a = make_dataset(str(self.root_a))
        self.assertEqual(dataset.replay_buffer.n_episodes, 12)
        self.assertEqual(dataset.source_names, ["root_a", "root_b"])
        np.testing.assert_array_equal(dataset.episode_source, [0] * 8 + [1] * 4)
        # Each source's validation split is its own, and root_a keeps its single-root split.
        np.testing.assert_array_equal(dataset.val_mask[:8], single_a.val_mask)
        np.testing.assert_array_equal(dataset.val_mask[8:], get_val_mask(4, .25, 42))
        self.assertGreater(dataset.val_mask[:8].sum(), 0)
        self.assertGreater(dataset.val_mask[8:].sum(), 0)
        self.assertEqual(len(dataset.window_source), len(dataset))
        for index in range(len(dataset)):
            item = dataset[index]
            source = int(item["source"])
            self.assertEqual(item["source"].dtype, torch.int64)
            self.assertEqual(source, dataset.window_source[index])
            name = dataset.source_names[source]
            np.testing.assert_allclose(item["action"][1, 6:].numpy(), SLOPES[name], atol=1e-5)
        validation = dataset.get_validation_dataset()
        self.assertIsNone(validation.sample_weights)
        self.assertEqual(sorted(set(validation.window_source.tolist())), [0, 1])
        for index in range(len(validation)):
            item = validation[index]
            name = validation.source_names[int(item["source"])]
            np.testing.assert_allclose(item["action"][1, 6:].numpy(), SLOPES[name], atol=1e-5)

    def test_cached_roots_match_uncached_and_keep_their_cache_names(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            root_a, root_b = Path(directory) / "root_a", Path(directory) / "root_b"
            write_root(root_a, n_episodes=3, steps=10, slope=SLOPES["root_a"])
            write_root(root_b, n_episodes=2, steps=10, slope=SLOPES["root_b"])
            roots = [str(root_a), str(root_b)]
            # Hydra hands the dataset a DictConfig, which the cache key requires.
            meta = OmegaConf.create(SHAPE_META)
            plain = make_dataset(roots, shape_meta=meta)
            built = make_dataset(roots, shape_meta=meta, use_cache=True)
            # The cache file is per root and named exactly as for a single root.
            key = {"shape_meta": SHAPE_META, "first_cut": 0, "encoder_window": 0,
                   "sync_downsample_stride": 1}
            name = f"midas_{hashlib.md5(json.dumps(key, sort_keys=True).encode('utf-8')).hexdigest()}.zarr.zip"
            self.assertTrue((root_a / name).is_file())
            self.assertTrue((root_b / name).is_file())
            loaded = make_dataset(roots, shape_meta=meta, use_cache=True)
            single = make_dataset(str(root_b), shape_meta=meta, use_cache=True)
            for dataset in (built, loaded):
                np.testing.assert_array_equal(dataset.episode_source, plain.episode_source)
                np.testing.assert_array_equal(dataset.replay_buffer.episode_ends[:],
                                              plain.replay_buffer.episode_ends[:])
                for key in ("action", "camera_0"):
                    np.testing.assert_array_equal(dataset.replay_buffer[key][:], plain.replay_buffer[key][:])
            np.testing.assert_array_equal(single.replay_buffer["action"][:],
                                          plain.replay_buffer["action"][30:])

    def test_output_dirs_use_the_first_root(self):
        import shortcut_policy.unet_shortcut.train_unet_image_shortcut_policy  # noqa: F401 (resolvers)

        config_dir = Path(__file__).resolve().parents[1] / "shortcut_policy/config"
        for value, expected in (("[/data/base,/data/dagger_r1]", "/data/base/outputs/"),
                                ("/data/base", "/data/base/outputs/")):
            with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
                cfg = compose(config_name="train_unet_image_trimi_hand_cond_dinov2_shortcut_policy",
                              overrides=[f"tasks.dataset.dataset_path={value}"], return_hydra_config=True)
            self.assertTrue(cfg.hydra.run.dir.startswith(expected))
            self.assertTrue(cfg.hydra.sweep.dir.startswith(expected))
            self.assertTrue(cfg.multi_run.run_dir.startswith(expected))

    def test_source_names_and_weights_by_name(self):
        dataset = make_dataset([str(self.root_a), str(self.root_b)], source_names=["d0", "dagger"],
                               source_weights={"dagger": 1.0})
        self.assertEqual(dataset.source_names, ["d0", "dagger"])
        self.assertEqual(dataset.sample_weights[dataset.window_source == 0].sum(), 0)
        np.testing.assert_allclose(dataset.sample_weights[dataset.window_source == 1].sum(), 1.)
        with self.assertRaisesRegex(ValueError, "not sources"):
            make_dataset([str(self.root_a), str(self.root_b)], source_weights={"other": 1.0})

    def test_sample_weights_give_each_source_its_requested_share(self):
        dataset = make_dataset([str(self.root_a), str(self.root_b)],
                               source_weights=OmegaConf.create([.7, .3]))
        weights = dataset.sample_weights
        self.assertEqual(weights.dtype, np.float64)
        self.assertEqual(len(weights), len(dataset))
        np.testing.assert_allclose(weights.sum(), 1.)
        for source, share in enumerate((.7, .3)):
            selected = weights[dataset.window_source == source]
            np.testing.assert_allclose(selected.sum(), share)
            np.testing.assert_allclose(selected, selected[0])  # uniform inside a source
        for bad in ([1.], [1., -1.], [0., 0.], [1., float("nan")]):
            with self.assertRaises(ValueError):
                make_dataset([str(self.root_a), str(self.root_b)], source_weights=bad)

    def test_source_without_training_windows_warns_and_gets_no_mass(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            weights = _source_window_weights(np.array([0, 0, 0]), 2, [.5, .5], ["a", "b"])
        self.assertTrue(any("no training windows" in str(w.message) for w in caught))
        np.testing.assert_allclose(weights, [1 / 3] * 3)
        self.assertIsNone(_source_window_weights(np.array([0, 1]), 2, None))

    def test_train_loader_draws_sources_by_weight(self):
        from shortcut_policy.unet_shortcut.train_unet_image_shortcut_policy import make_train_dataloader

        weighted = make_dataset([str(self.root_a), str(self.root_b)], source_weights=[.7, .3])
        loader = make_train_dataloader(weighted, OmegaConf.create(
            {"batch_size": 4, "shuffle": True, "num_workers": 0}))
        self.assertIsInstance(loader.sampler, WeightedRandomSampler)
        self.assertEqual(loader.sampler.num_samples, len(weighted))
        torch.manual_seed(0)
        draws = np.concatenate([list(iter(loader.sampler)) for _ in range(300)])
        self.assertAlmostEqual((weighted.window_source[draws] == 1).mean(), .3, delta=.02)
        plain = make_dataset([str(self.root_a), str(self.root_b)])
        loader = make_train_dataloader(plain, OmegaConf.create(
            {"batch_size": 4, "shuffle": True, "num_workers": 0}))
        self.assertIsInstance(loader.sampler, RandomSampler)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["source"].shape), (4,))

    def test_normalizer_coverage_reports_each_source(self):
        from shortcut_policy.unet_shortcut.train_unet_image_shortcut_policy import normalized_action_coverage

        dataset = make_dataset([str(self.root_a), str(self.root_b)])
        # Fit on root_a only: root_b's faster hand motion leaves [-1, 1].
        coverage = normalized_action_coverage(make_dataset(str(self.root_a)).get_normalizer(), dataset)
        self.assertEqual(set(coverage), {"all", "root_a", "root_b"})
        self.assertEqual(coverage["root_a"], 0.)
        self.assertGreater(coverage["root_b"], 0.)
        self.assertGreater(coverage["all"], 0.)
        refit = normalized_action_coverage(dataset.get_normalizer(), dataset)
        self.assertEqual(max(refit.values()), 0.)
        actions, sources = dataset.relative_actions(max_windows=5, seed=1)
        self.assertEqual(actions.shape, (5, 4, 19))
        self.assertEqual(len(sources), 5)

    def test_split_batch_by_source(self):
        from shortcut_policy.unet_shortcut.train_unet_image_shortcut_policy import split_batch_by_source

        batch = make_batch(5)
        batch["source"] = torch.tensor([0, 1, 0, 1, 1])
        subsets = {source: (count, subset) for source, count, subset in split_batch_by_source(batch)}
        self.assertEqual({s: c for s, (c, _) in subsets.items()}, {0: 2, 1: 3})
        torch.testing.assert_close(subsets[0][1]["action"], batch["action"][[0, 2]])
        torch.testing.assert_close(subsets[1][1]["obs"]["camera_0"], batch["obs"]["camera_0"][[1, 3, 4]])

    def test_init_weights_load_model_and_ema_only(self):
        from shortcut_policy.unet_shortcut.train_unet_image_shortcut_policy import load_init_weights

        model, ema = make_policy(), make_policy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.ckpt"
            torch.save({"cfg": None, "pickles": {},
                        "state_dicts": {"model": model.state_dict(), "ema_model": ema.state_dict(),
                                        "optimizer": {"unused": 1}}}, path, pickle_module=dill)
            target, target_ema = make_policy(), make_policy()
            target.normalizer = LinearNormalizer()  # a fresh policy has an empty normalizer
            load_init_weights(target, target_ema, path)
            for loaded, expected in ((target, model), (target_ema, ema)):
                state = loaded.state_dict()
                self.assertEqual(set(state), set(expected.state_dict()))
                for key, value in expected.state_dict().items():
                    torch.testing.assert_close(state[key], value)
            torch.save({"cfg": None, "pickles": {}, "state_dicts": {"model": model.state_dict()}},
                       path, pickle_module=dill)
            load_init_weights(target, target_ema, path)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(target_ema.state_dict()[key], value)

    def test_workspace_warm_start_keeps_normalizer_and_logs_source_losses(self):
        from shortcut_policy.unet_shortcut import train_unet_image_shortcut_policy as train
        import matplotlib.pyplot as plt

        root = Path(__file__).resolve().parents[1]

        def config(output):
            with initialize_config_dir(version_base=None, config_dir=str(root / "shortcut_policy/config")):
                cfg = compose(config_name="train_unet_image_trimi_hand_cond_dinov2_shortcut_policy")
            OmegaConf.set_struct(cfg, False)
            cfg.output_dir = output
            cfg.device = "cpu"
            cfg.policy = {"_target_": "test_shortcut_policy.make_policy"}
            cfg.tasks.dataset = {"_target_": "test_dagger_dataset.ToySourceDataset", "dataset_path": output}
            cfg.tasks.env_runner = {"_target_": "test_dagger_dataset.ToyRunner", "output_dir": output}
            cfg.dataloader = {"batch_size": 4, "shuffle": True}
            cfg.val_dataloader = {"batch_size": 4}
            cfg.training.num_epochs = 1
            cfg.training.lr_warmup_steps = 0
            cfg.training.val_every = cfg.training.sample_every = 1
            cfg.logging.mode = "disabled"
            cfg.checkpoint.topk.k = 0
            return cfg

        self.assertIsNone(config("/tmp/unused").training.init_ckpt)
        self.assertFalse(config("/tmp/unused").training.keep_ckpt_normalizer)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            cfg = config(first)
            cfg.training.keep_ckpt_normalizer = True
            with self.assertRaisesRegex(ValueError, "keep_ckpt_normalizer"):
                train.TrainDiffusionUnetImageWorkspace(cfg).run()
            cfg.training.keep_ckpt_normalizer = False
            previous = train.TrainDiffusionUnetImageWorkspace(cfg)
            previous.run()
            checkpoint = previous.get_checkpoint_path()
            saved = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
            saved_normalizer = {k: v for k, v in saved["state_dicts"]["model"].items()
                                if k.startswith("normalizer.")}

            cfg = config(second)
            cfg.training.init_ckpt = str(checkpoint)
            cfg.training.keep_ckpt_normalizer = True
            workspace = train.TrainDiffusionUnetImageWorkspace(cfg)
            with patch.object(train, "WeightedRandomSampler", wraps=WeightedRandomSampler) as sampler:
                workspace.run()
            sampler.assert_called_once()
            self.assertEqual(workspace.optimizer_step, 3)  # fresh schedule: ceil(10/4), 1 epoch
            for model in (workspace.model, workspace.ema_model):
                state = model.state_dict()
                for key, value in saved_normalizer.items():
                    torch.testing.assert_close(state[key].cpu(), value)
            logs = [json.loads(line) for line in (Path(second) / "logs.json.txt").read_text().splitlines()]
            final = logs[-1]
            self.assertIn("val_loss", final)
            self.assertIn("val_loss/toy_a", final)
            self.assertIn("val_loss/toy_b", final)
            # A task observation the checkpoint's normalizer never saw is refused up front.
            with tempfile.TemporaryDirectory() as third:
                cfg = config(third)
                cfg.training.init_ckpt = str(checkpoint)
                cfg.training.keep_ckpt_normalizer = True
                cfg.shape_meta = {"obs": {"camera_0": {}, "wrist_force": {}}, "action": {"shape": [3]}}
                with self.assertRaisesRegex(ValueError, "wrist_force"):
                    train.TrainDiffusionUnetImageWorkspace(cfg).run()
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
