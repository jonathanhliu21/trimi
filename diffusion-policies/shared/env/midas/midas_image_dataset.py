import copy
import hashlib
import json
import os
import shutil
import warnings
from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import zarr
from filelock import FileLock
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits
from torch.utils.data import Dataset

from shared.env.midas.se3_utils import absolute_to_relative
from shared.models.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from shared.utils.pytorch_util import dict_apply
from shared.utils.replay_buffer import ReplayBuffer
from shared.utils.sampler import SequenceSampler, downsample_mask, get_val_mask


def _resolve_midas_root(dataset_path: str) -> str:
    """
    Accept either:
      - <dataset_path>/camera, <dataset_path>/syncs, ...
      - <dataset_path>/data/camera, <dataset_path>/data/syncs, ...
    """
    direct_candidates = [
        os.path.join(dataset_path, "camera"),
        os.path.join(dataset_path, "processed_encoder_t"),
        os.path.join(dataset_path, "absolute_action_t"),
        os.path.join(dataset_path, "syncs"),
    ]
    if all(os.path.isdir(p) for p in direct_candidates):
        return dataset_path

    nested_root = os.path.join(dataset_path, "data")
    nested_candidates = [
        os.path.join(nested_root, "camera"),
        os.path.join(nested_root, "processed_encoder_t"),
        os.path.join(nested_root, "absolute_action_t"),
        os.path.join(nested_root, "syncs"),
    ]
    if all(os.path.isdir(p) for p in nested_candidates):
        return nested_root

    return dataset_path


def _dataset_roots(dataset_path) -> List[str]:
    """One dataset root (str / PathLike) or several (list, tuple, Hydra ListConfig)."""
    if isinstance(dataset_path, (str, os.PathLike)):
        return [os.fspath(dataset_path)]
    if OmegaConf.is_config(dataset_path):
        dataset_path = OmegaConf.to_container(dataset_path, resolve=True)
    roots = [os.fspath(path) for path in dataset_path]
    if not roots:
        raise ValueError("dataset_path must name at least one dataset root")
    return roots


def _resolve_source_names(roots: List[str], source_names=None) -> List[str]:
    """Display names of the sources (one per root), used in logs and source_weights dicts."""
    if source_names is not None:
        if OmegaConf.is_config(source_names):
            source_names = OmegaConf.to_container(source_names, resolve=True)
        names = [str(name) for name in source_names]
        if len(names) != len(roots):
            raise ValueError(
                f"source_names has {len(names)} entries for {len(roots)} dataset roots"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"source_names must be unique, got {names}")
        return names
    names = [os.path.basename(os.path.normpath(root)) or root for root in roots]
    if len(set(names)) != len(names):
        # Repeated basenames (e.g. round_1/data, round_2/data): number every source.
        names = [f"{index}_{name}" for index, name in enumerate(names)]
    return names


def _per_source_val_mask(
    episode_source: np.ndarray, n_sources: int, val_ratio: float, seed: int
) -> np.ndarray:
    """Validation episodes drawn separately inside every source.

    Each source's mask is get_val_mask over that source's own episodes with the
    same seed, so a source with >= 2 episodes always gets validation episodes,
    and a root keeps the same validation episodes whether it is loaded alone or
    with other roots. With one source this is exactly get_val_mask over all
    episodes, as before multi-root support.
    """
    val_mask = np.zeros(len(episode_source), dtype=bool)
    for source in range(n_sources):
        episodes = np.flatnonzero(episode_source == source)
        if len(episodes) == 0:
            continue
        val_mask[episodes] = get_val_mask(
            n_episodes=len(episodes), val_ratio=val_ratio, seed=seed
        )
    return val_mask


# Per-step flag in the replay buffer: False where the sync's optional train_mask
# excludes the row (DAgger exports: a kinesthetic takeover's handoff frames).
# Windows touching such a row are never sampled.
TRAIN_ROW_KEY = "train_row"
# Cache contents version; 2 added TRAIN_ROW_KEY.
REPLAY_BUFFER_FORMAT = 2


def _drop_untrainable_windows(
    sampler: SequenceSampler, replay_buffer: ReplayBuffer
) -> SequenceSampler:
    """Remove the sampler's windows whose buffer rows include an excluded row."""
    if len(sampler.indices) == 0 or TRAIN_ROW_KEY not in replay_buffer.keys():
        return sampler
    excluded = np.concatenate(
        [[0], np.cumsum(~replay_buffer[TRAIN_ROW_KEY][:].astype(bool))]
    )
    starts, ends = sampler.indices[:, 0], sampler.indices[:, 1]
    sampler.indices = sampler.indices[excluded[ends] == excluded[starts]]
    return sampler


def _window_sources(
    sampler: SequenceSampler, episode_ends: np.ndarray, episode_source: np.ndarray
) -> np.ndarray:
    """Source id of every sampler window, from the episode of its first buffer step."""
    if len(sampler.indices) == 0:
        return np.zeros(0, dtype=np.int64)
    episodes = np.searchsorted(
        np.asarray(episode_ends), sampler.indices[:, 0], side="right"
    )
    return np.asarray(episode_source, dtype=np.int64)[episodes]


def _source_window_weights(
    window_source: np.ndarray,
    n_sources: int,
    source_weights,
    source_names: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    """Per-window sampling weights that give each source its requested share of draws.

    weight(window) = w_source / n_windows_in_source, normalized to sum to 1, so a
    source's share of the draws is w_source / sum(w) regardless of its size.
    ``source_weights`` is a list aligned with the sources or a {name: weight}
    mapping (missing names get 0). None keeps uniform sampling over windows.
    """
    if source_weights is None:
        return None
    if OmegaConf.is_config(source_weights):
        source_weights = OmegaConf.to_container(source_weights, resolve=True)
    names = list(source_names) if source_names is not None else [str(s) for s in range(n_sources)]
    if isinstance(source_weights, Mapping):
        unknown = sorted(set(map(str, source_weights)) - set(names))
        if unknown:
            raise ValueError(f"source_weights names {unknown} are not sources {names}")
        weights = [float(source_weights.get(name, 0.0)) for name in names]
    else:
        weights = [float(weight) for weight in source_weights]
    weights = np.asarray(weights, dtype=np.float64)
    if (weights.shape != (n_sources,) or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError(
            f"source_weights must be {n_sources} finite, nonnegative weights with a "
            f"positive sum, got {source_weights}"
        )
    window_source = np.asarray(window_source, dtype=np.int64)
    counts = np.bincount(window_source, minlength=n_sources)
    per_window = np.zeros(len(window_source), dtype=np.float64)
    for source in range(n_sources):
        if counts[source] == 0:
            if weights[source] > 0:
                warnings.warn(
                    f"[MidasImageDataset] source {names[source]!r} has no training "
                    f"windows; its weight {weights[source]:g} gets no sampling mass"
                )
            continue
        per_window[window_source == source] = weights[source] / counts[source]
    total = per_window.sum()
    if total <= 0:
        raise ValueError("source_weights give no training window any sampling mass")
    return per_window / total


def _load_root_replay_buffer(
    dataset_path: str,
    shape_meta: dict,
    first_cut: int,
    encoder_window: int,
    sync_downsample_stride: int,
    use_cache: bool,
    replay_buffer: Optional[ReplayBuffer] = None,
) -> ReplayBuffer:
    """Load one dataset root, appending its episodes to ``replay_buffer`` when given.

    The optional cache is per root (``<root>/midas_<hash>.zarr.zip``, keyed by the
    loader settings), so adding a root never rebuilds another root's cache and a
    single root keeps its cache file name.
    """
    if not use_cache:
        return _build_replay_buffer(
            dataset_path=dataset_path,
            shape_meta=shape_meta,
            first_cut=first_cut,
            encoder_window=encoder_window,
            sync_downsample_stride=sync_downsample_stride,
            store=zarr.MemoryStore(),
            replay_buffer=replay_buffer,
        )

    cache_key = {
        "shape_meta": OmegaConf.to_container(shape_meta),
        "first_cut": int(first_cut),
        "encoder_window": int(encoder_window),
        "sync_downsample_stride": sync_downsample_stride,
        "buffer_format": REPLAY_BUFFER_FORMAT,
    }
    cache_hash = hashlib.md5(
        json.dumps(cache_key, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache_zarr_path = os.path.join(
        dataset_path, f"midas_{cache_hash}.zarr.zip"
    )
    cache_lock_path = cache_zarr_path + ".lock"

    print("[MidasImageDataset] Acquiring cache lock")
    with FileLock(cache_lock_path):
        if not os.path.exists(cache_zarr_path):
            try:
                print("[MidasImageDataset] Cache missing, building replay buffer")
                root_buffer = _build_replay_buffer(
                    dataset_path=dataset_path,
                    shape_meta=shape_meta,
                    first_cut=first_cut,
                    encoder_window=encoder_window,
                    sync_downsample_stride=sync_downsample_stride,
                    store=zarr.MemoryStore(),
                )
                print("[MidasImageDataset] Saving cache")
                with zarr.ZipStore(cache_zarr_path) as zip_store:
                    root_buffer.save_to_store(store=zip_store)
            except Exception as exc:
                if os.path.exists(cache_zarr_path):
                    if os.path.isdir(cache_zarr_path):
                        shutil.rmtree(cache_zarr_path)
                    else:
                        os.remove(cache_zarr_path)
                raise exc
        else:
            print("[MidasImageDataset] Loading cached replay buffer")
            with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                root_buffer = ReplayBuffer.copy_from_store(
                    src_store=zip_store,
                    store=zarr.MemoryStore(),
                )
    if replay_buffer is None:
        return root_buffer
    for episode in range(root_buffer.n_episodes):
        replay_buffer.add_episode(root_buffer.get_episode(episode, copy=True))
    return replay_buffer


class MidasImageDataset(Dataset):
    """
    Midas image dataset with Merlin-compatible on-disk layout.

    Stored replay-buffer action is absolute [eef_6, hand_13]. Returned training
    action is window-relative [eef_rel_6, hand_delta_13].

    ``dataset_path`` may list several roots (for example the base demos and
    DAgger rounds). Their episodes share one replay buffer in root order;
    ``episode_source`` / ``window_source`` give each episode's / window's root
    index, validation episodes are drawn inside every root, and batches carry a
    ``source`` id when there is more than one root. ``source_weights`` (one per
    root, or {name: weight}) sets each root's share of the training draws
    through ``sample_weights``; None keeps uniform sampling over windows.

    A sync file may carry ``train_mask`` (one bool per sync row; DAgger full
    rollouts set it False on kinesthetic handoff frames). Training and validation
    windows that touch a False row are dropped; without it every row trains.
    """

    def __init__(
        self,
        shape_meta: dict,
        dataset_path,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        num_obs_steps: int = None,
        num_latency_steps: int = 0,
        use_cache: bool = False,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int = None,
        first_cut: int = 0,
        encoder_window: int = 0,
        sync_downsample_stride: int = 1,
        source_weights=None,
        source_names=None,
    ):
        super().__init__()

        roots = _dataset_roots(dataset_path)
        for root in roots:
            assert os.path.isdir(root), f"Dataset path does not exist: {root}"
        sync_downsample_stride = int(sync_downsample_stride)
        if sync_downsample_stride < 1:
            raise ValueError(
                "sync_downsample_stride must be >= 1, "
                f"got {sync_downsample_stride}."
            )
        names = _resolve_source_names(roots, source_names)

        replay_buffer = None
        episode_source = []
        for source, root in enumerate(roots):
            before = 0 if replay_buffer is None else replay_buffer.n_episodes
            replay_buffer = _load_root_replay_buffer(
                dataset_path=root,
                shape_meta=shape_meta,
                first_cut=first_cut,
                encoder_window=encoder_window,
                sync_downsample_stride=sync_downsample_stride,
                use_cache=use_cache,
                replay_buffer=replay_buffer,
            )
            episode_source.extend([source] * (replay_buffer.n_episodes - before))
        episode_source = np.asarray(episode_source, dtype=np.int64)

        rgb_keys = []
        lowdim_keys = []
        for key, attr in shape_meta["obs"].items():
            data_type = attr.get("type", "low_dim")
            if data_type == "rgb":
                rgb_keys.append(key)
            elif data_type == "low_dim":
                lowdim_keys.append(key)
            else:
                raise ValueError(f"Unsupported obs type '{data_type}' for key '{key}'")

        key_first_k = {}
        if num_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = num_obs_steps

        # Per-source split; with one root this is get_val_mask over all episodes.
        val_mask = _per_source_val_mask(
            episode_source=episode_source,
            n_sources=len(roots),
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed,
        )
        sample_keys = [*rgb_keys, *lowdim_keys, "action"]
        sampler = _drop_untrainable_windows(
            SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=horizon + num_latency_steps,
                pad_before=pad_before,
                pad_after=pad_after,
                keys=sample_keys,
                episode_mask=train_mask,
                key_first_k=key_first_k,
            ),
            replay_buffer,
        )

        self.replay_buffer = replay_buffer
        self.sample_keys = sample_keys
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.num_obs_steps = num_obs_steps
        self.val_mask = val_mask
        self.train_mask = train_mask
        self.horizon = horizon
        self.num_latency_steps = num_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.first_cut = int(first_cut)
        self.encoder_window = int(encoder_window)
        self.sync_downsample_stride = sync_downsample_stride
        self.dataset_roots = roots
        self.source_names = names
        self.n_sources = len(roots)
        self.episode_source = episode_source
        self.window_source = _window_sources(
            sampler, replay_buffer.episode_ends[:], episode_source
        )
        # Training draws only; validation stays uniform over its windows.
        self.sample_weights = _source_window_weights(
            self.window_source, self.n_sources, source_weights, names
        )

        print(
            "[MidasImageDataset] Replay buffer: "
            f"episodes={self.replay_buffer.n_episodes}, steps={self.replay_buffer.n_steps}"
        )
        if self.n_sources > 1:
            for source, name in enumerate(names):
                in_source = episode_source == source
                share = ""
                if self.sample_weights is not None:
                    mass = self.sample_weights[self.window_source == source].sum()
                    share = f", draw share={mass:.3f}"
                print(
                    f"[MidasImageDataset] source {source} {name!r}: "
                    f"episodes={int(in_source.sum())} "
                    f"(val {int((in_source & val_mask).sum())}), "
                    f"train windows={int((self.window_source == source).sum())}{share}"
                )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = _drop_untrainable_windows(
            SequenceSampler(
                replay_buffer=self.replay_buffer,
                sequence_length=self.horizon + self.num_latency_steps,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                keys=self.sample_keys,
                episode_mask=self.val_mask,
            ),
            self.replay_buffer,
        )
        val_set.val_mask = ~self.val_mask
        val_set.window_source = _window_sources(
            val_set.sampler, self.replay_buffer.episode_ends[:], self.episode_source
        )
        val_set.sample_weights = None
        return val_set

    def relative_actions(
        self, max_windows: Optional[int] = None, seed: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Window-relative training targets [N, horizon, 19] and their window sources.

        Reads only the action array (no video frames) for the training windows,
        or a seeded random subset of at most ``max_windows`` of them.
        """
        sampler = _drop_untrainable_windows(
            SequenceSampler(
                replay_buffer=self.replay_buffer,
                sequence_length=self.horizon + self.num_latency_steps,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                keys=["action"],
                episode_mask=self.train_mask,
            ),
            self.replay_buffer,
        )
        indices = np.arange(len(sampler))
        if max_windows is not None and len(indices) > max_windows:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(len(indices), size=int(max_windows), replace=False))
        actions = []
        for idx in indices:
            abs_action = sampler.sample_sequence(idx)["action"].astype(np.float32)
            if self.num_latency_steps > 0:
                abs_action = abs_action[self.num_latency_steps :]
            actions.append(absolute_to_relative(abs_action))
        if not actions:
            action_dim = int(self.replay_buffer["action"].shape[-1])
            return np.zeros((0, self.horizon, action_dim), dtype=np.float32), np.zeros(0, np.int64)
        return np.stack(actions).astype(np.float32), self.window_source[indices]

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        print("[MidasImageDataset] Computing relative action normalization stats...")
        all_relative = []
        for idx in range(len(self.sampler)):
            data = self.sampler.sample_sequence(idx)
            abs_action = data["action"].astype(np.float32)
            if self.num_latency_steps > 0:
                abs_action = abs_action[self.num_latency_steps :]
            all_relative.append(absolute_to_relative(abs_action))
        all_relative = np.concatenate(all_relative, axis=0)
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(all_relative)

        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key]
            )
        for key in self.rgb_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_identity()

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        obs_t_slice = slice(self.num_obs_steps)
        obs_dict = {}

        for key in self.rgb_keys:
            obs_dict[key] = (
                np.moveaxis(data[key][obs_t_slice], -1, 1).astype(np.float32) / 255.0
            )
            del data[key]

        for key in self.lowdim_keys:
            lowdim_obs = data[key][obs_t_slice].astype(np.float32)
            if key == "hand_pose_abs" and lowdim_obs.shape[0] > 0:
                lowdim_obs = np.repeat(
                    lowdim_obs[[-1]], repeats=lowdim_obs.shape[0], axis=0
                )
            obs_dict[key] = lowdim_obs
            del data[key]

        abs_action = data["action"].astype(np.float32)
        if self.num_latency_steps > 0:
            print("[MidasImageDataset] hit self.num_latency_steps > 0")
            abs_action = abs_action[self.num_latency_steps :]
        action = absolute_to_relative(abs_action)

        result = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action),
        }
        if self.n_sources > 1:
            # Root index of this window, for per-source validation losses.
            result["source"] = torch.tensor(int(self.window_source[idx]), dtype=torch.int64)
        return result


def _sorted_files(directory: str, suffix: str) -> List[str]:
    files = [f for f in os.listdir(directory) if f.endswith(suffix)]
    files.sort()
    return [os.path.join(directory, f) for f in files]


def _rolling_average(data: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return data.astype(np.float32)

    padded = np.pad(data, ((window, window), (0, 0)), mode="edge")
    kernel_size = window * 2 + 1
    smoothed = np.zeros_like(data, dtype=np.float32)

    for dim in range(data.shape[1]):
        col = padded[:, dim].astype(np.float64)
        cumsum = np.cumsum(col, dtype=np.float64)
        cumsum[kernel_size:] = cumsum[kernel_size:] - cumsum[:-kernel_size]
        smoothed[:, dim] = (
            cumsum[kernel_size - 1 : kernel_size - 1 + len(data)] / kernel_size
        )

    return smoothed.astype(np.float32)


def _load_video_rgb(video_path: str, out_hw: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    frames = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[0] != out_h or frame.shape[1] != out_w:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    cap.release()

    if not frames:
        print("[MidasImageDataset] frames failed to load in _load_video_rgb(...)")
        return np.zeros((0, out_h, out_w, 3), dtype=np.uint8)

    return np.asarray(frames, dtype=np.uint8)


def _build_replay_buffer(
    dataset_path: str,
    shape_meta: dict,
    first_cut: int,
    encoder_window: int,
    sync_downsample_stride: int,
    store,
    replay_buffer: Optional[ReplayBuffer] = None,
) -> ReplayBuffer:
    """Load one root's episodes into a new buffer, or append them to ``replay_buffer``."""
    obs_shape_meta = shape_meta["obs"]

    rgb_keys = [
        key
        for key, value in obs_shape_meta.items()
        if value.get("type", "low_dim") == "rgb"
    ]
    lowdim_keys = [
        key
        for key, value in obs_shape_meta.items()
        if value.get("type", "low_dim") == "low_dim"
    ]

    if len(rgb_keys) != 1:
        raise ValueError(
            f"MidasImageDataset expects exactly 1 RGB obs key, got {len(rgb_keys)}"
        )

    rgb_key = rgb_keys[0]
    rgb_shape = tuple(obs_shape_meta[rgb_key]["shape"])
    if len(rgb_shape) != 3 or rgb_shape[0] != 3:
        raise ValueError(
            f"RGB obs shape must be [3, H, W], got {rgb_shape} for key '{rgb_key}'"
        )
    out_h, out_w = rgb_shape[1], rgb_shape[2]

    for key in lowdim_keys:
        shape = tuple(obs_shape_meta[key].get("shape", ()))
        if key == "hand_pose_abs":
            if shape != (13,):
                raise ValueError(
                    f"Low-dim obs '{key}' must have shape [13], got {shape}"
                )
        else:
            raise ValueError(
                f"Unsupported low-dim obs key '{key}' for MIDAS dataset conversion. "
                "Supported keys: hand_pose_abs."
            )

    action_shape = tuple(shape_meta["action"]["shape"])
    if action_shape != (19,):
        raise ValueError(
            f"Action shape must be [19] for MIDAS (6 EEF + 13 hand), got {action_shape}"
        )

    midas_root = _resolve_midas_root(dataset_path)

    video_dir = os.path.join(midas_root, "camera", "mp4_files")
    encoder_dir = os.path.join(midas_root, "processed_encoder_t")
    action_dir = os.path.join(midas_root, "absolute_action_t")
    sync_dir = os.path.join(midas_root, "syncs")

    for required_dir in [video_dir, encoder_dir, action_dir, sync_dir]:
        if not os.path.isdir(required_dir):
            raise FileNotFoundError(
                f"Required MIDAS directory not found: {required_dir}"
            )

    vid_files = _sorted_files(video_dir, ".mp4")
    enc_files = _sorted_files(encoder_dir, ".npy")
    act_files = _sorted_files(action_dir, ".npy")
    sync_files = _sorted_files(sync_dir, ".npz")

    n = len(vid_files)
    if not (len(enc_files) == n and len(act_files) == n and len(sync_files) == n):
        raise RuntimeError(
            "MIDAS file count mismatch after sorting: "
            f"video={len(vid_files)} encoder={len(enc_files)} "
            f"action={len(act_files)} sync={len(sync_files)}"
        )

    if replay_buffer is None:
        replay_buffer = ReplayBuffer.create_empty_zarr(storage=store)

    kept_episodes = 0
    dropped_episodes = 0
    total_steps = 0

    for i in range(n):
        raw_hand = np.load(enc_files[i]).astype(np.float32)
        raw_eef = np.load(act_files[i]).astype(np.float32)
        sync_data = np.load(sync_files[i])

        if raw_hand.ndim != 2 or raw_hand.shape[1] != 13:
            raise ValueError(
                "MIDAS hand file must have shape [T, 13], "
                f"got {raw_hand.shape}: {enc_files[i]}"
            )
        if raw_eef.ndim != 2 or raw_eef.shape[1] != 6:
            raise ValueError(
                "MIDAS EEF file must have shape [T, 6], "
                f"got {raw_eef.shape}: {act_files[i]}"
            )

        frame_idx = sync_data["frame_idx"].astype(np.int64)
        encoder_idx = sync_data["encoder_idx"].astype(np.int64)
        pose_idx = sync_data["pose_idx"].astype(np.int64)
        # Optional: DAgger exports mark kinesthetic handoff frames False; every row trains otherwise.
        train_row = (
            sync_data["train_mask"].astype(bool)
            if "train_mask" in sync_data
            else np.ones(len(frame_idx), dtype=bool)
        )

        if not (len(frame_idx) == len(encoder_idx) == len(pose_idx) == len(train_row)):
            print(
                f"[MidasImageDataset] Dropping episode {i}: sync array length mismatch"
            )
            dropped_episodes += 1
            continue

        if len(frame_idx) <= first_cut:
            dropped_episodes += 1
            continue

        frame_idx = frame_idx[first_cut:]
        encoder_idx = encoder_idx[first_cut:]
        pose_idx = pose_idx[first_cut:]
        train_row = train_row[first_cut:]
        if sync_downsample_stride > 1:
            frame_idx = frame_idx[::sync_downsample_stride]
            encoder_idx = encoder_idx[::sync_downsample_stride]
            pose_idx = pose_idx[::sync_downsample_stride]
            train_row = train_row[::sync_downsample_stride]

        if len(frame_idx) == 0:
            dropped_episodes += 1
            continue

        smoothed_hand = _rolling_average(raw_hand, window=encoder_window)
        video_frames = _load_video_rgb(vid_files[i], out_hw=(out_h, out_w))

        if len(video_frames) == 0:
            print(f"[MidasImageDataset] Dropping episode {i}: video has no frames")
            dropped_episodes += 1
            continue

        if len(raw_eef) == 0 or len(smoothed_hand) == 0:
            print(f"[MidasImageDataset] Dropping episode {i}: empty EEF or hand array")
            dropped_episodes += 1
            continue

        max_frame = len(video_frames) - 1
        max_pose = len(raw_eef) - 1
        max_hand = len(smoothed_hand) - 1

        frame_idx = np.clip(frame_idx, 0, max_frame)
        pose_idx = np.clip(pose_idx, 0, max_pose)
        encoder_idx = np.clip(encoder_idx, 0, max_hand)

        episode_images = video_frames[frame_idx]
        episode_eef = raw_eef[pose_idx].astype(np.float32)
        episode_hand = smoothed_hand[encoder_idx].astype(np.float32)
        episode_state = np.concatenate([episode_eef, episode_hand], axis=-1).astype(
            np.float32
        )

        episode = {
            rgb_key: episode_images.astype(np.uint8),
            "action": episode_state,
            # Every episode has it, so roots with and without train_mask share one buffer.
            TRAIN_ROW_KEY: train_row,
        }
        if "hand_pose_abs" in lowdim_keys:
            episode["hand_pose_abs"] = episode_hand.astype(np.float32)

        replay_buffer.add_episode(episode)
        kept_episodes += 1
        total_steps += len(episode_state)

    if kept_episodes == 0:
        raise RuntimeError(f"No usable episodes found in MIDAS dataset {dataset_path}")

    print(
        "[MidasImageDataset] Built replay buffer: "
        f"kept={kept_episodes}, dropped={dropped_episodes}, total_steps={total_steps}"
    )

    return replay_buffer
