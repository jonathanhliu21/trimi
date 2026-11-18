import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

import cv2
import dill
import hydra
import numpy as np
import torch

from shared.env.midas.se3_utils import relative_to_absolute


class MidasPolicyInference:
    """
    Lightweight MIDAS policy inference wrapper.

    Input observation is RGB, optionally plus current absolute 13D hand state
    when included in the checkpoint's shape_meta. The current absolute EEF pose
    is only used after sampling to convert relative actions to absolute commands.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        use_ema: bool = True,
        num_inference_steps: Optional[int] = None,
        action_mode: str = "chunk",
    ):
        if action_mode not in {"first", "chunk", "all"}:
            raise ValueError(
                f"Unsupported action_mode '{action_mode}'. Use one of: first, chunk, all."
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.action_mode = action_mode
        self.checkpoint_path = checkpoint_path

        with open(checkpoint_path, "rb") as f:
            payload = torch.load(f, pickle_module=dill)
        cfg = payload["cfg"]
        workspace_cls = hydra.utils.get_class(cfg._target_)
        # Standalone inference does not initialize HydraConfig. Some workspaces
        # (notably the shortcut-policy trainer) resolve output_dir during
        # construction, so give them an existing directory explicitly.
        workspace = workspace_cls(cfg, output_dir=str(Path(checkpoint_path).parent))
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model
        if use_ema and getattr(cfg.training, "use_ema", False):
            if workspace.ema_model is None:
                raise RuntimeError(
                    "Requested EMA policy, but checkpoint/workspace has no EMA model."
                )
            policy = workspace.ema_model

        policy.eval().to(self.device)
        if num_inference_steps is not None:
            policy.num_inference_steps = int(num_inference_steps)

        self.workspace = workspace
        self.cfg = cfg
        self.policy = policy

        self.shape_meta = self._resolve_shape_meta(cfg)
        self._validate_shape_meta(self.shape_meta)

        self.rgb_key = "camera_0"
        self.hand_pose_key = (
            "hand_pose_abs" if "hand_pose_abs" in self.shape_meta["obs"] else None
        )
        self.expected_chw = tuple(self.shape_meta["obs"][self.rgb_key]["shape"])
        self.action_dim = int(self.shape_meta["action"]["shape"][0])
        self.num_obs_steps = int(policy.num_obs_steps)

        self._img_history: Deque[np.ndarray] = deque(maxlen=self.num_obs_steps)

    def reset(self) -> None:
        self._img_history.clear()
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def predict(self, image: np.ndarray, robot_state: np.ndarray) -> Dict[str, object]:
        """
        Run one inference step.

        Args:
            image: RGB image in HWC format.
            robot_state: Absolute current [eef_6, hand_13] state with shape (19,).

        Returns:
            Dict with absolute action outputs and metadata. The policy predicts
            relative [eef_6, hand_13] actions; this wrapper converts them back
            to absolute commands using robot_state.
        """
        robot_state = np.asarray(robot_state, dtype=np.float32).reshape(-1)
        if robot_state.shape[0] != self.action_dim:
            raise ValueError(
                f"robot_state must have shape ({self.action_dim},), got {robot_state.shape}."
            )

        t0 = time.time()
        obs_torch = self._prepare_obs_torch(image=image, robot_state=robot_state)

        with torch.no_grad():
            result = self.policy.predict_action(obs_torch)
        latency = time.time() - t0

        rel_action_chunk = result["action"][0].detach().cpu().numpy().astype(np.float32)
        rel_action_pred = result["action_pred"][0].detach().cpu().numpy().astype(
            np.float32
        )

        action_chunk = relative_to_absolute(rel_action_chunk, robot_state)
        action_pred = relative_to_absolute(rel_action_pred, robot_state)

        output: Dict[str, object] = {
            "latency_sec": float(latency),
            "device": str(self.device),
            "num_inference_steps": int(self.policy.num_inference_steps),
        }

        if self.action_mode == "first":
            output["action"] = action_chunk[0].astype(np.float32)
        elif self.action_mode == "chunk":
            output["action"] = action_chunk
            output["action_chunk"] = action_chunk
        else:
            output["action"] = action_chunk[0].astype(np.float32)
            output["action_chunk"] = action_chunk
            output["action_pred"] = action_pred

        return output

    @staticmethod
    def _resolve_shape_meta(cfg) -> dict:
        if hasattr(cfg, "shape_meta"):
            return cfg.shape_meta
        if hasattr(cfg, "tasks") and hasattr(cfg.tasks, "shape_meta"):
            return cfg.tasks.shape_meta
        if hasattr(cfg, "task") and hasattr(cfg.task, "shape_meta"):
            return cfg.task.shape_meta
        raise ValueError(
            "Could not find shape_meta in cfg. Checked cfg.shape_meta, "
            "cfg.tasks.shape_meta, cfg.task.shape_meta."
        )

    @staticmethod
    def _validate_shape_meta(shape_meta: dict) -> None:
        obs_meta = shape_meta.get("obs", {})
        if "camera_0" not in obs_meta:
            raise ValueError("shape_meta.obs must contain key 'camera_0'.")
        if "robot_eef_pose" in obs_meta:
            raise ValueError(
                "MIDAS inference expects no absolute EEF pose observation condition."
            )

        cam_shape = tuple(obs_meta["camera_0"].get("shape", ()))
        if len(cam_shape) != 3 or cam_shape[0] != 3:
            raise ValueError(
                f"Expected camera_0 shape [3,H,W], got {obs_meta['camera_0'].get('shape')}."
            )

        if "hand_pose_abs" in obs_meta:
            hand_meta = obs_meta["hand_pose_abs"]
            hand_shape = tuple(hand_meta.get("shape", ()))
            hand_type = hand_meta.get("type", "low_dim")
            if hand_type != "low_dim" or hand_shape != (13,):
                raise ValueError(
                    "Expected hand_pose_abs to be low_dim with shape [13], "
                    f"got type={hand_type}, shape={hand_meta.get('shape')}."
                )

        action_shape = tuple(shape_meta.get("action", {}).get("shape", ()))
        if action_shape != (19,):
            raise ValueError(
                "Expected action shape [19] for MIDAS, "
                f"got {shape_meta.get('action', {}).get('shape')}."
            )

    def _prepare_obs_torch(
        self, image: np.ndarray, robot_state: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        img = self._prepare_image(image)
        self._img_history.append(img)

        while len(self._img_history) < self.num_obs_steps:
            self._img_history.appendleft(self._img_history[0].copy())

        img_stack = np.stack(list(self._img_history), axis=0)
        img_stack = np.moveaxis(img_stack, -1, 1).astype(np.float32)

        obs_np = {
            self.rgb_key: img_stack[None],
        }
        if self.hand_pose_key is not None:
            hand_pose = robot_state[6:].astype(np.float32)
            if hand_pose.shape[0] != 13:
                raise ValueError(f"Expected 13D MIDAS hand state, got {hand_pose.shape}.")
            hand_pose_hist = np.repeat(
                hand_pose[None], repeats=self.num_obs_steps, axis=0
            )
            obs_np[self.hand_pose_key] = hand_pose_hist[None]

        return {
            key: torch.from_numpy(value).to(self.device, non_blocking=True)
            for key, value in obs_np.items()
        }

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array.")
        if image.ndim != 3:
            raise ValueError(
                f"Expected HWC image (3D array), got shape {image.shape}."
            )
        if image.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel RGB image in HWC format, got shape {image.shape}."
            )

        expected_h = int(self.expected_chw[1])
        expected_w = int(self.expected_chw[2])
        if image.shape[0] != expected_h or image.shape[1] != expected_w:
            image = cv2.resize(
                image, (expected_w, expected_h), interpolation=cv2.INTER_AREA
            )

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
            if image.min() < 0.0 or image.max() > 1.0:
                raise ValueError(
                    "Float image must be in [0,1], "
                    f"got min={image.min()} max={image.max()}."
                )
        return image
