from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.models.common.normalizer import LinearNormalizer
from shared.models.components.positional_embedding import SinusoidalPosEmb
from shared.models.unet.conditional_unet1d import ConditionalUnet1D
from shared.vision.common.multi_image_obs_encoder import ObsEncoder
from shared.utils.pytorch_util import dict_apply
from shortcut_policy.shortcut_model import ShortcutModel


class UnetImageShortcutPolicy(nn.Module):
    """Observation-conditioned trajectory flow with optional shortcut training."""

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: ObsEncoder,
        horizon: int,
        num_action_steps: int,
        num_obs_steps: int,
        num_inference_steps: int = 128,
        global_obs_cond: bool = True,
        embed_dim_D: int = 256,
        down_dims: Tuple[int] = (256, 512, 1024),
        kernel_size: int = 5,
        num_groups: int = 8,
        film_modulation_scale: bool = True,
        max_train_steps: int = 128,
        shortcut_fraction: float = 0.25,
        **kwargs,
    ):
        super().__init__()
        if not global_obs_cond:
            raise NotImplementedError("Shortcut policy supports global_obs_cond=True only")
        if not 0 <= shortcut_fraction < 1:
            raise ValueError("shortcut_fraction must be in [0, 1)")
        if (not 1 <= num_obs_steps <= horizon
                or not 1 <= num_action_steps <= horizon - num_obs_steps + 1):
            raise ValueError("Observation and executed action horizons must fit horizon")
        if horizon % (2 ** (len(down_dims) - 1)):
            raise ValueError("horizon must be divisible by the U-Net downsampling factor")

        action_dim_Fa = shape_meta["action"]["shape"][0]
        obs_feat_dim_Fo = obs_encoder.output_shape()[0]
        self.distance_encoder = nn.Sequential(
            SinusoidalPosEmb(embed_dim_D),
            nn.Linear(embed_dim_D, embed_dim_D * 4),
            nn.Mish(),
            nn.Linear(embed_dim_D * 4, embed_dim_D),
        )
        self.model = ConditionalUnet1D(
            inp_dim_F=action_dim_Fa,
            cond_dim_L=None,
            cond_dim_G=obs_feat_dim_Fo * num_obs_steps + embed_dim_D,
            embed_dim_D=embed_dim_D,
            down_dims=down_dims,
            kernel_size=kernel_size,
            num_groups=num_groups,
            film_modulation_scale=film_modulation_scale,
        )
        self.obs_encoder = obs_encoder
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feat_dim_Fo = obs_feat_dim_Fo
        self.action_dim_Fa = action_dim_Fa
        self.num_action_steps = num_action_steps
        self.num_obs_steps = num_obs_steps
        self.num_inference_steps = num_inference_steps
        self.global_obs_cond = global_obs_cond
        self.max_train_steps = max_train_steps
        self.shortcut_fraction = shortcut_fraction
        self.shortcut_model = ShortcutModel(
            self.forward_model, num_inference_steps=num_inference_steps,
            max_train_steps=max_train_steps,
        )

    def forward_model(self, z, t, distance=None, cond_BTL=None, cond_BG=None):
        t = torch.as_tensor(t, device=z.device, dtype=z.dtype).reshape(-1).expand(z.shape[0])
        distance = torch.as_tensor(
            0.0 if distance is None else distance, device=z.device, dtype=z.dtype
        ).reshape(-1).expand(z.shape[0])
        # Scale continuous [0,1] inputs to a useful sinusoidal frequency range.
        distance_feat = self.distance_encoder(distance * self.max_train_steps)
        cond = torch.cat([cond_BG, distance_feat], dim=-1)
        return self.model(
            sample_BTF=z, timesteps_B=t * self.max_train_steps,
            cond_BTL=cond_BTL, cond_BG=cond,
        )

    def reset(self):
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _encode_obs(self, obs):
        normalized_obs = self.normalizer.normalize(obs)
        batch_size = next(iter(normalized_obs.values())).shape[0]
        if any(x.shape[1] < self.num_obs_steps for x in normalized_obs.values()):
            raise ValueError("Insufficient observation history")
        flattened = dict_apply(
            normalized_obs,
            lambda x: x[:, :self.num_obs_steps].reshape(-1, *x.shape[2:]),
        )
        return self.obs_encoder(flattened).reshape(batch_size, -1)

    @torch.no_grad()
    def predict_action(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.predict_action_shortcut(obs)

    @torch.no_grad()
    def predict_action_shortcut(self, obs, num_inference_steps=None, *, sampler=None):
        steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        cond_BG = self._encode_obs(obs)
        z0 = torch.randn(
            (cond_BG.shape[0], self.horizon, self.action_dim_Fa),
            dtype=self.dtype, device=self.device,
        )
        sample = self.shortcut_model.sample_ode_shortcut if sampler is None else sampler
        trajectory = sample(
            z0=z0, num_inference_steps=steps, cond_BG=cond_BG,
            flow_only=self.shortcut_fraction == 0,
        )
        action_pred = self.normalizer["action"].unnormalize(trajectory[-1])
        start = self.num_obs_steps - 1
        return {"action": action_pred[:, start:start + self.num_action_steps],
                "action_pred": action_pred}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        cond_BG = self._encode_obs(batch["obs"])
        z1 = self.normalizer["action"].normalize(batch["action"])
        batch_size = z1.shape[0]
        # Keep a flow-matching anchor even for very small batches. B=1 uses FM.
        num_shortcut = 0
        if self.shortcut_fraction:
            num_shortcut = min(
                batch_size - 1, max(1, int(batch_size * self.shortcut_fraction))
            )
        num_flow = batch_size - num_shortcut
        z0 = torch.randn_like(z1)
        flow = self.shortcut_model.get_train_tuple(z0[:num_flow], z1[:num_flow])
        if num_shortcut:
            shortcut = self.shortcut_model.get_shortcut_train_tuple(
                z0[num_flow:], z1[num_flow:], cond_BG=cond_BG[num_flow:].detach(),
            )
            z_t, t, target, distance = [torch.cat(parts) for parts in zip(flow, shortcut)]
        else:
            z_t, t, target, distance = flow
        prediction = self.forward_model(z_t, t, distance=distance, cond_BG=cond_BG)
        return F.mse_loss(prediction, target)
