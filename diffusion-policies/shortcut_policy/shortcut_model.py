import math

import torch


class ShortcutModel:
    """Linear flow matching with binary self-consistency shortcuts.

    ``model(z, t, distance=...)`` predicts velocity (displacement / step size).
    The finest teacher steps query distance=0, anchoring shortcuts to flow matching.
    """

    def __init__(self, model=None, num_inference_steps=128, device=None,
                 max_train_steps=128):
        if max_train_steps < 2 or max_train_steps & (max_train_steps - 1):
            raise ValueError("max_train_steps must be a power of two >= 2")
        self.model = model
        self.num_inference_steps = num_inference_steps
        self.max_train_steps = max_train_steps
        # Device is deliberately taken from each input, including after policy.to().

    def get_train_tuple(self, z0=None, z1=None):
        batch_size = z0.shape[0]
        t = torch.rand(batch_size, device=z0.device, dtype=z0.dtype)
        t_view = t.view(batch_size, *([1] * (z0.ndim - 1)))
        z_t = (1 - t_view) * z0 + t_view * z1
        return z_t, t, z1 - z0, torch.zeros_like(t)

    @torch.no_grad()
    def get_shortcut_train_tuple(self, z0=None, z1=None, **model_kwargs):
        batch_size = z0.shape[0]
        # Half-step sizes: 1/2, 1/4, ..., 1/M. Student lengths: 1, ..., 2/M.
        exponent = torch.randint(1, int(math.log2(self.max_train_steps)) + 1,
                                 (batch_size,), device=z0.device)
        half_step = torch.pow(2.0, -exponent).to(z0.dtype)
        distance = 2 * half_step
        # Only query grid-aligned starts with t + distance <= 1.
        num_starts = (1 / distance).round().long()
        t = (torch.rand(batch_size, device=z0.device) * num_starts).floor() * distance
        view_shape = (batch_size,) + (1,) * (z0.ndim - 1)
        t_view = t.view(view_shape)
        z_t = (1 - t_view) * z0 + t_view * z1
        teacher_distance = torch.where(
            half_step <= 1 / self.max_train_steps, 0.0, half_step
        )
        first = self.model(z_t, t, distance=teacher_distance, **model_kwargs)
        midpoint = z_t + first * half_step.view(view_shape)
        second = self.model(midpoint, t + half_step,
                            distance=teacher_distance, **model_kwargs)
        return z_t, t, (first + second) / 2, distance

    @torch.no_grad()
    def sample_ode_shortcut(self, z0=None, num_inference_steps=None,
                            flow_only=False, **model_kwargs):
        steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("num_inference_steps must be a positive integer")
        if not flow_only and (steps > self.max_train_steps or steps & (steps - 1)):
            raise ValueError("Shortcut inference needs a power of two <= max_train_steps")
        dt = 1.0 / steps
        # At the finest resolution, use the directly supervised flow field.
        distance = 0.0 if flow_only or steps == self.max_train_steps else dt
        z = z0.clone()
        traj = [z.clone()]
        for i in range(steps):
            t = z.new_full((z.shape[0],), i * dt)
            pred = self.model(z, t, distance=distance, **model_kwargs)
            z = z + pred * dt
            traj.append(z.clone())
        return traj

    @torch.no_grad()
    def sample_2step_shortcut(self, z0=None, **model_kwargs):
        return self.sample_ode_shortcut(z0, num_inference_steps=2, **model_kwargs)
