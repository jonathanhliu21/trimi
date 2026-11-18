"""Real-time chunking guidance (inference/predictor/rtc.py). The sampling tests need torch and diffusers
(the GPU predictor environment) and are skipped without them."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from inference.predictor.rtc import guidance_weights

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_DIFFUSERS = HAS_TORCH and importlib.util.find_spec("diffusers") is not None
T = 16


class GuidanceWeightTests(unittest.TestCase):
    def test_weights_follow_the_paper_soft_mask(self):
        weights = guidance_weights(T, overlap=8, frozen=3)
        np.testing.assert_array_equal(weights[:3], 1.0)
        np.testing.assert_array_equal(weights[8:], 0.0)
        c = (8 - 3) / (8 - 3 + 1)
        self.assertAlmostEqual(weights[3], c * (np.e ** c - 1) / (np.e - 1))
        self.assertTrue(np.all(np.diff(weights[2:9]) < 0))
        np.testing.assert_array_equal(guidance_weights(T, 0, 0), 0.0)
        np.testing.assert_array_equal(guidance_weights(T, 4, 4)[:4], 1.0)
        with self.assertRaises(ValueError):
            guidance_weights(T, overlap=4, frozen=5)


def gp_policy():
    """A 'policy' whose denoiser is exact for a smooth Gaussian-process prior over 16-step chunks,
    with the deployed checkpoint's DDIM settings."""
    import torch
    from diffusers import DDIMScheduler

    steps = np.arange(T)
    prior = 0.3 ** 2 * np.exp(-0.5 * (steps[:, None] - steps[None, :]) ** 2 / 4.0 ** 2) + 1e-6 * np.eye(T)
    covariance = torch.tensor(prior)
    scheduler = DDIMScheduler(num_train_timesteps=50, beta_start=1e-4, beta_end=0.02,
                              beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")

    class Policy(torch.nn.Module):
        horizon, num_inference_steps, global_obs_cond, kwargs = T, 16, True, {}
        noise_scheduler, device, dtype = scheduler, torch.device("cpu"), torch.float64
        normalizer = {"action": SimpleNamespace(normalize=lambda x: x)}

        def model(self, sample_BTF, timesteps_B, cond_BTL=None, cond_BG=None):
            alpha_bar = scheduler.alphas_cumprod[timesteps_B].to(sample_BTF)
            a, s = alpha_bar.sqrt(), (1 - alpha_bar).sqrt()
            gain = a * covariance @ torch.linalg.inv(a ** 2 * covariance + s ** 2 * torch.eye(T, dtype=torch.float64))
            return (sample_BTF - a * torch.einsum("ij,bjf->bif", gain, sample_BTF)) / s

        def conditional_sample(self, cond_data_BTF, cond_mask_BTF, cond_BTL=None, cond_BG=None, **kwargs):
            trajectory = torch.randn(cond_data_BTF.shape, dtype=torch.float64)
            scheduler.set_timesteps(self.num_inference_steps)
            for t in scheduler.timesteps:
                trajectory = scheduler.step(self.model(trajectory, t), t, trajectory).prev_sample
            return trajectory

        def predict_action(self, obs):
            cond = torch.zeros(obs["batch"], T, obs["dim"], dtype=torch.float64)
            sample = self.conditional_sample(cond, torch.zeros_like(cond, dtype=torch.bool))
            return {"action": sample[:, :8], "action_pred": sample}

    return Policy(), np.linalg.cholesky(prior)


@unittest.skipUnless(HAS_DIFFUSERS, "needs torch and diffusers")
class GuidedSamplingTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.manual_seed(0)

    def test_guided_chunk_continues_the_previous_chunk_at_the_takeover(self):
        import torch
        from inference.predictor.rtc import guided_sample

        policy, cholesky = gp_policy()
        rng = np.random.default_rng(1)
        delay, gaps = 2, {}
        for beta in (0.0, 20.0):
            torch.manual_seed(0)
            gap = []
            for _ in range(5):
                prev = cholesky @ rng.normal(size=T)  # previous chunk from new index 0 on
                weights = torch.tensor(guidance_weights(T, 8, delay + 1))[None, :, None]
                cond = torch.zeros(100, T, 1, dtype=torch.float64)
                x = guided_sample(policy, torch.tensor(prev)[None, :, None], weights, beta, cond,
                                  torch.zeros_like(cond, dtype=torch.bool))[..., 0].numpy()
                gap.append(np.abs(x[:, delay + 1] - prev[delay + 1]).mean())  # first new action executed
            gaps[beta] = np.mean(gap)
        typical_step = 0.3 * np.sqrt(2 * (1 - np.exp(-1 / 32)))
        self.assertLess(gaps[20.0], 1.5 * typical_step)
        self.assertLess(gaps[20.0], 0.4 * gaps[0.0])

    def test_chunker_returns_full_horizon_and_restores_the_policy_sampler(self):
        from inference.predictor.rtc import RealTimeChunker

        policy, _ = gp_policy()
        wrapper = SimpleNamespace(policy=policy,
                                  _prepare_obs_torch=lambda image, robot_state: {"batch": 1, "dim": 19})
        chunker = RealTimeChunker(wrapper, to_relative=lambda actions, state: actions - state,
                                  to_absolute=lambda relative, state: relative + state)
        state = np.full(19, 0.5, dtype=np.float32)
        prev = state + 0.05 * np.arange(8, dtype=np.float32)[:, None]  # moving 0.05 per step
        chunk = chunker.predict(None, state, prev, delay=2, max_guidance=20.0)
        self.assertEqual(chunk.shape, (T, 19))
        self.assertNotIn("conditional_sample", vars(policy))
        # Continues the previous chunk where it takes over (index 3) ...
        self.assertLess(np.abs(chunk[1:4] - prev[1:4]).mean(), 0.05)
        # ... and is still a policy sample beyond the overlap, not a collapsed one.
        self.assertGreater(np.ptp(chunk[8:], axis=0).mean(), 0.05)
        unguided = chunker.predict(None, state, prev[:0], delay=0, max_guidance=20.0)
        self.assertEqual(unguided.shape, (T, 19))
        self.assertGreater(np.abs(unguided[1:4] - prev[1:4]).mean(), 0.05)


@unittest.skipUnless(HAS_TORCH, "needs torch and the sibling diffusion-policies checkout")
class FlowGuidanceTests(unittest.TestCase):
    def make_chunker(self, shortcut_fraction):
        import torch
        from torch import nn
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'diffusion-policies'))
        from shortcut_policy.unet_shortcut.unet_image_shortcut_policy import UnetImageShortcutPolicy
        from shared.models.common.normalizer import LinearNormalizer
        from inference.predictor.rtc import RealTimeChunker

        class Encoder(nn.Module):
            def output_shape(self):
                return (3,)

            def forward(self, obs):
                return obs['camera_0'].mean((-2, -1))

        torch.set_num_threads(1)
        policy = UnetImageShortcutPolicy(
            shape_meta={'action': {'shape': [19]}}, obs_encoder=Encoder(),
            horizon=16, num_action_steps=8, num_obs_steps=1,
            num_inference_steps=16, embed_dim_D=8, down_dims=(8, 16),
            kernel_size=3, num_groups=8, shortcut_fraction=shortcut_fraction)
        normalizer = LinearNormalizer()
        # Nonidentity action normalization verifies that guidance targets and
        # predictions use the same normalized space.
        normalizer.fit({'action': torch.linspace(-2, 3, 80)[:, None].repeat(1, 19),
                        'camera_0': torch.rand(8, 3, 4, 4)})
        policy.set_normalizer(normalizer)
        distances = []

        def velocity(z, t, distance=None, **kwargs):
            distances.append(distance)
            return z * 0

        policy.shortcut_model.model = velocity
        wrapper = SimpleNamespace(policy=policy, _prepare_obs_torch=lambda **kwargs: {
            'camera_0': torch.zeros(1, 1, 3, 4, 4)})
        return RealTimeChunker(wrapper, lambda a, s: a - s, lambda a, s: a + s), distances

    def test_real_flow_policy_guides_overlap_and_keeps_unguided_suffix(self):
        import torch
        for fraction in (0.0, 0.25):
            with self.subTest(shortcut_fraction=fraction):
                chunker, distances = self.make_chunker(fraction)
                state = np.full(19, .4, dtype=np.float32)
                prev = state + np.arange(8, dtype=np.float32)[:, None] * .02
                torch.manual_seed(42)
                plain = chunker.predict(None, state, prev, 2, 0)
                torch.manual_seed(42)
                guided = chunker.predict(None, state, prev, 2, 5)
                self.assertEqual(guided.shape, (16, 19))
                self.assertTrue(np.isfinite(guided).all())
                self.assertLess(np.abs(guided[:4] - prev[:4]).mean(),
                                .2 * np.abs(plain[:4] - prev[:4]).mean())
                np.testing.assert_allclose(guided[8:], plain[8:], atol=1e-6)
                self.assertTrue(all(d == (0 if fraction == 0 else 1/16) for d in distances))
                self.assertFalse(any(p.grad is not None for p in chunker.policy.parameters()))
                self.assertNotIn('conditional_sample', vars(chunker.policy))
                torch.manual_seed(42)
                empty = chunker.predict(None, state, prev[:0], 0, 5)
                np.testing.assert_allclose(empty, plain, atol=1e-6)

    def test_flow_guidance_uses_endpoint_jacobian(self):
        import torch
        from inference.predictor.rtc import guided_flow_sample
        chunker, _ = self.make_chunker(0)
        policy = chunker.policy
        # v=-z gives endpoint t*z and Jacobian t*I, rather than I.
        policy.shortcut_model.model = lambda z, t, **kw: -z
        actual_sampler = policy.shortcut_model.sample_ode_shortcut
        def check_velocity(*, velocity_fn, **kwargs):
            z = torch.ones(1, 16, 19)
            result = velocity_fn(z, torch.tensor([.5]), distance=0)
            # endpoint=.5; correction=.5*(2-.5)=.75; coefficient=2.
            torch.testing.assert_close(result, torch.full_like(z, .5))
            self.assertFalse(result.requires_grad)
            return [result]
        policy.shortcut_model.sample_ode_shortcut = check_velocity
        try:
            guided_flow_sample(policy, torch.full((1, 16, 19), 2.),
                               torch.ones(1, 16, 1), 5)
        finally:
            policy.shortcut_model.sample_ode_shortcut = actual_sampler

    def test_guidance_runs_through_actual_unet_for_flow_and_shortcuts(self):
        import torch
        for fraction in (0.0, .25):
            with self.subTest(shortcut_fraction=fraction):
                chunker, _ = self.make_chunker(fraction)
                policy = chunker.policy
                policy.shortcut_model.model = policy.forward_model
                policy.num_inference_steps = 4
                state = np.zeros(19, dtype=np.float32)
                prev = np.zeros((8, 19), dtype=np.float32)
                torch.manual_seed(7)
                plain = chunker.predict(None, state, prev, 2, 0)
                torch.manual_seed(7)
                guided = chunker.predict(None, state, prev, 2, 2)
                self.assertTrue(np.isfinite(guided).all())
                self.assertGreater(np.abs(guided - plain).max(), 1e-4)
                self.assertTrue(all(p.grad is None for p in policy.parameters()))


if __name__ == "__main__":
    unittest.main()
