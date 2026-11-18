"""Run with: python -m unittest discover -s tests -v"""
import copy
from pathlib import Path
import unittest
import tempfile

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch import nn

from shortcut_policy.shortcut_model import ShortcutModel
from shortcut_policy.unet_shortcut.unet_image_shortcut_policy import UnetImageShortcutPolicy
from shared.models.common.normalizer import LinearNormalizer


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 4)

    def output_shape(self):
        return (6,)

    def forward(self, obs):
        return torch.cat([self.proj(obs['camera_0'].mean((-2, -1))), obs['hand_pose_abs']], -1)


class ImageOnlyToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 4)

    def output_shape(self):
        return (4,)

    def forward(self, obs):
        return self.proj(obs['camera_0'].mean((-2, -1)))


def make_policy(shortcut_fraction=0.25, image_only=False):
    obs_encoder = ImageOnlyToyEncoder() if image_only else ToyEncoder()
    policy = UnetImageShortcutPolicy(
        shape_meta={'action': {'shape': [3]}}, obs_encoder=obs_encoder,
        horizon=8, num_action_steps=4, num_obs_steps=1, num_inference_steps=4,
        embed_dim_D=8, down_dims=(8, 16), kernel_size=3, num_groups=8,
        shortcut_fraction=shortcut_fraction,
    )
    normalizer = LinearNormalizer()
    normalizer_data = {
        'camera_0': torch.rand(8, 3, 4, 4),
        'action': torch.randn(8, 3),
    }
    if not image_only:
        normalizer_data['hand_pose_abs'] = torch.randn(8, 2)
    normalizer.fit(normalizer_data)
    policy.set_normalizer(normalizer)
    return policy


def make_batch(batch_size):
    return {'obs': {'camera_0': torch.rand(batch_size, 1, 3, 4, 4),
                    'hand_pose_abs': torch.randn(batch_size, 1, 2)},
            'action': torch.randn(batch_size, 8, 3)}


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path=None):
        self.batch = make_batch(10)

    def __len__(self):
        return 10

    def __getitem__(self, index):
        return {'obs': {k: v[index] for k, v in self.batch['obs'].items()},
                'action': self.batch['action'][index]}

    def get_normalizer(self):
        return make_policy().normalizer

    def get_validation_dataset(self):
        return self


class ToyRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy):
        return {}


class ShortcutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_flow_interpolation_and_batch_one(self):
        z0, z1 = torch.randn(1, 8, 3), torch.randn(1, 8, 3)
        z, t, target, d = ShortcutModel().get_train_tuple(z0, z1)
        self.assertEqual(t.shape, (1,))
        torch.testing.assert_close(z, (1-t[:, None, None])*z0+t[:, None, None]*z1)
        torch.testing.assert_close(target, z1-z0)
        self.assertEqual(d.item(), 0)

    def test_bootstrap_equation_boundaries_and_anchor(self):
        calls = []
        def field(z, t, distance):
            self.assertFalse(torch.is_grad_enabled())
            calls.append((z.clone(), t.clone(), distance.clone()))
            return z + t[:, None, None] + distance[:, None, None]
        wrapper = ShortcutModel(field)
        z0, z1 = torch.randn(4096, 4, 2), torch.randn(4096, 4, 2)
        z, t, target, d = wrapper.get_shortcut_train_tuple(z0, z1)
        self.assertTrue(torch.all(t+d <= 1))
        self.assertEqual(set(d.tolist()), {1., .5, .25, .125, .0625, .03125, .015625})
        torch.testing.assert_close(t/d, (t/d).round())
        first = z + t[:, None, None] + calls[0][2][:, None, None]
        midpoint = z + first*d[:, None, None]/2
        torch.testing.assert_close(calls[1][0], midpoint)
        torch.testing.assert_close(calls[1][1], t+d/2)
        second = midpoint + (t+d/2)[:, None, None] + calls[1][2][:, None, None]
        torch.testing.assert_close(target, (first+second)/2)
        self.assertTrue(torch.all(calls[0][2][d == 1/64] == 0))
        self.assertFalse(target.requires_grad)
        # Singleton bootstrap must retain vector times.
        self.assertEqual(wrapper.get_shortcut_train_tuple(z0[:1], z1[:1])[1].shape, (1,))

    def test_integrator_and_step_validation(self):
        calls = []
        def field(z, t, distance):
            calls.append((t, distance))
            return torch.ones_like(z)*2
        wrapper = ShortcutModel(field, device='cuda')
        z = torch.zeros(1, 8, 3)
        for steps in (1, 2, 4, 16, 128):
            calls.clear()
            traj = wrapper.sample_ode_shortcut(z, steps)
            self.assertEqual(len(calls), steps)
            torch.testing.assert_close(traj[-1], z+2)
            self.assertEqual(calls[0][1], 0 if steps == 128 else 1/steps)
        for steps in (0, -1, 3, 256):
            with self.assertRaises(ValueError):
                wrapper.sample_ode_shortcut(z, steps)
        wrapper.sample_ode_shortcut(z, 3, flow_only=True)
        self.assertEqual(calls[-1][1], 0)

    def test_shortcut_target_matches_two_step_endpoint(self):
        # Equation (4): one target step must reproduce the endpoint of two
        # half steps, even when the field depends on state, time and distance.
        for resolution in (2, 8, 128):
            calls = []

            def field(z, t, distance):
                velocity = z.square() * 0.1 + t[:, None, None] + distance[:, None, None]
                calls.append((z.clone(), velocity.clone(), distance.clone()))
                return velocity

            wrapper = ShortcutModel(field, max_train_steps=resolution)
            z, t, target, step = wrapper.get_shortcut_train_tuple(
                torch.randn(256, 8, 3), torch.randn(256, 8, 3)
            )
            half = step[:, None, None] / 2
            teacher_endpoint = calls[1][0] + half * calls[1][1]
            student_target_endpoint = z + step[:, None, None] * target
            torch.testing.assert_close(student_target_endpoint, teacher_endpoint)
            self.assertTrue(torch.all(t + step <= 1))
            self.assertTrue(torch.all(step >= 2 / resolution))
            self.assertTrue(torch.all(calls[0][2][step == 2 / resolution] == 0))

    def test_bootstrap_gradient_only_uses_student_branch(self):
        # Analytical semi-gradient: d/dw mean((w*z - stopgrad(target))^2).
        weight = nn.Parameter(torch.tensor(0.7))
        teacher_condition = torch.randn(8, 1, 1, requires_grad=True)

        def field(z, t, distance, condition):
            return weight * z + condition + t[:, None, None]

        wrapper = ShortcutModel(field)
        z, t, target, step = wrapper.get_shortcut_train_tuple(
            torch.randn(8, 8, 3), torch.randn(8, 8, 3),
            condition=teacher_condition,
        )
        self.assertFalse(target.requires_grad)
        student_condition = teacher_condition.detach().clone().requires_grad_(True)
        prediction = field(z, t, step, student_condition)
        expected_weight_gradient = (2 * (prediction.detach() - target) * z).mean()
        torch.nn.functional.mse_loss(prediction, target).backward()
        torch.testing.assert_close(weight.grad, expected_weight_gradient)
        self.assertIsNone(teacher_condition.grad)
        self.assertGreater(student_condition.grad.abs().sum().item(), 0)

    def test_step_conditioning_and_backward(self):
        policy = make_policy()
        z, t, cond = torch.randn(2, 8, 3), torch.zeros(2), torch.randn(2, 6)
        a = policy.forward_model(z, t, distance=0., cond_BG=cond)
        for other_t, other_distance, other_obs in (
            (t, .5, cond), (t + .5, 0., cond), (t, 0., cond + 1),
        ):
            b = policy.forward_model(z, other_t, distance=other_distance, cond_BG=other_obs)
            self.assertFalse(torch.allclose(a, b))
        for batch_size in (1, 4):
            policy.zero_grad()
            loss = policy.compute_loss(make_batch(batch_size))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for module in (policy.model, policy.distance_encoder, policy.obs_encoder):
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))

    def test_prediction_honors_steps_and_trimi_keys(self):
        policy = make_policy().eval()
        original = policy.shortcut_model.model
        calls = []
        def recording_model(*args, **kwargs):
            calls.append(kwargs['distance'])
            return original(*args, **kwargs)
        policy.shortcut_model.model = recording_model
        obs = make_batch(1)['obs']
        for steps in (1, 2, 4):
            calls.clear()
            policy.num_inference_steps = steps
            result = policy.predict_action(obs)
            self.assertEqual(len(calls), steps)
            self.assertEqual(result['action'].shape, (1, 4, 3))
            self.assertEqual(result['action_pred'].shape, (1, 8, 3))
            self.assertFalse(result['action'].requires_grad)
        policy = make_policy(shortcut_fraction=0)
        self.assertTrue(torch.isfinite(policy.compute_loss(make_batch(4))))
        policy.predict_action_shortcut(obs, 3)

    def test_rgb_only_flow_matching_trains_without_hand_pose(self):
        policy = make_policy(shortcut_fraction=0, image_only=True)
        batch = {
            'obs': {'camera_0': torch.rand(4, 1, 3, 4, 4)},
            'action': torch.randn(4, 8, 3),
        }

        loss = policy.compute_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in policy.obs_encoder.parameters()
        ))

        result = policy.eval().predict_action(batch['obs'])
        self.assertEqual(result['action'].shape, (4, 4, 3))
        self.assertEqual(result['action_pred'].shape, (4, 8, 3))

    def test_deepcopy_wrapper_uses_copied_policy(self):
        policy = make_policy()
        ema = copy.deepcopy(policy)
        self.assertIs(ema.shortcut_model.model.__self__, ema)
        self.assertIsNot(ema.model, policy.model)
        ema.to(dtype=torch.float64)
        result = ema.predict_action({k: v.double() for k, v in make_batch(1)['obs'].items()})
        self.assertEqual(result['action'].dtype, torch.float64)

    def test_config_training_matches_diffusion_reference(self):
        root = Path(__file__).resolve().parents[1]
        OmegaConf.register_new_resolver('eval', eval, replace=True)
        with initialize_config_dir(version_base=None, config_dir=str(root/'diffusion_policy/config')):
            reference = compose(config_name='train_unet_image_trimi_hand_cond_dinov2_policy')
        with initialize_config_dir(version_base=None, config_dir=str(root/'shortcut_policy/config')):
            shortcut = compose(config_name='train_unet_image_trimi_hand_cond_dinov2_shortcut_policy')
        # The independent tasks may select different sensors/data paths. Compare
        # training settings with the same task rather than coupling those choices.
        reference.tasks = copy.deepcopy(shortcut.tasks)
        # Compose API does not install Hydra's runtime output-dir context.
        reference.tasks.env_runner.output_dir = '/tmp/test'
        shortcut.tasks.env_runner.output_dir = '/tmp/test'
        for key in ('tasks', 'optimizer', 'ema', 'dataloader', 'val_dataloader'):
            self.assertEqual(OmegaConf.to_container(reference[key], resolve=True),
                             OmegaConf.to_container(shortcut[key], resolve=True))
        self.assertEqual(reference.policy.obs_encoder, shortcut.policy.obs_encoder)
        self.assertEqual(shortcut.policy.num_action_steps, 8)
        self.assertEqual(shortcut.policy.num_inference_steps, 16)
        self.assertEqual(shortcut.training.num_epochs, 500)
        self.assertNotIn('noise_scheduler', shortcut.policy)
        # No dependency on the diffusion config search path or parent configs.
        primary = OmegaConf.load(root/'shortcut_policy/config/train_unet_image_trimi_hand_cond_dinov2_shortcut_policy.yaml')
        self.assertEqual(list(primary.defaults), ['_self_', {'tasks': 'trimi_image_hand_cond'}])
        self.assertNotIn('searchpath', primary.hydra)
        task = OmegaConf.load(root/'shortcut_policy/config/tasks/trimi_image_hand_cond.yaml')
        self.assertNotIn('defaults', task)
        self.assertNotIn('hand_pose_abs', task.shape_meta.obs)

    def test_workspace_training_validation_and_accumulation(self):
        from shortcut_policy.unet_shortcut.train_unet_image_shortcut_policy import TrainDiffusionUnetImageWorkspace
        import matplotlib.pyplot as plt
        root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(version_base=None, config_dir=str(root/'shortcut_policy/config')):
            cfg = compose(config_name='train_unet_image_trimi_hand_cond_dinov2_shortcut_policy')
        with tempfile.TemporaryDirectory() as output:
            OmegaConf.set_struct(cfg, False)
            cfg.output_dir = output
            cfg.device = 'cpu'
            cfg.policy = {'_target_': 'test_shortcut_policy.make_policy'}
            cfg.tasks.dataset = {'_target_': 'test_shortcut_policy.ToyDataset',
                                 'dataset_path': output}
            cfg.tasks.env_runner = {'_target_': 'test_shortcut_policy.ToyRunner',
                                    'output_dir': output}
            cfg.dataloader = {'batch_size': 2}
            cfg.val_dataloader = {'batch_size': 2}
            cfg.training.num_epochs = 2
            cfg.training.gradient_accumulate_every = 2
            cfg.training.lr_warmup_steps = 0
            cfg.training.val_every = cfg.training.sample_every = 1
            cfg.logging.mode = 'disabled'
            cfg.checkpoint.save_last_ckpt = False
            cfg.checkpoint.topk.k = 0
            workspace = TrainDiffusionUnetImageWorkspace(cfg)
            workspace.run()
            self.assertEqual(workspace.optimizer_step, 6)  # ceil(5/2) * 2
            self.assertTrue(workspace.model.training)
            self.assertFalse(workspace.ema_model.training)
            self.assertIsNone(workspace.ema_model.model.final_projection[1].weight.grad)
            checkpoint = workspace.save_checkpoint(use_thread=False)
            restored = TrainDiffusionUnetImageWorkspace(cfg)
            restored.load_checkpoint(checkpoint, weights_only=False)
            self.assertEqual(restored.optimizer_step, workspace.optimizer_step)
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
