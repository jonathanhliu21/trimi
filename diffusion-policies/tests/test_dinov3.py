"""DINOv3 integration checks; no network or pretrained weights required."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from torch import nn

from shared.models.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from shared.vision.dinov3 import DinoV3FeatureExtractor, get_dinov3


class TinyDinoV3(nn.Module):
    """Match Meta's feature dictionary, including separate storage tokens."""
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 384)

    def forward_features(self, images):
        patches = torch.nn.functional.adaptive_avg_pool2d(images, (2, 2))
        patches = self.projection(patches.flatten(2).transpose(1, 2))
        return {'x_norm_clstoken': patches.mean(1),
                'x_norm_patchtokens': patches,
                'x_storage_tokens': patches.new_full((images.shape[0], 4, 384), 10000)}


class DinoV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_loader_weights_and_trainability(self):
        backbone = TinyDinoV3().requires_grad_(False)
        with patch('torch.hub.load', return_value=backbone) as hub:
            encoder = get_dinov3(weights='/tmp/dinov3.pth')
        self.assertTrue(all(p.requires_grad for p in encoder.backbone.parameters()))
        self.assertEqual(hub.call_args.kwargs['weights'], '/tmp/dinov3.pth')
        self.assertTrue(hub.call_args.kwargs['pretrained'])
        encoder(torch.randn(2, 3, 32, 32)).square().mean().backward()
        self.assertGreater(backbone.projection.weight.grad.abs().sum().item(), 0)
        with patch('torch.hub.load', return_value=TinyDinoV3()) as hub:
            frozen = get_dinov3(pretrained=False, trainable=False,
                                repo='/tmp/dinov3-source', source='local')
        self.assertFalse(any(p.requires_grad for p in frozen.backbone.parameters()))
        self.assertNotIn('weights', hub.call_args.kwargs)
        self.assertEqual(hub.call_args.kwargs['source'], 'local')
        with patch('torch.hub.load') as hub:
            with self.assertRaisesRegex(ValueError, 'pretrained weights are required'):
                get_dinov3()
            hub.assert_not_called()

    def test_pooling_projection_and_checkpoint(self):
        images = torch.randn(2, 3, 32, 32)
        for output in ('cls', 'mean_pool'):
            encoder = DinoV3FeatureExtractor(TinyDinoV3(), output=output)
            features = encoder.backbone.forward_features(images)
            expected = features['x_norm_clstoken'] if output == 'cls' else features['x_norm_patchtokens'].mean(1)
            torch.testing.assert_close(encoder(images), expected)
        encoder = DinoV3FeatureExtractor(TinyDinoV3(), proj_dim=16)
        self.assertEqual(encoder(images).shape, (2, 16))
        restored = copy.deepcopy(encoder)
        restored.load_state_dict(encoder.state_dict())
        torch.testing.assert_close(restored(images), encoder(images))

    def test_shortcut_config_backprop_updates_backbone(self):
        root = Path(__file__).resolve().parents[1]
        OmegaConf.register_new_resolver('eval', eval, replace=True)
        with initialize_config_dir(version_base=None, config_dir=str(root/'shortcut_policy/config')):
            cfg = compose(config_name='train_unet_image_trimi_hand_cond_dinov3_shortcut_policy',
                          overrides=['policy.obs_encoder.vision_backbone.weights=/tmp/dinov3.pth'])
        self.assertFalse(cfg.training.freeze_encoder)
        self.assertTrue(cfg.policy.obs_encoder.vision_backbone.trainable)
        self.assertEqual(cfg.policy.obs_encoder.vision_backbone.name, 'dinov3_vits16')
        self.assertEqual(cfg.tasks.shape_meta.action.shape, [19])
        cfg.policy.down_dims = [8, 16]
        cfg.policy.embed_dim_D = 8
        backbone = TinyDinoV3()
        with patch('torch.hub.load', return_value=backbone):
            policy = instantiate(cfg.policy)
        self.assertEqual(policy.obs_feat_dim_Fo, 397)
        normalizer = LinearNormalizer()
        normalizer.fit({'action': torch.randn(64, 19), 'hand_pose_abs': torch.randn(64, 13)})
        normalizer['camera_0'] = SingleFieldLinearNormalizer.create_identity()
        policy.set_normalizer(normalizer)
        optimizer = instantiate(cfg.optimizer, params=policy.parameters())
        batch = {'obs': {'camera_0': torch.rand(4, 1, 3, 240, 320),
                         'hand_pose_abs': torch.randn(4, 1, 13)},
                 'action': torch.randn(4, 16, 19)}
        before = backbone.projection.weight.detach().clone()
        loss = policy.compute_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(backbone.projection.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, backbone.projection.weight))
        self.assertIsNot(copy.deepcopy(policy).obs_encoder, policy.obs_encoder)


if __name__ == '__main__':
    unittest.main()
