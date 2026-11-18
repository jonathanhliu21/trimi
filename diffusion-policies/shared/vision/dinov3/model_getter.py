"""DINOv3 ViT features with gradients preserved for policy fine-tuning."""
from pathlib import Path
from typing import Optional

import torch
from torch import nn


class DinoV3FeatureExtractor(nn.Module):
    """Expose CLS or mean patch features, excluding DINOv3 storage tokens."""

    def __init__(self, backbone: nn.Module, output: str = "cls",
                 proj_dim: Optional[int] = None):
        super().__init__()
        if output not in {"cls", "mean_pool"}:
            raise ValueError(f"Unsupported output type: {output}")
        self.backbone = backbone
        self.output = output
        self.proj = nn.Identity() if proj_dim is None else nn.LazyLinear(proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Do not use no_grad/detach here: the policy loss fine-tunes the backbone.
        features = self.backbone.forward_features(x)
        if self.output == "cls":
            features = features["x_norm_clstoken"]
        else:
            features = features["x_norm_patchtokens"].mean(dim=1)
        return self.proj(features)


def get_dinov3(
    name: str = "dinov3_vits16",
    pretrained: bool = True,
    weights: Optional[str] = None,
    output: str = "cls",
    proj_dim: Optional[int] = None,
    trainable: bool = True,
    repo: str = "facebookresearch/dinov3",
    source: str = "github",
    force_reload: bool = False,
):
    """Load Meta's DINOv3 ViT via Torch Hub.

    weights: Authorized checkpoint path or URL; required when pretrained=True.
    trainable: Whether backbone parameters receive gradients (default True).
    repo/source: GitHub repo (optionally pinned with :ref), or local checkout
        with source='local'. No transformers dependency is required.
    pretrained=False explicitly selects random initialization for smoke tests.
    """
    if output not in {"cls", "mean_pool"}:
        raise ValueError(f"Unsupported output type: {output}")
    if source not in {"github", "local"}:
        raise ValueError("source must be 'github' or 'local'")
    if not name.startswith("dinov3_vit"):
        raise ValueError("This feature extractor supports DINOv3 ViT backbones only")
    if pretrained and not weights:
        raise ValueError(
            "DINOv3 pretrained weights are required. Set "
            "policy.obs_encoder.vision_backbone.weights to a checkpoint path or URL, "
            "or set DINOV3_WEIGHTS. Use pretrained=False only for random initialization."
        )
    if source == "local":
        repo = str(Path(repo).expanduser().resolve())
    load_kwargs = {"pretrained": pretrained}
    if pretrained:
        load_kwargs["weights"] = str(weights)
    backbone = torch.hub.load(
        repo_or_dir=repo, model=name, source=source,
        force_reload=force_reload, **load_kwargs,
    )
    backbone.requires_grad_(trainable)
    return DinoV3FeatureExtractor(backbone, output=output, proj_dim=proj_dim)
