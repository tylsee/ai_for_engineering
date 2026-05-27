"""
Faster R-CNN with EfficientNet-B0 backbone for defect detection.
Uses ImageNet-pretrained EfficientNet-B0 features as the backbone
and adds a region proposal + RoI-align detection head.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.ops as ops
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

NUM_CLASSES = 6  # 5 defect classes + background (index 0)


class _EfficientNetBackbone(nn.Module):
    """Wraps EfficientNet-B0 features to return an OrderedDict for FasterRCNN."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        eff = efficientnet_b0(weights=weights)
        self.features = eff.features
        self.out_channels = 1280   # EfficientNet-B0 last feature map channels

    def forward(self, x: torch.Tensor) -> OrderedDict:
        return OrderedDict([('0', self.features(x))])


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    backbone = _EfficientNetBackbone(pretrained=pretrained)

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )

    roi_pooler = ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )
    return model


def load_checkpoint(path: str, device: torch.device) -> nn.Module:
    model = build_model(pretrained=False)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model
