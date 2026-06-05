"""SSDLite320-MobileNetV3-Large detector for defect detection.

Supports custom anchor aspect ratios. The default torchvision anchors cover aspect
ratios {1/3 .. 3}, but the cleaned defect dataset has cracks/corrosion spanning
{0.19 .. 6.7} (5th–95th pct). Passing anchor_aspect_ratios=(2, 3, 5) widens the
generated boxes to {1/5 .. 5}, improving recall on long/narrow defects.
"""
from functools import partial

import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssdlite import SSDLiteHead


def _reinit_from_scratch(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def build_model(num_classes=6, pretrained=True, anchor_aspect_ratios=None,
                size=(320, 320)):
    """
    SSDLite320 with MobileNetV3-Large backbone (full, non-reduced-tail architecture).
    pretrained=True loads ImageNet backbone weights; detection head is fresh.
    num_classes includes background (e.g. 5 defect classes + 1 background = 6).

    anchor_aspect_ratios: None -> torchvision default ({1/3..3}). A tuple like (2, 3, 5)
    rebuilds the anchor generator and the matching head so the model also predicts
    very tall/wide boxes ({1/5..5}) better suited to cracks and corrosion streaks.

    Always builds the full architecture (reduced_tail=False) so that checkpoints
    saved with pretrained=True can be loaded with pretrained=False.
    """
    # Always pass a non-None weights_backbone so torchvision uses the full
    # MobileNetV3-Large architecture (reduced_tail=False).
    weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
    )

    if anchor_aspect_ratios is not None:
        # Rebuild anchor generator with wider aspect ratios, and the head to match.
        out_channels = det_utils.retrieve_out_channels(model.backbone, size)
        num_maps = len(out_channels)
        ar = list(anchor_aspect_ratios)
        anchor_generator = DefaultBoxGenerator(
            [ar for _ in range(num_maps)], min_ratio=0.2, max_ratio=0.95,
        )
        num_anchors = anchor_generator.num_anchors_per_location()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        model.anchor_generator = anchor_generator
        model.head = SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer)

    if not pretrained:
        _reinit_from_scratch(model)

    return model
