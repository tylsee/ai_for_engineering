"""SSDLite320-MobileNetV3-Large detector for defect detection."""
from functools import partial
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights


def build_model(num_classes=6, pretrained=True):
    """
    SSDLite320 with MobileNetV3-Large backbone (full, non-reduced-tail architecture).
    pretrained=True loads ImageNet backbone weights; detection head is fresh.
    num_classes includes background (e.g. 5 defect classes + 1 background = 6).

    Always builds the full architecture (reduced_tail=False) so that checkpoints
    saved with pretrained=True can be loaded with pretrained=False.
    """
    # Always pass a non-None weights_backbone so torchvision uses the full
    # MobileNetV3-Large architecture (reduced_tail=False).  When pretrained=False
    # we immediately re-initialise all weights, discarding the backbone values.
    weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
    )
    if not pretrained:
        import torch.nn as nn
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    return model
