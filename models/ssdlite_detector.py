"""SSDLite320-MobileNetV3-Large detector for defect detection."""
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights


def build_model(num_classes=6, pretrained=True):
    """
    SSDLite320 with MobileNetV3-Large backbone.
    pretrained=True loads ImageNet backbone weights; detection head is fresh.
    num_classes includes background (e.g. 5 defect classes + 1 background = 6).
    """
    weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
    )
    return model
