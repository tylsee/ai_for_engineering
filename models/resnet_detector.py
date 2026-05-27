"""
Faster R-CNN with ResNet-50 FPN backbone for defect detection.
Loads COCO-pretrained weights and replaces the classification head
with a new head for our 5 defect classes.
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_CLASSES = 6  # 5 defect classes + background (index 0)


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_checkpoint(path: str, device: torch.device) -> nn.Module:
    model = build_model(pretrained=False)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model
