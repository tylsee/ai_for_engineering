import os
import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from ultralytics import YOLO, RTDETR
from src.analyser import DefectAnalyzer

# Safely point to the models directory from ANY execution path
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
CONFIDENCE_THRESHOLD = 0.40

class DefectModelWrapper:
    def __init__(self):
        self.model = None
        self.model_type = None
        self.active_classes = {} 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pytorch_transform = T.Compose([T.ToTensor()])
        self.current_model_path = None

    def load_model(self, model_filename):
        path = os.path.join(MODELS_DIR, model_filename)
        print(f"Loading {model_filename} from {path}...")
        
        try:
            if "yolo" in model_filename.lower() or "rt_detr" in model_filename.lower():
                self.model_type = "ultralytics"
                self.model = YOLO(path) if "yolo" in model_filename.lower() else RTDETR(path)
                self.active_classes = self.model.names 
            
            elif "faster_rcnn" in model_filename.lower():
                self.model_type = "pytorch_frcnn"
                self.active_classes = {1: "corrosion", 2: "cracks", 3: "paint_degradation", 4: "potholes", 5: "spalling"}
                backbone = resnet_fpn_backbone(backbone_name='resnet18', weights=None, trainable_layers=3)
                self.model = FasterRCNN(backbone, num_classes=6)
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(self.device).eval()

            elif "efficientdet" in model_filename.lower():
                self.model_type = "pytorch_effdet"
                self.active_classes = {1: "corrosion", 2: "cracks", 3: "paint_degradation", 4: "potholes", 5: "spalling"}
                from effdet import create_model, DetBenchPredict
                
                # ---> FIXED: Changed num_classes=5 to num_classes=6 to match the checkpoint architecture <---
                net = create_model('tf_efficientdet_d0', pretrained=False, num_classes=6, image_size=(640, 640))
                
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
                elif isinstance(checkpoint, dict):
                    net.load_state_dict(checkpoint, strict=False)
                else:
                    net = checkpoint
                self.model = DetBenchPredict(net) if not isinstance(net, DetBenchPredict) else net
                self.model.to(self.device).eval()

            self.current_model_path = path
            return True
        except Exception as e:
            print(f"Load Error for {model_filename}: {e}")
            return False

    def predict(self, frame):
        if not self.model or not self.current_model_path:
            return []

        predictions = []
        orig_h, orig_w = frame.shape[:2]

        if self.model_type == "ultralytics":
            results = self.model(frame, verbose=False)[0] 
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls[0])
                    raw_name = self.active_classes.get(cls_id, f"Class {cls_id}")
                    std_name = DefectAnalyzer.normalize_class_name(raw_name)
                    predictions.append({"box": [x1, y1, x2, y2], "class": std_name, "conf": conf})

        elif self.model_type == "pytorch_frcnn":
            TARGET_SIZE = (640, 640) 
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, TARGET_SIZE)
            img_tensor = [self.pytorch_transform(frame_resized).to(self.device).float()]
            
            with torch.no_grad():
                outputs = self.model(img_tensor)[0]
            
            scale_x, scale_y = orig_w / TARGET_SIZE[0], orig_h / TARGET_SIZE[1]

            for i in range(len(outputs['boxes'])):
                conf = float(outputs['scores'][i])
                if conf >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = outputs['boxes'][i].tolist()
                    x1, x2, y1, y2 = int(x1 * scale_x), int(x2 * scale_x), int(y1 * scale_y), int(y2 * scale_y)
                    cls_id = int(outputs['labels'][i])
                    raw_name = self.active_classes.get(cls_id, f"Class {cls_id}")
                    std_name = DefectAnalyzer.normalize_class_name(raw_name)
                    predictions.append({"box": [x1, y1, x2, y2], "class": std_name, "conf": conf})

        elif self.model_type == "pytorch_effdet":
            TARGET_SIZE = (640, 640) 
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, TARGET_SIZE)
            img_tensor = self.pytorch_transform(frame_resized).unsqueeze(0).to(self.device).float()
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
            
            detections = outputs[0] 
            scale_x, scale_y = orig_w / TARGET_SIZE[0], orig_h / TARGET_SIZE[1]

            for det in detections:
                conf = float(det[4])
                if conf >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = det[0:4].tolist()
                    x1, x2, y1, y2 = int(x1 * scale_x), int(x2 * scale_x), int(y1 * scale_y), int(y2 * scale_y)
                    cls_id = int(det[5])
                    raw_name = self.active_classes.get(cls_id, f"Class {cls_id}")
                    std_name = DefectAnalyzer.normalize_class_name(raw_name)
                    predictions.append({"box": [x1, y1, x2, y2], "class": std_name, "conf": conf})

        return predictions