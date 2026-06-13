import os
import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image
from tkinter import filedialog
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision.transforms as T
import threading
from ultralytics import YOLO, RTDETR

# ==========================================
# 1. CONSTANTS & STANDARDIZATION DEFINITION
# ==========================================
AVAILABLE_MODELS = [
    "efficientdet_best.pth",       # Full post-processing fixed payload file
    "faster_rcnn_resnet18_best.pth",    # Faster R-CNN save path
    "rt_detr_best.pt",
    "yolov11s_best.pt",
    "yolov8s_best.pt"
]

RECOMMENDED_MODEL = "yolov8s_best.pt" 
MODELS_DIR = "models"
CONFIDENCE_THRESHOLD = 0.40  

# Global normalization dictionary to reconcile all framework formats
STANDARD_CLASS_MAP = {
    "cracks": "Cracks",
    "crack": "Cracks",
    "spalling": "Spalling",
    "corrosion": "Corrosion",
    "potholes": "Potholes",
    "pothole": "Potholes",
    "paint degradation": "Paint Degradation",
    "paint_degradation": "Paint Degradation"
}

# ==============================================================================
# 🧠 MULTI-CRITERIA FUSION SEVERITY WEIGHTS
# Higher weights represent severe structural asset degradation risks
# ==============================================================================
CLASS_CRITICALITY_WEIGHTS = {
    "Potholes": 5.0,              # Structural integrity failure/immediate patch requirement
    "Cracks": 3.5,                # Active material stress degradation
    "Spalling": 3.5,              # Concrete cracking/reinforcement exposure risk
    "Corrosion": 2.0,             # Surface oxidized metal structural decay
    "Paint Degradation": 1.0,     # Cosmetic weathering layer failure
    "Unknown": 1.0                # Base metric fallback
}

# Strict color map (OpenCV uses BGR format)
CLASS_COLORS = {
    "Cracks": (0, 0, 255),              # Red
    "Spalling": (0, 165, 255),          # Orange
    "Corrosion": (0, 255, 255),         # Yellow
    "Potholes": (255, 0, 0),            # Blue
    "Paint Degradation": (0, 255, 0),    # Green
    "Unknown": (128, 128, 128)          # Grey fallback
}

# Hex equivalent colors for the Tkinter GUI Legend
GUI_LEGEND_COLORS = {
    "Cracks": "#FF0000",
    "Spalling": "#FFA500",
    "Corrosion": "#FFFF00",
    "Potholes": "#0000FF",
    "Paint Degradation": "#00FF00",
    "Unknown Class": "#808080"
}

# ==========================================
# 2. REAL AI INFERENCE WRAPPER 
# ==========================================
class DefectModelWrapper:
    def __init__(self):
        self.model = None
        self.model_type = None
        self.active_classes = {} 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pytorch_transform = T.Compose([T.ToTensor()])

    def normalize_class_name(self, raw_name):
        clean_name = str(raw_name).strip().lower()
        return STANDARD_CLASS_MAP.get(clean_name, raw_name)

    def load_model(self, model_filename):
        path = os.path.join(MODELS_DIR, model_filename)
        print(f"Loading {model_filename}...")
        
        try:
            # --- ULTRALYTICS MODELS LOADING ---
            if "yolo" in model_filename.lower() or "rt_detr" in model_filename.lower():
                self.model_type = "ultralytics"
                self.model = YOLO(path) if "yolo" in model_filename.lower() else RTDETR(path)
                self.active_classes = self.model.names 
            
            # --- FASTER R-CNN MODELS LOADING ---
            elif "faster_rcnn" in model_filename.lower():
                self.model_type = "pytorch_frcnn"
                self.active_classes = {
                    1: "corrosion", 
                    2: "cracks", 
                    3: "paint_degradation", 
                    4: "potholes", 
                    5: "spalling"
                }
                
                backbone = resnet_fpn_backbone(backbone_name='resnet18', weights=None, trainable_layers=3)
                self.model = FasterRCNN(backbone, num_classes=6)
                
                # 1. Load the checkpoint file into memory first
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                
                # 2. Extract the state_dict if it is nested, or use the whole checkpoint
                state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
                
                # 3. Apply the state_dict to the model using strict=False here
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(self.device).eval()

            # --- EFFICIENTDET MODELS LOADING ---
            elif "efficientdet" in model_filename.lower():
                self.model_type = "pytorch_effdet"
                self.active_classes = {
                    1: "corrosion", 
                    2: "cracks", 
                    3: "paint_degradation", 
                    4: "potholes", 
                    5: "spalling"
                }
                
                from effdet import create_model, DetBenchPredict
                net = create_model('tf_efficientdet_d0', pretrained=False, num_classes=5, image_size=(640, 640))
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
                elif isinstance(checkpoint, dict):
                    net.load_state_dict(checkpoint, strict=False)
                else:
                    net = checkpoint
                
                if not isinstance(net, DetBenchPredict):
                    self.model = DetBenchPredict(net)
                else:
                    self.model = net
                
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
                    std_name = self.normalize_class_name(raw_name)
                    predictions.append({"box": [x1, y1, x2, y2], "class": std_name, "conf": conf})

        elif self.model_type == "pytorch_frcnn":
            TARGET_SIZE = (640, 640) 
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, TARGET_SIZE)
            img_tensor = [self.pytorch_transform(frame_resized).to(self.device).float()]
            
            with torch.no_grad():
                outputs = self.model(img_tensor)[0]
            
            scale_x = orig_w / TARGET_SIZE[0]
            scale_y = orig_h / TARGET_SIZE[1]

            for i in range(len(outputs['boxes'])):
                conf = float(outputs['scores'][i])
                if conf >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = outputs['boxes'][i].tolist()
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    cls_id = int(outputs['labels'][i])
                    raw_name = self.active_classes.get(cls_id, f"Class {cls_id}")
                    std_name = self.normalize_class_name(raw_name)
                    predictions.append({"box": [x1, y1, x2, y2], "class": std_name, "conf": conf})

        elif self.model_type == "pytorch_effdet":
            TARGET_SIZE = (640, 640) 
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, TARGET_SIZE)
            img_tensor = self.pytorch_transform(frame_resized).unsqueeze(0).to(self.device).float()
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
            
            detections = outputs[0] 
            scale_x = orig_w / TARGET_SIZE[0]
            scale_y = orig_h / TARGET_SIZE[1]

            for det in detections:
                conf = float(det[4])
                if conf >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = det[0:4].tolist()
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    cls_id = int(det[5])
                    raw_name = self.active_classes.get(cls_id, f"Class {cls_id}")
                    std_name = self.normalize_class_name(raw_name)
                    predictions.append({"box": [x1, y1, x2, y2], "class": std_name, "conf": conf})

        return predictions

# ==========================================
# 3. MAIN DASHBOARD APPLICATION (GUI)
# ==========================================
class FacilitiesDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Facilities Management Dashboard")
        self.geometry("1350x880")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.video_capture = None
        self.is_webcam_running = False
        self.current_frame = None  
        self.model_wrapper = DefectModelWrapper()
        
        self.display_models = []
        for model in AVAILABLE_MODELS:
            if model == RECOMMENDED_MODEL:
                self.display_models.insert(0, f"{model} (Recommended)")
            else:
                self.display_models.append(model)

        self.build_ui()

    def build_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Sidebar Left
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="Defect Detection", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(20, 10))
        
        ctk.CTkLabel(self.sidebar, text="Select AI Model:").pack(anchor="w", padx=20, pady=(10, 0))
        self.model_dropdown = ctk.CTkOptionMenu(self.sidebar, values=self.display_models, command=self.on_model_change)
        self.model_dropdown.pack(padx=20, pady=10, fill="x")
        
        ctk.CTkLabel(self.sidebar, text="Input Source:").pack(anchor="w", padx=20, pady=(20, 0))
        self.source_var = ctk.StringVar(value="Upload")
        
        ctk.CTkRadioButton(self.sidebar, text="Upload Image", variable=self.source_var, value="Upload", command=self.toggle_source).pack(anchor="w", padx=20, pady=10)
        ctk.CTkRadioButton(self.sidebar, text="Live Webcam", variable=self.source_var, value="Webcam", command=self.toggle_source).pack(anchor="w", padx=20, pady=10)

        self.btn_upload = ctk.CTkButton(self.sidebar, text="Choose Image", command=self.upload_image)
        self.btn_upload.pack(padx=20, pady=20, fill="x")

        self.btn_webcam = ctk.CTkButton(self.sidebar, text="Start Webcam", command=self.toggle_webcam, state="disabled")
        self.btn_webcam.pack(padx=20, pady=0, fill="x")

        self.btn_compare = ctk.CTkButton(self.sidebar, text="Compare All Models", fg_color="#ff8c00", hover_color="#cc7000", command=self.compare_models)
        self.btn_compare.pack(padx=20, pady=40, fill="x")

        # Main Center Window Frame
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.view_label = ctk.CTkLabel(self.main_frame, text="No Image Loaded", font=ctk.CTkFont(size=24))
        self.view_label.grid(row=1, column=0, sticky="nsew")

        # Stats Frame Right
        self.stats_frame = ctk.CTkFrame(self, width=320, corner_radius=10)
        self.stats_frame.grid(row=0, column=2, padx=(0, 20), pady=20, sticky="nsew")

        ctk.CTkLabel(self.stats_frame, text="Inspection Overview", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=20)
        
        self.lbl_total = ctk.CTkLabel(self.stats_frame, text="Total Defects Detected: 0", font=ctk.CTkFont(size=14))
        self.lbl_total.pack(anchor="w", padx=20, pady=5)
        
        # Real-time Multi-Criteria Slider Gauge Indicator
        ctk.CTkLabel(self.stats_frame, text="Weighted Multi-Criteria Severity:", font=ctk.CTkFont(size=13)).pack(anchor="w", padx=20, pady=(15, 2))
        
        self.severity_slider = ctk.CTkSlider(self.stats_frame, from_=0, to=100, number_of_steps=100)
        self.severity_slider.set(0)
        self.severity_slider.configure(state="disabled") 
        self.severity_slider.pack(padx=20, pady=5, fill="x")
        
        self.lbl_severity_status = ctk.CTkLabel(self.stats_frame, text="Status: Clear", font=ctk.CTkFont(size=13, weight="bold"), text_color="#a0a0a0")
        self.lbl_severity_status.pack(anchor="w", padx=20, pady=(0, 10))

        # Class Color Legend Subgrid
        ctk.CTkLabel(self.stats_frame, text="Class Color Legend:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=20, pady=(15, 5))
        for class_name, hex_color in GUI_LEGEND_COLORS.items():
            legend_row = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
            legend_row.pack(anchor="w", padx=25, pady=2)
            
            color_box = ctk.CTkLabel(legend_row, text="■", text_color=hex_color, font=ctk.CTkFont(size=18))
            color_box.pack(side="left")
            
            lbl_name = ctk.CTkLabel(legend_row, text=f" {class_name}", font=ctk.CTkFont(size=13))
            lbl_name.pack(side="left", padx=5)

        self.alerts_box = ctk.CTkTextbox(self.stats_frame, height=250, state="disabled")
        self.alerts_box.pack(padx=20, pady=20, fill="both", expand=True)

        self.on_model_change(self.model_dropdown.get())

    def on_model_change(self, selected_value):
        actual_filename = selected_value.replace(" (Recommended)", "").strip()
        self.model_wrapper.load_model(actual_filename)
        
        if self.current_frame is not None and not self.is_webcam_running:
            self.process_and_display_image(self.current_frame.copy())

    def toggle_source(self):
        if self.source_var.get() == "Upload":
            self.btn_upload.configure(state="normal")
            self.btn_webcam.configure(state="disabled")
            if self.is_webcam_running:
                self.toggle_webcam()
        else:
            self.btn_upload.configure(state="disabled")
            self.btn_webcam.configure(state="normal")

    # ==============================================================================
    # 🛠️ CORE LOGIC UPDATE: RE-ENGINEERED INFERENCE ENGINE
    # Implements: Total Frame Cumulative Score = Sum(Weight * Area Ratio * Confidence)
    # ==============================================================================
    def run_inference_and_draw(self, frame):
        h, w = frame.shape[:2]
        predictions = self.model_wrapper.predict(frame)
        
        current_frame_defects = len(predictions)
        cumulative_weighted_score = 0.0
        alerts = []
        display_frame = frame.copy()

        for p in predictions:
            box, cls_name, conf = p["box"], p["class"], p["conf"]
            
            if cls_name not in CLASS_COLORS:
                display_name = f"Unknown ({cls_name})"
                color = CLASS_COLORS["Unknown"]
                criticality_weight = CLASS_CRITICALITY_WEIGHTS["Unknown"]
            else:
                display_name = cls_name
                color = CLASS_COLORS[cls_name]
                criticality_weight = CLASS_CRITICALITY_WEIGHTS.get(cls_name, 1.0)
            
            # 1. Size Metric: Physical coverage area relative to overall image frame
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            size_metric = box_area / (w * h)
            
            # 2. Multi-Criteria Fusion Math Engine per instance
            instance_severity = criticality_weight * size_metric * conf
            cumulative_weighted_score += instance_severity
            
            # Categorize text alerts contextually
            if instance_severity > 0.35: alert_tag = "Critical Threat"
            elif instance_severity > 0.10: alert_tag = "Moderate Risk"
            else: alert_tag = "Minor Defect"
            
            alerts.append(f"[{alert_tag}] {display_name} ({conf:.2f})")
            
            # Standard drawing pipelines
            cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), color, 3)
            label = f"{display_name} | {conf:.2f}"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(display_frame, (box[0], box[1] - th - 12), (box[0] + tw + 4, box[1]), color, -1)
            cv2.putText(display_frame, label, (box[0] + 2, box[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # 3. Calibration: Normalize cumulative score into a clean 0-100% GUI metric.
        # A combined frame score of 0.25 (e.g., a massive pothole or several combined medium cracks)
        # represents a structurally compromised environment that saturates the gauge to 100%.
        normalized_severity_index = min(100.0, (cumulative_weighted_score / 0.25) * 100.0) if current_frame_defects > 0 else 0.0

        return display_frame, current_frame_defects, normalized_severity_index, alerts

    def update_ui_safely(self, display_frame, total, severity_score, alerts):
        if self.is_webcam_running or self.source_var.get() == "Upload":
            self.update_stats(total, severity_score, alerts)

            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            label_w = self.view_label.winfo_width()
            label_h = self.view_label.winfo_height()
            if label_w < 100 or label_h < 100:  
                label_w, label_h = 800, 600
                
            img.thumbnail((label_w, label_h), Image.Resampling.LANCZOS)
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
            
            self.view_label.configure(image=imgtk, text="")
            self.view_label.image = imgtk

    def process_and_display_image(self, frame):
        df, t, s, a = self.run_inference_and_draw(frame)
        self.update_ui_safely(df, t, s, a)

    def update_stats(self, total, severity_score, alerts):
        self.lbl_total.configure(text=f"Total Defects Detected: {total}")
        
        self.severity_slider.configure(state="normal")
        self.severity_slider.set(severity_score)
        self.severity_slider.configure(state="disabled")
        
        # Dynamic threshold alerting brackets
        if severity_score == 0:
            self.lbl_severity_status.configure(text="Status: Clear / Nominal Health", text_color="#a0a0a0")
        elif severity_score <= 20:
            self.lbl_severity_status.configure(text=f"Status: Safe / Superficial Wear ({severity_score:.1f}%)", text_color="#00FF00")
        elif severity_score <= 50:
            self.lbl_severity_status.configure(text=f"Status: Action Required / Moderate ({severity_score:.1f}%)", text_color="#FFFF00")
        elif severity_score <= 80:
            self.lbl_severity_status.configure(text=f"Status: High Degradation Danger ({severity_score:.1f}%)", text_color="#FFA500")
        else:
            self.lbl_severity_status.configure(text=f"Status: STRUCTURAL COMPROMISE FAILURE ({severity_score:.1f}%)", text_color="#FF4A4A")
        
        self.alerts_box.configure(state="normal")
        self.alerts_box.delete("1.0", "end")
        self.alerts_box.insert("end", f"MODEL: {self.model_dropdown.get()}\n\nRECENT ALERTS:\n" + "-"*20 + "\n")
        for alert in alerts: self.alerts_box.insert("end", alert + "\n")
        self.alerts_box.configure(state="disabled")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            frame = cv2.imread(file_path)
            if frame is not None:
                self.current_frame = frame  
                self.process_and_display_image(self.current_frame.copy())

    def toggle_webcam(self):
        if not self.is_webcam_running:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened(): return
            
            self.is_webcam_running = True
            self.btn_webcam.configure(text="Stop Webcam", fg_color="red")
            threading.Thread(target=self.webcam_inference_thread, daemon=True).start()
        else:
            self.is_webcam_running = False
            self.btn_webcam.configure(text="Start Webcam", fg_color=["#3a7ebf", "#1f538d"])
            if self.video_capture: self.video_capture.release()
            self.view_label.configure(image=None, text="Webcam Stopped")

    def webcam_inference_thread(self):
        while self.is_webcam_running and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret: continue
            self.current_frame = frame  
            df, t, s, a = self.run_inference_and_draw(frame)
            if self.is_webcam_running: 
                self.after(0, self.update_ui_safely, df, t, s, a)

    def compare_models(self):
        if self.current_frame is None: return

        comp_window = ctk.CTkToplevel(self)
        comp_window.title("Model Comparison View")
        comp_window.geometry("1100x700")

        original_model = self.model_dropdown.get().replace(" (Recommended)", "").strip()

        row, col = 0, 0
        for model_name in AVAILABLE_MODELS:
            success = self.model_wrapper.load_model(model_name)
            frame_copy = self.current_frame.copy()
            
            if success:
                predictions = self.model_wrapper.predict(frame_copy)
                for p in predictions:
                    box, cls_name = p["box"], p["class"]
                    
                    if cls_name not in CLASS_COLORS:
                        display_name = f"Unknown ({cls_name})"
                        color = CLASS_COLORS["Unknown"]
                    else:
                        display_name = cls_name
                        color = CLASS_COLORS[cls_name]
                    
                    cv2.rectangle(frame_copy, (box[0], box[1]), (box[2], box[3]), color, 3)
                    cv2.putText(frame_copy, display_name, (box[0], box[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            else:
                cv2.putText(frame_copy, "FAILED TO LOAD", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((350, 350), Image.Resampling.LANCZOS)
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
            
            lbl = ctk.CTkLabel(comp_window, image=imgtk, text=model_name, font=("Arial", 14, "bold"), compound="top")
            lbl.grid(row=row, column=col, padx=10, pady=10)
            
            col += 1
            if col > 2: 
                col = 0
                row += 1

        self.model_wrapper.load_model(original_model)

if __name__ == "__main__":
    app = FacilitiesDashboard()
    app.mainloop()