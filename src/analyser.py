import cv2

# ==============================================================================
# 1. CONSTANTS & STANDARDIZATION DEFINITION
# ==============================================================================
STANDARD_CLASS_MAP = {
    "cracks": "Cracks", "crack": "Cracks",
    "spalling": "Spalling",
    "corrosion": "Corrosion",
    "potholes": "Potholes", "pothole": "Potholes",
    "paint degradation": "Paint Degradation", "paint_degradation": "Paint Degradation"
}

CLASS_CRITICALITY_WEIGHTS = {
    "Potholes": 5.0, "Cracks": 3.5, "Spalling": 3.5,
    "Corrosion": 2.0, "Paint Degradation": 1.0, "Unknown": 1.0
}

CLASS_COLORS = {
    "Cracks": (0, 0, 255), "Spalling": (0, 165, 255),
    "Corrosion": (0, 255, 255), "Potholes": (255, 0, 0),
    "Paint Degradation": (0, 255, 0), "Unknown": (128, 128, 128)
}

GUI_LEGEND_COLORS = {
    "Cracks": "#FF0000", "Spalling": "#FFA500", "Corrosion": "#FFFF00",
    "Potholes": "#0000FF", "Paint Degradation": "#00FF00", "Unknown Class": "#808080"
}

class DefectAnalyzer:
    @staticmethod
    def normalize_class_name(raw_name):
        clean_name = str(raw_name).strip().lower()
        return STANDARD_CLASS_MAP.get(clean_name, raw_name)

    @staticmethod
    def evaluate_and_draw(frame, predictions):
        """Processes predictions, draws boxes, and calculates severity."""
        h, w = frame.shape[:2]
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
            
            # Size Metric
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            size_metric = box_area / (w * h)
            
            # Multi-Criteria Fusion Math
            instance_severity = criticality_weight * size_metric * conf
            cumulative_weighted_score += instance_severity
            
            # Text Alerts
            if instance_severity > 0.35: alert_tag = "Critical Threat"
            elif instance_severity > 0.10: alert_tag = "Moderate Risk"
            else: alert_tag = "Minor Defect"
            alerts.append(f"[{alert_tag}] {display_name} ({conf:.2f})")
            
            # Draw Standard Pipelines
            cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), color, 3)
            label = f"{display_name} | {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(display_frame, (box[0], box[1] - th - 12), (box[0] + tw + 4, box[1]), color, -1)
            cv2.putText(display_frame, label, (box[0] + 2, box[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Normalization
        normalized_severity_index = min(100.0, (cumulative_weighted_score / 0.25) * 100.0) if current_frame_defects > 0 else 0.0

        return display_frame, current_frame_defects, normalized_severity_index, alerts