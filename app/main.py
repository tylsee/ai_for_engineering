import sys
import os
# Ensure app can find src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import customtkinter as ctk
from PIL import Image
from tkinter import filedialog
import threading

from src.inference import DefectModelWrapper
from src.analyser import DefectAnalyzer, GUI_LEGEND_COLORS

AVAILABLE_MODELS = [
    "efficientdet_best.pth", "faster_rcnn_resnet18_best.pth", 
    "rt_detr_best.pt", "yolov11s_best.pt", "yolov8s_best.pt"
]
RECOMMENDED_MODEL = "yolov11s_best.pt" 

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
        
        ctk.CTkLabel(self.stats_frame, text="Weighted Multi-Criteria Severity:", font=ctk.CTkFont(size=13)).pack(anchor="w", padx=20, pady=(15, 2))
        
        self.severity_slider = ctk.CTkSlider(self.stats_frame, from_=0, to=100, number_of_steps=100)
        self.severity_slider.set(0)
        self.severity_slider.configure(state="disabled") 
        self.severity_slider.pack(padx=20, pady=5, fill="x")
        
        self.lbl_severity_status = ctk.CTkLabel(self.stats_frame, text="Status: Clear", font=ctk.CTkFont(size=13, weight="bold"), text_color="#a0a0a0")
        self.lbl_severity_status.pack(anchor="w", padx=20, pady=(0, 10))

        ctk.CTkLabel(self.stats_frame, text="Class Color Legend:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=20, pady=(15, 5))
        for class_name, hex_color in GUI_LEGEND_COLORS.items():
            legend_row = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
            legend_row.pack(anchor="w", padx=25, pady=2)
            ctk.CTkLabel(legend_row, text="■", text_color=hex_color, font=ctk.CTkFont(size=18)).pack(side="left")
            ctk.CTkLabel(legend_row, text=f" {class_name}", font=ctk.CTkFont(size=13)).pack(side="left", padx=5)

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
            
            # BUG FIX: Explicitly clear the text attribute when an image renders
            self.view_label.configure(image=imgtk, text="")
            self.view_label.image = imgtk

    def process_and_display_image(self, frame):
        predictions = self.model_wrapper.predict(frame)
        df, t, s, a = DefectAnalyzer.evaluate_and_draw(frame, predictions)
        self.update_ui_safely(df, t, s, a)

    def update_stats(self, total, severity_score, alerts):
        self.lbl_total.configure(text=f"Total Defects Detected: {total}")
        self.severity_slider.configure(state="normal")
        self.severity_slider.set(severity_score)
        self.severity_slider.configure(state="disabled")
        
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
            
            predictions = self.model_wrapper.predict(frame)
            df, t, s, a = DefectAnalyzer.evaluate_and_draw(frame, predictions)
            
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
                # Delegate entirely to the analyzer to draw everything
                frame_copy, _, _, _ = DefectAnalyzer.evaluate_and_draw(frame_copy, predictions)
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