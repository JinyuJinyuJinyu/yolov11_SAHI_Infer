import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import onnxruntime as ort
from threading import Thread
import time
from sahi.slicing import slice_image
import torch
from util import non_max_suppression
import torchvision
import csv

# Define class names 
CLASS_NAMES = ['hot_spot', 'low_temp', 'short_circuit']
NUM_CLASSES = len(CLASS_NAMES)

# Generate random colors for each class
np.random.seed(42)
BOX_COLORS = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype="uint8")

class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Object Detection")
        self.root.geometry("1280x720")

        # Variables
        self.video_path = None
        self.running = False
        self.model = None
        self.input_size = 640  # YOLO model input size (640x640)
        self.conf_thres = tk.DoubleVar(value=0.35)  # Default confidence threshold
        self.iou_thres = tk.DoubleVar(value=0.5)    # Default IoU threshold
        self.detection_counts = {name: 0 for name in CLASS_NAMES}  # Track detection counts
        self.total_detections = 0
        self.detection_results = []  # Store detection results for saving
        self.frame_count = 0  # Track frame number
        self.inference_times = []  # Track inference times for FPS calculation

        # GUI Components
        self.create_widgets()

    def create_widgets(self):
        # Main frame to organize layout
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Left frame for controls and video
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, padx=5)

        # Video selection button
        self.select_btn = tk.Button(self.left_frame, text="Select Video", command=self.select_video)
        self.select_btn.pack(pady=5)

        # Model selection button
        self.model_btn = tk.Button(self.left_frame, text="Select ONNX Model", command=self.select_model)
        self.model_btn.pack(pady=5)

        # Confidence threshold slider
        self.conf_label = tk.Label(self.left_frame, text="Confidence Threshold: 0.35")
        self.conf_label.pack(pady=5)
        self.conf_slider = tk.Scale(self.left_frame, from_=0.1, to=1.0, resolution=0.01, 
                                  orient=tk.HORIZONTAL, variable=self.conf_thres,
                                  command=lambda val: self.conf_label.config(text=f"Confidence Threshold: {float(val):.2f}"))
        self.conf_slider.pack(pady=5)

        # IoU threshold slider
        self.iou_label = tk.Label(self.left_frame, text="IoU Threshold: 0.50")
        self.iou_label.pack(pady=5)
        self.iou_slider = tk.Scale(self.left_frame, from_=0.1, to=1.0, resolution=0.01, 
                                 orient=tk.HORIZONTAL, variable=self.iou_thres,
                                 command=lambda val: self.iou_label.config(text=f"IoU Threshold: {float(val):.2f}"))
        self.iou_slider.pack(pady=5)

        # Start/Stop button
        self.start_btn = tk.Button(self.left_frame, text="Start Detection", command=self.toggle_detection)
        self.start_btn.pack(pady=5)

        # Save results button
        self.save_btn = tk.Button(self.left_frame, text="Save Results", command=self.save_results)
        self.save_btn.pack(pady=5)

        # Video display
        self.canvas = tk.Canvas(self.left_frame, width=960, height=540)
        self.canvas.pack(pady=5)

        # Status frame
        self.status_frame = tk.Frame(self.left_frame)
        self.status_frame.pack(pady=5)
        self.status = tk.Label(self.status_frame, text="Status: Idle")
        self.status.pack(anchor=tk.W, padx=5, pady=2)

        # Right frame for sub-windows
        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, padx=5, fill=tk.Y)

        # Detection count sub-window
        self.count_frame = tk.LabelFrame(self.right_frame, text="Detection Counts & Costs", width=200, height=200)
        self.count_frame.pack(pady=5, fill=tk.X)
        self.count_frame.pack_propagate(False)

        self.count_labels = {}
        for cls in CLASS_NAMES:
            label = tk.Label(self.count_frame, text=f"{cls}: 0")
            label.pack(anchor=tk.W, padx=5, pady=2)
            self.count_labels[cls] = label

        self.total_label = tk.Label(self.count_frame, text="Total Detections: 0")
        self.total_label.pack(anchor=tk.W, padx=5, pady=2)
        self.cost_label = tk.Label(self.count_frame, text="Estimated Repair Cost: $0")
        self.cost_label.pack(anchor=tk.W, padx=5, pady=2)

        # Geolocation sub-window
        self.geo_frame = tk.LabelFrame(self.right_frame, text="Geolocation Info", width=200, height=150)
        self.geo_frame.pack(pady=5, fill=tk.X)
        self.geo_frame.pack_propagate(False)

        tk.Label(self.geo_frame, text="Latitude: 37.7749° N").pack(anchor=tk.W, padx=5, pady=2)
        tk.Label(self.geo_frame, text="Longitude: 122.4194° W").pack(anchor=tk.W, padx=5, pady=2)
        tk.Label(self.geo_frame, text="Location: xx, xx").pack(anchor=tk.W, padx=5, pady=2)
        tk.Label(self.geo_frame, text="Panel ID: SP-xxx").pack(anchor=tk.W, padx=5, pady=2)

    def select_video(self):
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if self.video_path:
            self.status.config(text=f"Status: Video selected - {self.video_path.split('/')[-1]}")
            # Reset detection counts and results when a new video is selected
            self.reset_counts()

    def select_model(self):
        model_path = filedialog.askopenfilename(
            filetypes=[("ONNX files", "*.onnx")]
        )
        if model_path:
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
                self.model = ort.InferenceSession(model_path, providers=providers)
                self.status.config(text="Status: Model loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def reset_counts(self):
        """Reset detection counts, results, and inference times."""
        self.detection_counts = {name: 0 for name in CLASS_NAMES}
        self.total_detections = 0
        self.detection_results = []
        self.frame_count = 0
        self.inference_times = []
        for cls in CLASS_NAMES:
            self.count_labels[cls].config(text=f"{cls}: 0")
        self.total_label.config(text="Total Detections: 0")
        self.cost_label.config(text="Estimated Repair Cost: $0")

    def update_counts(self, detections, frame_number):
        """Update detection counts and store results."""
        for det in detections:
            cls = int(det[5])
            cls_name = CLASS_NAMES[cls]
            self.detection_counts[cls_name] += 1
            self.total_detections += 1
            # Store detection results
            self.detection_results.append({
                'frame_number': frame_number,
                'class': cls_name,
                'confidence': float(det[4]),
                'x1': float(det[0]),
                'y1': float(det[1]),
                'x2': float(det[2]),
                'y2': float(det[3])
            })
        
        # Update GUI labels
        for cls in CLASS_NAMES:
            self.count_labels[cls].config(text=f"{cls}: {self.detection_counts[cls]}")
        self.total_label.config(text=f"Total Detections: {self.total_detections}")
        self.cost_label.config(text=f"Estimated Repair Cost: ${self.total_detections * 100}")

    def save_results(self):
        """Save detection results to a CSV file."""
        if not self.detection_results:
            messagebox.showwarning("Warning", "No detection results to save!")
            return
        
        output_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if output_path:
            try:
                with open(output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['frame_number', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2'])
                    writer.writeheader()
                    for result in self.detection_results:
                        writer.writerow(result)
                messagebox.showinfo("Success", f"Results saved to {output_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def preprocess_image(self, image, resize_to_640=False):
        """Preprocess image to match Yolo_Dataset pipeline."""
        h, w = image.shape[:2]
        
        if resize_to_640 or (h <= self.input_size and w <= self.input_size):
            r = min(self.input_size / h, self.input_size / w)
            pad = (int(w * r), int(h * r))
            if (h, w) != pad[::-1]:
                image = cv2.resize(image, dsize=pad, interpolation=cv2.INTER_LINEAR)
            
            top, bottom = int((self.input_size - pad[1]) / 2), int((self.input_size - pad[1]) / 2 + (self.input_size - pad[1]) % 2)
            left, right = int((self.input_size - pad[0]) / 2), int((self.input_size - pad[0]) / 2 + (self.input_size - pad[0]) % 2)
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            ratio = (r, r)
            pad_info = (left, top)
        else:
            ratio = (1.0, 1.0)
            pad_info = (0, 0)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1)).astype(np.float32)
        image /= 255.0
        return image, ratio, pad_info, (h, w)

    def predict_onnx_single_img(self, img):
        """Run inference on a single 640x640 image."""
        img_np, ratio, pad, orig_shape = self.preprocess_image(img, resize_to_640=True)
        img_np = img_np[np.newaxis, ...]
        
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        outputs = self.model.run([output_name], {input_name: img_np})[0]
        
        outputs = torch.from_numpy(outputs).to('cuda' if torch.cuda.is_available() else 'cpu')
        detections = non_max_suppression(outputs, confidence_threshold=self.conf_thres.get(), iou_threshold=self.iou_thres.get())[0]
        
        if detections.shape[0] > 0:
            detections[:, :4] = detections[:, :4].clone()
            detections[:, [0, 2]] = (detections[:, [0, 2]] - pad[0]) / ratio[0]
            detections[:, [1, 3]] = (detections[:, [1, 3]] - pad[1]) / ratio[1]
        
        return detections

    def predict_onnx(self, image):
        """Run inference on an image, using SAHI for large images or resizing for small ones."""
        h, w = image.shape[:2]
        
        start_time = time.time()
        if h > self.input_size or w > self.input_size:
            slice_result = slice_image(
                image=image,
                slice_width=self.input_size,
                slice_height=self.input_size,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )
            all_detections = []
            
            for slice_info in slice_result:
                slice_img = slice_info['image']
                shift_x, shift_y = slice_info['starting_pixel']
                
                detections = self.predict_onnx_single_img(slice_img)
                
                if detections.shape[0] > 0:
                    detections[:, [0, 2]] += shift_x
                    detections[:, [1, 3]] += shift_y
                    all_detections.append(detections)
            
            if all_detections:
                all_detections = torch.cat(all_detections, dim=0)
                boxes = all_detections[:, :4]
                scores = all_detections[:, 4]
                labels = all_detections[:, 5]
                indices = torchvision.ops.nms(boxes, scores, self.iou_thres.get())
                all_detections = all_detections[indices]
            else:
                all_detections = torch.zeros((0, 6), device='cuda' if torch.cuda.is_available() else 'cpu')
        else:
            all_detections = self.predict_onnx_single_img(image)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        return all_detections

    def draw_detections(self, image, detections, ais, cis=-1):
        """Draw bounding boxes, confidence scores, class labels, and FPS on the image."""
        image = image.copy()
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls = int(cls)
            color = BOX_COLORS[cls].tolist()
            label = f"{CLASS_NAMES[cls]} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw FPS in the upper-left corner
        fps_text = f"avg_infer_speed: {ais:.2f}, current_infer_speed: {cis:.2f}"
        cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

    def process_video(self):
        if not self.video_path or not self.model:
            self.root.after(0, lambda: messagebox.showwarning("Warning", "Please select video and model first!"))
            return
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror("Error", "Failed to open video!"))
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = 1 / fps
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            detections = self.predict_onnx(frame)
            
            # Calculate FPS
            avg_is = sum(self.inference_times)/len(self.inference_times) if self.inference_times else 0.0
            
            if detections.shape[0] > 0:
                frame = self.draw_detections(frame, detections, avg_is, self.inference_times[-1])
                self.root.after(0, lambda det=detections, fn=self.frame_count: self.update_counts(det, fn))
            else:
                # Draw FPS even if no detections
                frame = self.draw_detections(frame, detections, avg_is,self.inference_times[-1])
            
            h, w = frame.shape[:2]
            r = min(960 / w, 540 / h)
            new_size = (int(w * r), int(h * r))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            
            self.root.after(0, lambda i=img: self.canvas.create_image(0, 0, anchor=tk.NW, image=i))
            self.canvas.image = img
            
            time.sleep(delay)
        
        cap.release()
        self.running = False
        self.root.after(0, lambda: self.start_btn.config(text="Start Detection"))
        self.root.after(0, lambda: self.status.config(text="Status: Idle"))

    def toggle_detection(self):
        if not self.running:
            if self.video_path and self.model:
                self.running = True
                self.start_btn.config(text="Stop Detection")
                self.status.config(text="Status: Running")
                Thread(target=self.process_video).start()
        else:
            self.running = False
            self.start_btn.config(text="Start Detection")
            self.status.config(text="Status: Stopping...")

    def on_closing(self):
        self.running = False
        time.sleep(0.5)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()