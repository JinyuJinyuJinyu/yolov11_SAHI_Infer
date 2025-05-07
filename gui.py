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
# Define class names 
CLASS_NAMES = ['hot_spot', 'low_temp', 'short_circuit']  #
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

        # GUI Components
        self.create_widgets()

    def create_widgets(self):
        # Video selection button
        self.select_btn = tk.Button(self.root, text="Select Video", command=self.select_video)
        self.select_btn.pack(pady=10)

        # Model selection button
        self.model_btn = tk.Button(self.root, text="Select ONNX Model", command=self.select_model)
        self.model_btn.pack(pady=10)

        # Start/Stop button
        self.start_btn = tk.Button(self.root, text="Start Detection", command=self.toggle_detection)
        self.start_btn.pack(pady=10)

        # Video display
        self.canvas = tk.Canvas(self.root, width=960, height=540)  # Adjusted for typical video aspect ratio
        self.canvas.pack(pady=10)

        # Status label
        self.status = tk.Label(self.root, text="Status: Idle")
        self.status.pack(pady=10)

    def select_video(self):
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if self.video_path:
            self.status.config(text=f"Status: Video selected - {self.video_path.split('/')[-1]}")

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

    def preprocess_image(self, image, resize_to_640=False):
        """Preprocess image to match Yolo_Dataset pipeline."""
        h, w = image.shape[:2]
        
        if resize_to_640 or (h <= self.input_size and w <= self.input_size):
            # Resize to 640x640 with padding (for small frames or single inference)
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
            # Keep original size for SAHI slicing
            ratio = (1.0, 1.0)
            pad_info = (0, 0)
        
        # Convert BGR to RGB, HWC to CHW, normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1)).astype(np.float32)  # HWC to CHW
        image /= 255.0  # Normalize to [0, 1]
        
        return image, ratio, pad_info, (h, w)

    def predict_onnx_single_img(self, img, conf_thres=0.1, iou_thres=0.5):
        """Run inference on a single 640x640 image."""
        # Preprocess to ensure 640x640
        img_np, ratio, pad, orig_shape = self.preprocess_image(img, resize_to_640=True)
        img_np = img_np[np.newaxis, ...]  # Add batch dimension (1, 3, 640, 640)
        
        # Run ONNX inference
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        outputs = self.model.run([output_name], {input_name: img_np})[0]
        
        # Convert to PyTorch tensor for NMS
        outputs = torch.from_numpy(outputs).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Apply non-maximum suppression
        detections = non_max_suppression(outputs, confidence_threshold=conf_thres, iou_threshold=iou_thres)[0]
        
        # Rescale detections to original slice size
        if detections.shape[0] > 0:
            detections[:, :4] = detections[:, :4].clone()
            detections[:, [0, 2]] = (detections[:, [0, 2]] - pad[0]) / ratio[0]  # x1, x2
            detections[:, [1, 3]] = (detections[:, [1, 3]] - pad[1]) / ratio[1]  # y1, y2
        
        return detections

    def predict_onnx(self, image, conf_thres=0.1, iou_thres=0.5):
        """Run inference on an image, using SAHI for large images or resizing for small ones."""
        h, w = image.shape[:2]
        
        if h > self.input_size or w > self.input_size:
            # Use SAHI for large images
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
                
                # Run inference on slice
                detections = self.predict_onnx_single_img(slice_img, conf_thres, iou_thres)
                
                # Shift detections to original image coordinates
                if detections.shape[0] > 0:
                    detections[:, [0, 2]] += shift_x  # x1, x2
                    detections[:, [1, 3]] += shift_y  # y1, y2
                    all_detections.append(detections)
            
            # Merge detections
            if all_detections:
                all_detections = torch.cat(all_detections, dim=0)
                # Apply NMS to remove duplicates across slices
                boxes = all_detections[:, :4]
                scores = all_detections[:, 4]
                labels = all_detections[:, 5]
                indices = torchvision.ops.nms(boxes, scores, iou_thres)
                all_detections = all_detections[indices]
            else:
                all_detections = torch.zeros((0, 6), device='cuda' if torch.cuda.is_available() else 'cpu')
            
            return all_detections
        else:
            # Resize small images to 640x640
            return self.predict_onnx_single_img(image, conf_thres, iou_thres)

    def draw_detections(self, image, detections):
        """Draw bounding boxes, confidence scores, and class labels on the image."""
        image = image.copy()
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls = int(cls)
            color = BOX_COLORS[cls].tolist()
            label = f"{CLASS_NAMES[cls]} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return image

    def process_video(self):
        if not self.video_path or not self.model:
            messagebox.showwarning("Warning", "Please select video and model first!")
            return
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open video!")
            return
        
        # Get video FPS for smoother playback
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = 1 / fps
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            detections = self.predict_onnx(frame, conf_thres=0.35, iou_thres=0.5)
            
            # Draw detections
            if detections.shape[0] > 0:
                frame = self.draw_detections(frame, detections)
            
            # Resize frame for display (maintain aspect ratio)
            h, w = frame.shape[:2]
            r = min(960 / w, 540 / h)
            new_size = (int(w * r), int(h * r))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
            
            # Convert for Tkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            
            # Update canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img  # Keep reference to avoid garbage collection
            
            self.root.update()
            time.sleep(delay)  # Match video FPS
        
        cap.release()
        self.running = False
        self.start_btn.config(text="Start Detection")
        self.status.config(text="Status: Idle")

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
        time.sleep(0.5)  # Give time for thread to stop
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()