# YOLO Object Detection for Solar Panel Inspection

This project implements a YOLOv11-based object detection system optimized for detecting defects (hot spots, low temperature, and short circuits) in solar panel images or videos. It includes training, evaluation, model quantization, and a Tkinter-based GUI for real-time video inference.

## Features
- **quantization YOLOv11**: Lightweight `yolo_v11_n` model for efficient defect detection.
- **Dataset Support**: Handles YOLO-format datasets with basic data augmentation (HSV, flips).
- **Image Slicing**: Uses SAHI to process large images by slicing them into 640x640 patches.
- **Quantization**: Supports dynamic quantization and quantization-aware training (QAT) for optimized inference.
- **GUI**: Real-time video inference with bounding box visualization, detection counts, and estimated repair costs. Accept any input dimension: for input dimension larger than 640X640, inferece process adopts sahi method. Input dimension smaller than 640X640, input stream will resized to 640X640 for inference.
- **ONNX Export**: Converts trained PyTorch models to ONNX for deployment.
    ```bash
    python pt2onnx.py --model weights/best_model.pt --input_size 640 --nc 3

## Project Structure
- `dataset.py`: Loads and preprocesses YOLO-format datasets with augmentation.
- `networks.py`: Defines YOLOv11 architecture (`yolo_v11_n`, `t`, `s`, `m`, `l`, `x`).
- `loss.py`: Implements loss functions (box, classification, DFL) for training.
- `util.py`: Provides utilities for metrics (mAP), non-maximum suppression (NMS), and smoothing.
- `train.py`: Training loop, evaluation, and quantization-aware training.
- `pt2onnx.py`: Converts PyTorch models (`.pt`) to ONNX format.
- `slice_coco.py`: Slices large images and COCO annotations using SAHI.
- `GUI.py`: Tkinter-based GUI for real-time video inference.
## Training
    ```bash
    python train.py --data data.yaml --input-size 640 --batch_size 32 --epochs 100 --num_workers 8 --pretrained weights/pretrained.pt

--data: Path to YAML file with dataset paths and class info.
--input-size: Image size (default: 640).
--batch_size: Training batch size.
--epochs: Number of epochs.
--num_workers: Data loader workers.
--pretrained: Path to pretrained model (optional).
--lightweight: Enable quantization and pruning.
--skip_train: Skip training for evaluation/quantization.
## GUI for Inference
    ```bash
    python GUI.py

Select a video (.mp4, .avi, .mov) and ONNX model.
Adjust confidence and IoU thresholds via sliders.
Start/stop detection to visualize bounding boxes and track defect counts.
Displays estimated repair cost ($100 per detection).

## Dataset Preparation
YAML File: Create data.yaml
""  train: /path/to/train/images
    val: /path/to/val/images
    test: /path/to/test/images
    nc: 3
    names: ['hot_spot', 'low_temp', 'short_circuit']""

Large Images: Slice high-resolution images with slice_coco.py:
    ```bash
    python slice_coco.py

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/JinyuJinyuJinyu/yolov11_scratch.git
   cd yolov11_scratch

## Dependencies
torch, torchvision
onnxruntime
opencv-python
numpy, pillow
sahi
albumentations
pandas, tqdm, pyyam