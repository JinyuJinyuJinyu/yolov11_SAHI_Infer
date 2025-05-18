from sahi.slicing import slice_coco
import cv2
import os

# Create output directory if it doesn't exist
output_dir = "/mnt/DATA/DATASETS/data/dlpj/slp_sahi1280/train1_labels_js"
output_coco_annotation_file_name = "train1.json"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Path to the input image
image_dir = "/mnt/DATA/DATASETS/data/dlpj/slp_sahi1280/train1"
coco_annotation_file_path = "/mnt/DATA/DATASETS/data/dlpj/slp_sahi1280/train1.json"



# Slice the image
coco_dict, save_path = slice_coco(
    coco_annotation_file_path=coco_annotation_file_path,
    image_dir=image_dir,
    output_dir = output_dir, 
    output_coco_annotation_file_name=output_coco_annotation_file_name,
    slice_height= 640,
    slice_width = 640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)




