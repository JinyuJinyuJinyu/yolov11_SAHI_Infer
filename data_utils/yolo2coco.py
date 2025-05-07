import fiftyone.utils.yolo as fouy
import fiftyone as fo

# Path to YOLO dataset
yolo_data_dir = "/mnt/DATA/DATASETS/data/dlpj/slp_sahi/train"
yolo_images_dir = f"{yolo_data_dir}/images"
yolo_labels_dir = f"{yolo_data_dir}/labels"
classes = ['hot_spot', 'low_temp', 'short_circuit']  

# Load YOLO dataset into FiftyOne
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.YOLOv4Dataset,
    data_path=yolo_images_dir,
    labels_path=yolo_labels_dir,
    classes=classes
)

# Export to COCO format
coco_json_path = "/mnt/DATA/DATASETS/data/dlpj/slp_sahi/train/coco_dataset.json"
dataset.export(
    export_dir="/mnt/DATA/DATASETS/data/dlpj/slp_sahi/train/output",
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth",
    export_media=False
)