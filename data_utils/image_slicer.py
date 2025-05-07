from sahi.slicing import slice_coco

# Paths
coco_annotation_file_path = '/mnt/DATA/DATASETS/data/dlpj/slp_sahi/train/output/labels.json'
image_dir = '/mnt/DATA/DATASETS/data/dlpj/slp_sahi/train/images'
output_dir = "/mnt/DATA/DATASETS/data/dlpj/slp_sahi/train/data_sahi"
output_coco_annotation_file_name = "labels_sahi.json"
# Slice images
coco_dict, coco_path = slice_coco(
    coco_annotation_file_path=coco_annotation_file_path,
    output_coco_annotation_file_name = output_coco_annotation_file_name,
    image_dir=image_dir,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,  # 20% overlap to avoid missing objects
    overlap_width_ratio=0.2,
    min_area_ratio=0.1,  # Ignore small annotations split across slices
    output_dir=output_dir
)