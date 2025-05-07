import sys
import os


from sahi.utils.coco import Coco, export_coco_as_yolov5

from glob import glob
import shutil
from tqdm import tqdm
# Initialize COCO object
coco = Coco.from_coco_dict_or_path('/mnt/DATA/DATASETS/data/dlpj/slp_sahi/train/data_sahi/labels_sahi.json_coco.json', 
                                   image_dir='/mnt/DATA/DATASETS/data/dlpj/slp_sahi/train/data_sahi/')

# Export to YOLOv5 format
data_yml_path = export_coco_as_yolov5(
    output_dir='/mnt/DATA/DATASETS/data/dlpj/slp_sahi/train/yolo_sahi',
    train_coco=coco,
    val_coco=None  # Set if you have a trainidation split
)

os.mkdir('/mnt/DATA/DATASETS/data/dlpj/slp_sahi/train/yolo_sahi/images')
os.mkdir('/mnt/DATA/DATASETS/data/dlpj/slp_sahi/train/yolo_sahi/labels')

imgfns = glob('/mnt/DATA/DATASETS/data/dlpj/slp_sahi/train/yolo_sahi/*/*.png')
annofns = glob('/mnt/DATA/DATASETS/data/dlpj/slp_sahi/train/yolo_sahi/*/*.txt')

for imgfn in tqdm(imgfns):
    shutil.move(imgfn, os.path.join('/mnt/DATA/DATASETS/data/dlpj/slp_sahi/train/yolo_sahi/images', os.path.basename(imgfn)))
for annfn in tqdm(annofns):
    shutil.move(annfn, os.path.join('/mnt/DATA/DATASETS/data/dlpj/slp_sahi/train/yolo_sahi/labels', os.path.basename(annfn)))    