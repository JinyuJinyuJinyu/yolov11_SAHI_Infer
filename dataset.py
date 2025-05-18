import math
import os
import random
from glob import glob
import cv2
import numpy
import torch
from PIL import Image
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np

FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff'

class Yolo_Dataset(Dataset):
    def __init__(self, filenames, input_size, params=None, augment=False):
        self.params = params
        self.augment = augment
        self.input_size = input_size

        # Read labels
        labels = self.load_label(filenames)
        self.labels = list(labels.values())
        self.filenames = list(labels.keys())  # update
        self.n = len(self.filenames)  # number of samples
        self.indices = range(self.n)
        self.albumentations = Albumentations()

    def __getitem__(self, index):
        index = self.indices[index]

        image, shape = self.load_image(index)
        h, w = image.shape[:2]
        # Resize
        image, ratio, pad = resize(image, self.input_size, self.augment)

        label = self.labels[index].copy()
        if label.size:
            label[:, 1:] = wh2xy(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])

        nl = len(label)  # number of labels
        h, w = image.shape[:2]
        cls = label[:, 0:1]
        box = label[:, 1:5]
        box = xy2wh(box, w, h)

        if self.augment:
            # Albumentations
            image, box, cls = self.albumentations(image, box, cls)
            nl = len(box)  # update after albumentations
            # HSV color-space
            augment_hsv(image, self.params)
            # Flip up-down
            if random.random() < self.params['flip_ud']:
                image = numpy.flipud(image)
                if nl:
                    box[:, 1] = 1 - box[:, 1]
            # Flip left-right
            if random.random() < self.params['flip_lr']:
                image = numpy.fliplr(image)
                if nl:
                    box[:, 0] = 1 - box[:, 0]

        target_cls = torch.zeros((nl, 1))
        target_box = torch.zeros((nl, 4))
        if nl:
            target_cls = torch.from_numpy(cls)
            target_box = torch.from_numpy(box)

        # Convert HWC to CHW, BGR to RGB
        sample = image.transpose((2, 0, 1))[::-1]
        sample = numpy.ascontiguousarray(sample)

        return torch.from_numpy(sample)/255., target_cls, target_box, torch.zeros(nl)

    def __len__(self):
        return len(self.filenames)

    def load_image(self, i):
        image = cv2.imread(self.filenames[i])
        h, w = image.shape[:2]
        r = self.input_size / max(h, w)
        if r != 1:
            image = cv2.resize(image,
                               dsize=(int(w * r), int(h * r)),
                               interpolation=resample() if self.augment else cv2.INTER_LINEAR)
        return image, (h, w)

    @staticmethod
    def collate_fn(batch):
        
        samples, cls, box, indices = [], [], [], []
        for i, (sample, cl, bx, idx) in enumerate(batch):
            samples.append(sample)
            # Ensure cls is 2D
            if cl.numel() == 0:
                cl = torch.zeros((0, 1), dtype=torch.float32)
            elif cl.dim() == 1:
                cl = cl.view(-1, 1)
            elif cl.dim() > 2:
                cl = cl.squeeze().view(-1, 1)
            cls.append(cl)
            box.append(bx)
            # Update indices to reflect batch position
            idx = torch.full((cl.shape[0],), i, dtype=torch.int64)
            indices.append(idx)

        samples = torch.stack(samples, dim=0).float()
        cls = torch.cat(cls, dim=0) if cls else torch.zeros((0, 1), dtype=torch.float32)
        box = torch.cat(box, dim=0) if box else torch.zeros((0, 4), dtype=torch.float32)
        indices = torch.cat(indices, dim=0) if indices else torch.zeros((0,), dtype=torch.int64)

        targets = {'cls': cls, 'box': box, 'idx': indices}
        return samples, targets
    @staticmethod
    def _process_label(filename, formats=FORMATS):
        """Helper function to process a single file's label."""
        try:
            # Verify images
            # with open(filename, 'rb') as f:
            #     image = Image.open(f)
            #     image.verify()  # PIL verify
            # shape = image.size  # image size
            # if not ((shape[0] > 9) and (shape[1] > 9)):
            #     print(f"Image size {shape} <10 pixels for {filename}")
            #     return filename, np.zeros((0, 5), dtype=np.float32)
            # if image.format.lower() not in formats:
            #     print(f"Invalid image format {image.format} for {filename}")
            #     return filename, np.zeros((0, 5), dtype=np.float32)

            # Verify labels
            a = f'{os.sep}images{os.sep}'
            b = f'{os.sep}labels{os.sep}'
            label_file = b.join(filename.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt'
            if os.path.isfile(label_file):
                with open(label_file) as f:
                    label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    label = np.array(label, dtype=np.float32)
                nl = len(label)
                if nl:
                    if not (label >= 0).all():
                        print(f"Negative values in label for {filename}")
                        return filename, np.zeros((0, 5), dtype=np.float32)
                    if label.shape[1] != 5:
                        print(f"Invalid label shape {label.shape} for {filename}")
                        return filename, np.zeros((0, 5), dtype=np.float32)
                    if not (label[:, 1:] <= 1).all():
                        print(f"Label values > 1 for {filename}")
                        return filename, np.zeros((0, 5), dtype=np.float32)
                    _, i = np.unique(label, axis=0, return_index=True)
                    if len(i) < nl:  # duplicate row check
                        label = label[i]  # remove duplicates
                else:
                    label = np.zeros((0, 5), dtype=np.float32)
            else:
                label = np.zeros((0, 5), dtype=np.float32)
            return filename, label
        except FileNotFoundError:
            print(f"File not found: {filename}")
            return filename, np.zeros((0, 5), dtype=np.float32)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return filename, np.zeros((0, 5), dtype=np.float32)
        
    @staticmethod
    def load_label(filenames):
        """Load labels using multi-processing."""
        # Determine number of processes (use min of CPU count and number of files)
        num_processes = min(int(2/3*cpu_count()), len(filenames), 16)  # Cap at 16 to avoid overhead
        print(f"Using {num_processes} processes for label loading")

        # Create a partial function with formats
        process_func = partial(Yolo_Dataset._process_label, formats=FORMATS)

        # Use Pool for multi-processing
        x = {}
        with Pool(processes=num_processes) as pool:
            # Map the processing function to filenames with progress bar
            results = list(tqdm(pool.imap(process_func, filenames), total=len(filenames), desc="Loading labels"))
        
        # Collect results into dictionary
        for filename, label in results:
            x[filename] = label

        return x


def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    # Convert nx4 boxes
    # from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def xy2wh(x, w, h):
    # warning: inplace clip
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1E-3)  # x1, x2
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1E-3)  # y1, y2

    # Convert nx4 boxes
    # from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    y = numpy.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def resample():
    choices = (cv2.INTER_AREA,
               cv2.INTER_CUBIC,
               cv2.INTER_LINEAR,
               cv2.INTER_NEAREST,
               cv2.INTER_LANCZOS4)
    return random.choice(seq=choices)


def augment_hsv(image, params):
    # HSV color-space augmentation
    h = params['hsv_h']
    s = params['hsv_s']
    v = params['hsv_v']

    r = numpy.random.uniform(-1, 1, 3) * [h, s, v] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    x = numpy.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype('uint8')
    lut_s = numpy.clip(x * r[1], 0, 255).astype('uint8')
    lut_v = numpy.clip(x * r[2], 0, 255).astype('uint8')

    hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed


def resize(image, input_size, augment):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(input_size / shape[0], input_size / shape[1])
    if not augment:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2

    if shape[::-1] != pad:  # resize
        image = cv2.resize(image,
                           dsize=pad,
                           interpolation=resample() if augment else cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border
    return image, (r, r), (w, h)


class Albumentations:
    def __init__(self):
        self.transform = None
        transforms = [A.Blur(p=0.01),
                        A.CLAHE(p=0.01),
                        A.ToGray(p=0.01),
                        A.MedianBlur(p=0.01)]
        self.transform = A.Compose(transforms, A.BboxParams('yolo', ['class_labels']))

    def __call__(self, image, box, cls):
        if self.transform:
            x = self.transform(image=image,
                               bboxes=box,
                               class_labels=cls)
            image = x['image']
            box = numpy.array(x['bboxes'])
            cls = numpy.array(x['class_labels'])
        return image, box, cls


if __name__ == "__main__":
    # Example usage

    # hsv_h: 0.015
    # hsv_s: 0.7
    # hsv_v: 0.4 wrt ultralytics

    dataset = Yolo_Dataset(filenames=glob('/mnt/DATA/DATASETS/data/dlpj/slp_sahi1280/sahi_test_data/train/images/*.jpg'), input_size=640, params={'flip_ud': 0.5, 'flip_lr': 0.5, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4}, augment=True)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)

    for images, targets in dataloader:
        print(images.shape, targets)
        break
    print("Dataset loaded successfully.")