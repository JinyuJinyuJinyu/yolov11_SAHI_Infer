from argparse import ArgumentParser
import torch
from networks import yolo_v11_n
from dataset import Yolo_Dataset
from torch.utils.data import DataLoader
from loss import ComputeLoss
from tqdm import tqdm
from glob import glob
from util import compute_ap, compute_metric, wh2xy, non_max_suppression
from copy import deepcopy
import yaml
import os
import pandas as pd


torch.manual_seed(0)
# finish training loop and train.py file, args. 
# then export onnx model, work on inference and GUI (done)
# add quantization and prunning (least priority)

# train file sahi, slice large images

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(model, optimizer, train_loader, valdataloader, scheduler ,device, epochs, loss_fn):
    val_metric = []
    best_model = None
    best_map = -1

    print("Training the model...")
    model.train()
    amp_scale = torch.amp.GradScaler()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            imgs, targets = imgs.to(DEVICE), targets
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                predictions = model(imgs)

                loss_box, loss_cls, loss_dfl = loss_fn(predictions, targets)

                total_loss = loss_box + loss_cls + loss_dfl  # Sum the loss components

            amp_scale.scale(total_loss).backward()
            amp_scale.step(optimizer)
            amp_scale.update()
            train_loss += total_loss.item()
        scheduler.step()
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

        mean_ap, map50, m_rec, m_pre = evaluate(model, valdataloader, loss_fn, device)  # Evaluate after each epoch
        val_metric.append({'train_avg_loss': avg_loss, 'mean_ap': mean_ap, 'map50': map50, 'm_rec': m_rec, 'm_pre': m_pre})
        if mean_ap > best_map:
            best_map = mean_ap
            best_model = deepcopy(model.state_dict()) # Save the best model
            print(f"Best model updated with mean mAP: {best_map:.4f}")
            torch.save(best_model, 'weights/best_model.pt')
            

    torch.save(deepcopy(model.state_dict()), 'weights/last_model.pt')
    pd.DataFrame.from_dict(val_metric).to_csv('weights/val_metric.csv')
    print("Training completed!")

def evaluate(model, data_loader, loss_fn, device):
    # eval and test the model
    model.eval()
     # Configure
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0
    m_rec = 0
    map50 = 0
    mean_ap = 0
    metrics = []
    p_bar = tqdm(data_loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))
    with torch.no_grad():
        for images, targets in p_bar:
            images = images.cuda() # .half()  # uint8 to fp16/32

            _, _, h, w = images.shape  # batch-size, channels, height, width
            scale = torch.tensor((w, h, w, h)).cuda()
            # Inference
            outputs = model(images)
            # NMS
            outputs = non_max_suppression(outputs)
            # Metrics

            for i, output in enumerate(outputs):
                idx = targets['idx'] == i
                cls = targets['cls'][idx]
                box = targets['box'][idx]

                cls = cls.cuda()
                box = box.cuda()

                metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

                if output.shape[0] == 0:
                    if cls.shape[0]:
                        metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                    continue
                # Evaluate
                if cls.shape[0]:
                    target = torch.cat(tensors=(cls, wh2xy(box) * scale), dim=1)
                    metric = compute_metric(output[:, :6], target, iou_v)
                # Append
                metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    # Compute metrics
    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = compute_ap(*metrics)

    # Print results
    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
    return mean_ap, map50, m_rec, m_pre


def get_image_data(dir):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(dir, ext)))
    return image_files

def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--data', required=True)
    parser.add_argument('--pretrained', required=False)
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--lightweight', '-l', default=False, type=bool)
    args = parser.parse_args()

    # augmentation parameters
    params = {'flip_ud': 0.5, 'flip_lr': 0.5, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4}

    with open(args.data, 'r') as file:
        data = yaml.safe_load(file)

    model = yolo_v11_n(int(data['nc']))

    if args.pretrained:
        pretained = torch.load(args.pretrained, weights_only=False)['model'].state_dict()
        filtered_dict = {k: v for k, v in pretained.items() if k in model.state_dict() and "head.cls" not in k}
        model.load_state_dict(filtered_dict, strict=False)

    model.to(DEVICE)
    # optim, scheduler, lossfn
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    CosineAnnealingLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)
    loss_fn = ComputeLoss(model)

    print("Loading dataset...")
    traindataset = Yolo_Dataset(filenames=get_image_data(data['train']), 
                                          input_size=640, params=params, augment=True)
    
    train_dataloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, collate_fn=traindataset.collate_fn)

    valdataset = Yolo_Dataset(filenames=get_image_data(data['val']), input_size=640)

    valdataloader = DataLoader(valdataset, batch_size=args.batch_size*2, shuffle=True, num_workers=args.num_workers, collate_fn=valdataset.collate_fn)

    testdataset = Yolo_Dataset(filenames=get_image_data(data['test']), input_size=640)

    testdataloader = DataLoader(testdataset, batch_size=args.batch_size*2, shuffle=True, num_workers=args.num_workers, collate_fn=testdataset.collate_fn)
    print("Dataset loaded successfully.")
    
    print("Starting training process...")
    train(model, optimizer, train_dataloader, valdataloader, CosineAnnealingLR, DEVICE, epochs=args.epochs, loss_fn=loss_fn)

    print("Training process completed.")    

    print('testing model')
    
    mean_ap, map50, m_rec, m_pre = evaluate(model, testdataloader, loss_fn, DEVICE)

    print(f"tese results: mAP: {mean_ap:.4f}, mAP50: {map50:.4f}, m_rec: {m_rec:.4f}, m_pre: {m_pre:.4f}")



    if args.lightweight:
        # pruning
        # quantization
        # fine-tuning
        pass
    
    with open('weights/test_results.txt', 'w+') as f:
        f.write(f"tese results: mAP: {mean_ap:.4f}, mAP50: {map50:.4f}, m_rec: {m_rec:.4f}, m_pre: {m_pre:.4f}")
        f.close()

if __name__ == "__main__":
    main()
