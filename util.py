import numpy as np
import torch
import torchvision
from time import time
import math

def smooth(y, f=0.1):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed

def compute_ap(tp, conf, output, target, plot=False, eps=1E-16):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Object-ness value from 0-1 (nparray).
        output:  Predicted object classes (nparray).
        target:  True object classes (nparray).
    # Returns
        The average precision
    """
    # Sort by object-ness
    i = np.argsort(-conf)
    tp, conf, output = tp[i], conf[i], output[i]

    # Find unique classes
    unique_classes, nt = np.unique(target, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    p = np.zeros((nc, 1000))
    r = np.zeros((nc, 1000))
    ap = np.zeros((nc, tp.shape[1]))
    px, py = np.linspace(start=0, stop=1, num=1000), []  # for plotting
    for ci, c in enumerate(unique_classes):
        i = output == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of outputs
        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = np.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = np.concatenate(([1.0], precision[:, j], [0.0]))

            # Compute the precision envelope
            m_pre = np.flip(np.maximum.accumulate(np.flip(m_pre)))

            # Integrate area under curve
            x = np.linspace(start=0, stop=1, num=101)  # 101-point interp (COCO)
            ap[ci, j] = np.trapz(np.interp(x, m_rec, m_pre), x)  # integrate
            if plot and j == 0:
                py.append(np.interp(px, m_rec, m_pre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap = ap50.mean(), ap.mean()
    return tp, fp, m_pre, m_rec, map50, mean_ap




def compute_metric(output, target, iou_v):
    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2) = target[:, 1:].unsqueeze(1).chunk(2, 2)
    (b1, b2) = output[:, :4].unsqueeze(0).chunk(2, 2)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    # IoU = intersection / (area1 + area2 - intersection)
    iou = intersection / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - intersection + 1e-7)

    correct = np.zeros((output.shape[0], iou_v.shape[0]))
    correct = correct.astype(bool)
    for i in range(len(iou_v)):
        # IoU > threshold and classes match
        x = torch.where((iou >= iou_v[i]) & (target[:, 0:1] == output[:, 5]))
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1),
                                 iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=output.device)



def non_max_suppression(outputs, confidence_threshold=0.25, iou_threshold=0.65): 
    max_wh = 7680
    max_det = 300
    max_nms = 30000

    bs = outputs.shape[0]  # batch size
    nc = outputs.shape[1] - 4  # number of classes
    xc = outputs[:, 4:4 + nc].amax(1) > confidence_threshold  # candidates

    # Settings
    start = time()
    limit = 0.5 + 0.05 * bs  # seconds to quit after
    output = [torch.zeros((0, 6), device=outputs.device)] * bs # [x1, y1, x2, y2, confidence, class_id]
    for index, x in enumerate(outputs):  # image index, image inference
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # matrix nx6 (box, confidence, cls)
        box, cls = x.split((4, nc), 1)
        box = wh2xy(box)  # (cx, cy, w, h) to (x1, y1, x2, y2)
        if nc > 1:
            i, j = (cls > confidence_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > confidence_threshold]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes, scores
        indices = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        indices = indices[:max_det]  # limit detections

        output[index] = x[indices]
        if (time() - start) > limit:
            break  # time limit exceeded

    return output

def wh2xy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y