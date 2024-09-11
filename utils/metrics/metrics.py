import torch
from torch import Tensor
import numpy as np

def accuracy(y_pred, labels):
    with torch.no_grad():
        batch_size = labels.size(0)
        pred = torch.argmax(y_pred, dim=1)
        correct = pred.eq(labels).float().sum(0)
        acc = correct * 100 / batch_size
    return [acc]


def make_batch(images):
    if not isinstance(images, list):
        images = [images]
    return torch.stack(images, 0)



#### Segmentation
def _get_class_data(gt_onehot, pred, class_num):
        # perform calculation on a batch
        # for precise result in a single image, plz set batch size to 1
        matrix = np.zeros((4, class_num))

        # calculate tp, fp, fn per class
        for i in range(class_num):
            # pred shape: (N, H, W)
            class_pred = pred[:, i, :, :]
            # gt shape: (N, H, W), binary array where 0 denotes negative and 1 denotes positive
            class_gt = gt_onehot[:, i, :, :]

            pred_flat = class_pred.contiguous().view(-1, )  # shape: (N * H * W, )
            gt_flat = class_gt.contiguous().view(-1, )  # shape: (N * H * W, )

            tp = torch.sum(gt_flat * pred_flat)
            tn = torch.sum((1 - gt_flat) * (1 - pred_flat))
            fp = torch.sum(pred_flat) - tp
            fn = torch.sum(gt_flat) - tp
            #print(f"Class {i}: TP = {tp.item()}, FP = {fp.item()}, FN = {fn.item()}, TN = {tn.item()}")
        
            matrix[:, i] = tp.item(), fp.item(), fn.item(), tn.item()

        return matrix

def calculate_multi_metrics2(gt, pred, class_num, average = True):
    # calculate metrics in multi-class segmentation
    eps = 1e-6
    matrix = _get_class_data(gt, pred, class_num)
    matrix = matrix[:, 1:]
    
    # tp = np.sum(matrix[0, :])
    # fp = np.sum(matrix[1, :])
    # fn = np.sum(matrix[2, :])
    # tn = np.sum(matrix[3, :])
        

    pixel_acc = (np.sum(matrix[0, :]) + np.sum(matrix[3, :]) + eps) / (np.sum(matrix[0, :]) + np.sum(matrix[1, :]) + np.sum(matrix[2, :]) + np.sum(matrix[3, :]))
    dice = (2 * matrix[0] + eps) / (2 * matrix[0] + matrix[1] + matrix[2] + eps)
    iou = (matrix[0] + eps) / (matrix[0]  +matrix[1] + matrix[2] + eps)
    precision = (matrix[0] + eps) / (matrix[0] + matrix[1] + eps)
    recall = (matrix[0] + eps) / (matrix[0] + matrix[2] + eps)

    if average:
        dice = np.average(dice)
        iou = np.average(iou)
        precision = np.average(precision)
        recall = np.average(recall)

    return pixel_acc, dice, iou, precision, recall

def calculate_multi_metrics(gt, pred, class_num, average = True):
    import segmentation_models_pytorch as smp
    tp, fp, fn, tn = smp.metrics.get_stats(pred.long(), gt.long(), mode="multiclass", num_classes=class_num)

    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

    pixel_acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")

    dice = (2*tp) / (2*tp + fp + fn)

    precision = smp.metrics.positive_predictive_value(tp, fp, fn, tn, reduction="micro")

    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
    return pixel_acc, dice, iou, precision, recall
