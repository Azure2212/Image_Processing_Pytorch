from sgu24project.models.segmentation_models_pytorch.model import Resnet50UnetMultitask_v2 
from sgu24project.utils.datasets.rafdb_ds_with_mask_v2 import RafDataSet_Mask
import segmentation_models_pytorch as smp
import torchvision.models as models
import matplotlib.pyplot as plt
import imgaug
import random
import torch
import numpy as np

import cv2

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--rs-dir', default= "/kaggle/working/ResnetDuck_Cbam_cuaTuan", type=str, help='rs dir in kaggle')
parser.add_argument('--number-image-test', default= 5, type=int, help='number_image_test')
parser.add_argument('--model-name', default= "Resnet50UnetMultitask_v2", type=str, help='model2load')
args, unknown = parser.parse_known_args()

seed = 113
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


CLASSES = ['eyes_mask', 'eyebrows_mask', 'nose_mask', 'mouth_mask', 'face_mask']
OUT_CLASSES = len(CLASSES) + 1 
model = Resnet50UnetMultitask_v2(in_channels=3, num_seg_classes=OUT_CLASSES, num_cls_classes=7)

test_dataset = RafDataSet_Mask(data_type = 'test', configs = configs , classes=CLASSES)

# Khởi tạo mô hình
model = Resnet50UnetMultitask_v2(in_channels=3, num_seg_classes=6, num_cls_classes=7)

# Tải trọng số
state = torch.load(args.rs_dir)
model.load_state_dict(state["net"])

params = smp.encoders.get_preprocessing_params("resnet50")
std = torch.tensor(params["std"]).view(1, 3, 1, 1)
mean = torch.tensor(params["mean"]).view(1, 3, 1, 1)
images, masks = next(iter(test_loader))
with torch.no_grad():
    model = model.float().cuda()
    images = (images - mean) / std
    images = images.to(dtype=torch.float).cuda()
    masks = masks.to(dtype=torch.float).cuda()
    model.eval()
    seg_logits = model(images)
pr_masks = seg_logits.sigmoid()
pr_masks = (pr_masks > 0.5).float()
print(pr_masks.shape)
for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, masks, pr_masks)):
    # Number of samples visualized
    if idx <= args.number_image_test:
        plt.figure(figsize=(20, args.number_image_test))
        plt.subplot(1, 13, 1)
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        plt.imshow(image_np)
        plt.title("Image")
        plt.axis("off")
        
        plt.subplot(1, 13, 2)
        plt.imshow(gt_mask[0].cpu().numpy().squeeze())
        plt.title(f"{CLASSES[0]}(gt)")
        plt.axis("off")

        plt.subplot(1, 13, 3)
        plt.imshow(pr_mask[0].cpu().numpy().squeeze())
        plt.title(f"{CLASSES[0]}(pt)")
        plt.axis("off")

        plt.subplot(1, 13, 4)
        plt.imshow(gt_mask[1].cpu().numpy().squeeze())
        plt.title(plt.title(f"{CLASSES[1]}(gt)"))
        plt.axis("off")

        plt.subplot(1, 13, 5)
        plt.imshow(pr_mask[1].cpu().numpy().squeeze())
        plt.title(f"{CLASSES[1]}(pt)")
        plt.axis("off")
        
        plt.subplot(1, 13, 6)
        plt.imshow(gt_mask[2].cpu().numpy().squeeze())
        plt.title(plt.title(f"{CLASSES[2]}(gt)"))
        plt.axis("off")

        plt.subplot(1, 13, 7)
        plt.imshow(pr_mask[2].cpu().numpy().squeeze())
        plt.title(f"{CLASSES[2]}(pt)")
        plt.axis("off")
        
        plt.subplot(1, 13, 8)
        plt.imshow(gt_mask[3].cpu().numpy().squeeze())
        plt.title(plt.title(f"{CLASSES[3]}(gt)"))
        plt.axis("off")

        plt.subplot(1, 13, 9)
        plt.imshow(pr_mask[3].cpu().numpy().squeeze())
        plt.title(f"{CLASSES[3]}(pt)")
        plt.axis("off")
        
        plt.subplot(1, 13, 10)
        plt.imshow(gt_mask[4].cpu().numpy().squeeze())
        plt.title(plt.title(f"{CLASSES[4]}(gt)"))
        plt.axis("off")

        plt.subplot(1, 13, 11)
        plt.imshow(pr_mask[4].cpu().numpy().squeeze())
        plt.title(f"{CLASSES[4]}(pt)")
        plt.axis("off")
        
        plt.subplot(1, 13, 12)
        plt.imshow(gt_mask[5].cpu().numpy().squeeze())
        plt.title("background(gt)")
        plt.axis("off")

        plt.subplot(1, 13, 13)
        plt.imshow(pr_mask[5].cpu().numpy().squeeze())
        plt.title("background(gt)")
        plt.axis("off")
        plt.show()
    else:
        break