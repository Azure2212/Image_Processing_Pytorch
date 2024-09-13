from sgu24project.models.segmentation_models_pytorch.model import Resnet50UnetMultitask_v2 
from sgu24project.utils.datasets.rafdb_ds_with_mask_v2 import RafDataSet_Mask
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
import imgaug
import random
import torch
import numpy as np
import os
import json

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

config_path = "sgu24project/configs/config_rafdb.json"

configs = json.load(open(config_path))
#CLASSES = ['car', 'road', 'pavement', 'building', 'unlabelled']
CLASSES = ['eyes_mask', 'eyebrows_mask', 'nose_mask', 'mouth_mask', 'face_mask']

configs["num_seg_classes"] = len(CLASSES)+1

CLASSES = ['eyes_mask', 'eyebrows_mask', 'nose_mask', 'mouth_mask', 'face_mask']
OUT_CLASSES = len(CLASSES) + 1 
model = Resnet50UnetMultitask_v2(in_channels=3, num_seg_classes=OUT_CLASSES, num_cls_classes=7)

test_dataset = RafDataSet_Mask(data_type = 'test', configs = configs , classes=CLASSES)
test_loader = DataLoader(test_dataset, batch_size= 1,num_workers=1,
                      pin_memory=True, shuffle=False)
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

print(f'number_image_test: {args.number_image_test}')
for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, masks, pr_masks)):
    if idx < args.number_image_test:  # idx < number_image_test to process the correct number of images
        plt.figure(figsize=(20, 5))  # Adjust figure size as needed
        
        # Image
        plt.subplot(1, 13, 1)
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        plt.imshow(image_np)
        plt.title("Image")
        plt.axis("off")
        
        # Ground Truth Masks
        for i in range(len(gt_mask)):
            plt.subplot(1, 13, i + 2)
            plt.imshow(gt_mask[i].cpu().numpy().squeeze())
            plt.title(f"{CLASSES[i]}(gt)")
            plt.axis("off")
        
        # Prediction Masks
        for i in range(len(pr_mask)):
            plt.subplot(1, 13, len(gt_mask) + i + 2)
            plt.imshow(pr_mask[i].cpu().numpy().squeeze())
            plt.title(f"(pt)")
            plt.axis("off")

        plt.show()
    else:
        break