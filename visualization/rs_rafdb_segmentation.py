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
test_loader = DataLoader(test_dataset, batch_size= args.number_image_test,num_workers=1,
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
print(images.shape)
print(f'number_image_test: {args.number_image_test}')
end = min(args.number_image_test, len(images))

# fig, ax = plt.subplots(end, 13, figsize=(20, 6))
# for i in range(end):
#     image_np = images[i].cpu().numpy().transpose(1, 2, 0)
#     image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
#     ax[i, 0].imshow(image_np)
#     ax[i, 0].set_title('image')
#     ax[i, 0].axis('off')

#     ax[i, 1].imshow(masks[i][0].cpu().numpy(), cmap='gray')
#     ax[i, 1].set_title(f"{CLASSES[0]} (gt)")
#     ax[i, 1].axis('off')

#     ax[i, 2].imshow(pr_masks[i][0].cpu().numpy(), cmap='gray')
#     ax[i, 2].set_title('pd')
#     ax[i, 2].axis('off')

#     ax[i, 3].imshow(masks[i][1].cpu().numpy(), cmap='gray')
#     ax[i, 3].set_title(f"{CLASSES[1]} (gt)")
#     ax[i, 3].axis('off')

#     ax[i, 4].imshow(pr_masks[i][1].cpu().numpy(), cmap='gray')
#     ax[i, 4].set_title('pd')
#     ax[i, 4].axis('off')

#     ax[i, 5].imshow(masks[i][2].cpu().numpy(), cmap='gray')
#     ax[i, 5].set_title(f"{CLASSES[2]} (gt)")
#     ax[i, 5].axis('off')

#     ax[i, 6].imshow(pr_masks[i][2].cpu().numpy(), cmap='gray')
#     ax[i, 6].set_title('pd')
#     ax[i, 6].axis('off')

#     ax[i, 7].imshow(masks[i][3].cpu().numpy(), cmap='gray')
#     ax[i, 7].set_title(f"{CLASSES[3]} (gt)")
#     ax[i, 7].axis('off')

#     ax[i, 8].imshow(pr_masks[i][3].cpu().numpy(), cmap='gray')
#     ax[i, 8].set_title('pd')
#     ax[i, 8].axis('off')

#     ax[i, 9].imshow(masks[i][4].cpu().numpy(), cmap='gray')
#     ax[i, 9].set_title(f"{CLASSES[4]} (gt)")
#     ax[i, 9].axis('off')

#     ax[i, 10].imshow(pr_masks[i][4].cpu().numpy(), cmap='gray')
#     ax[i, 10].set_title('pd')
#     ax[i, 10].axis('off')

#     ax[i, 11].imshow(masks[i][5].cpu().numpy(), cmap='gray')
#     ax[i, 11].set_title(f"{CLASSES[5]} (gt)")
#     ax[i, 11].axis('off')

#     ax[i, 12].imshow(pr_masks[i][5].cpu().numpy(), cmap='gray')
#     ax[i, 12].set_title('pd')
#     ax[i, 12].axis('off')

# plt.tight_layout()
# plt.show()




for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, masks, pr_masks)):
    # Number of samples visualized
    if idx <= args.number_image_test:
        print(f'vo day {idx}')
        plt.figure(figsize=(20, 6))
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
        plt.title(f"(pt)")
        plt.axis("off")

        plt.subplot(1, 13, 4)
        plt.imshow(gt_mask[1].cpu().numpy().squeeze())
        plt.title(plt.title(f"{CLASSES[1]}(gt)"))
        plt.axis("off")

        plt.subplot(1, 13, 5)
        plt.imshow(pr_mask[1].cpu().numpy().squeeze())
        plt.title(f"(pt)")
        plt.axis("off")
        
        plt.subplot(1, 13, 6)
        plt.imshow(gt_mask[2].cpu().numpy().squeeze())
        plt.title(plt.title(f"{CLASSES[2]}(gt)"))
        plt.axis("off")

        plt.subplot(1, 13, 7)
        plt.imshow(pr_mask[2].cpu().numpy().squeeze())
        plt.title(f"(pt)")
        plt.axis("off")
        
        plt.subplot(1, 13, 8)
        plt.imshow(gt_mask[3].cpu().numpy().squeeze())
        plt.title(plt.title(f"{CLASSES[3]}(gt)"))
        plt.axis("off")

        plt.subplot(1, 13, 9)
        plt.imshow(pr_mask[3].cpu().numpy().squeeze())
        plt.title(f"(pt)")
        plt.axis("off")
        
        plt.subplot(1, 13, 10)
        plt.imshow(gt_mask[4].cpu().numpy().squeeze())
        plt.title(plt.title(f"{CLASSES[4]}(gt)"))
        plt.axis("off")

        plt.subplot(1, 13, 11)
        plt.imshow(pr_mask[4].cpu().numpy().squeeze())
        plt.title(f"(pt)")
        plt.axis("off")
        
        plt.subplot(1, 13, 12)
        plt.imshow(gt_mask[5].cpu().numpy().squeeze())
        plt.title("bg(gt)")
        plt.axis("off")

        plt.subplot(1, 13, 13)
        plt.imshow(pr_mask[5].cpu().numpy().squeeze())
        plt.title("(pt)")
        plt.axis("off")
        plt.show()
    else:
        break