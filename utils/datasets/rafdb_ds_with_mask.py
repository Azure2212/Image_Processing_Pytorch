import cv2
from PIL import Image
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import torchvision
import torch
import os 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from sgu24project.utils.augs.augmenters import seg_raf , seg_raftest1, seg_raftest2, CustomTransform
from sgu24project.utils.augs.mask_segmentation import mediapipe_tool

import warnings
warnings.filterwarnings('ignore')
class RafDataSet_Mask(Dataset):
    def __init__(self, data_type, configs, ttau=False, len_tta=48, use_albumentation=True):
        self.use_albumentation = use_albumentation
        self.data_type = data_type
        self.configs = configs
        self.ttau = ttau
        self.len_tta = len_tta
        self.shape = (configs["image_size"], configs["image_size"])

        df = pd.read_csv(os.path.join(self.configs["raf_path"], configs["label_path"]), sep=' ', header=None, names=['name', 'label'])

        if data_type == 'train':
            self.data = df[df['name'].str.startswith('train')]
        else:
            self.data = df[df['name'].str.startswith('test')]

        file_names = self.data.loc[:, 'name'].values
        self.labels = self.data.loc[:, 'label'].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        _, self.sample_counts = np.unique(self.labels, return_counts=True)
        # print(f' distribution of {data_type} samples: {self.sample_counts}')

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0] + "_aligned.jpg"
            path = os.path.join(self.configs["raf_path"], self.configs["image_path"], f)
            self.file_paths.append(path)

        self.transform_image = A.Compose([
            A.Resize(width=224, height=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        self.transform = A.Compose([
            A.Resize(width=224, height=224),
            ToTensorV2(),
        ])
        self.train_augmentation = A.Compose([ 
            A.Rotate(limit=25, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ShiftScaleRotate(scale_limit=0.05, rotate_limit=0, shift_limit=0, p=0.5),
            ToTensorV2(),
        ])
        
        self.seg_raftest2 = A.Compose([
            # No direct equivalent for RemoveSaturation, so it is skipped
             A.Affine(scale=(1.0, 1.05)),          # Scale the image
             ToTensorV2()
        ])

        self.seg_raftest1 = A.Compose([
            A.HorizontalFlip(p=1),  # Horizontal flip
            A.Affine(rotate=(-25, 25)),  # Rotate by -25 to 25 degrees
            ToTensorV2()
        ])
        
        self.custom_transform = CustomTransform()
        self.mediapipe_tool = mediapipe_tool()
    def __len__(self):
        return len(self.file_paths)
    
    def is_ttau(self):
        return self.ttau
    
     def get_mask(self, masks_dir):      
        print(f'masks_dir:{masks_dir}')
        eyebrows_mask = cv2.imread(f'{masks_dir}/eyebrows_mask.jpg', cv2.IMREAD_GRAYSCALE)
        eyes_mask = cv2.imread(f'{masks_dir}/eyes_mask.jpg', cv2.IMREAD_GRAYSCALE)
        nose_mask = cv2.imread(f'{masks_dir}/nose_mask.jpg', cv2.IMREAD_GRAYSCALE)
        mouth_mask = cv2.imread(f'{masks_dir}/mouth_mask.jpg', cv2.IMREAD_GRAYSCALE)
        face_mask = cv2.imread(f'{masks_dir}/face_mask.jpg', cv2.IMREAD_GRAYSCALE)

        H, W = eyes_mask.shape
        combined_mask = np.zeros((H, W), dtype=np.uint8)
        combined_mask[face_mask == 255] = 5
        combined_mask[eyes_mask == 255] = 1
        combined_mask[eyebrows_mask == 255] = 2
        combined_mask[nose_mask == 255] = 3
        combined_mask[mouth_mask == 255] = 4

        return combined_mask
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
         mask = self.get_mask(masks_dir = path.replace('/Image/aligned/', '/Image/masks/').split('.')[0]) 
        image = image[:, :, ::-1]  # Convert BGR to RGB
          # Assuming this function is defined elsewhere
        if self.data_type == "test" and self.ttau:
            image_copy = image.copy()
            mask_copy = mask.copy()

            # Apply seg_raftest1 to the copied image and mask
            transformed1 = seg_raftest1(image=image_copy, mask=mask_copy)
            image1 = transformed1['image']
            mask1 = transformed1['mask']
            
            # Make copies of the original image and mask for seg_raftest2
            image_copy2 = image.copy()
            mask_copy2 = mask.copy()

            # Apply seg_raftest2 to the copied image and mask
            transformed2 = seg_raftest2(image=image_copy2, mask=mask_copy2)
            image2 = transformed2['image']
            mask2 = transformed2['mask']
            # Combine the transformed images and masks
            images = [image1, image2]
            masks = [mask1, mask2]

            # Fetch the label
            label = self.labels[idx]

            return images, masks, label
        
        label = self.labels[idx]
        transformed = self.custom_transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed.get('mask', None)
    
        if self.data_type == "train":
            if self.use_albumentation:
                image = image.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
                #mask = mask.permute(0,1).numpy()
                mask = mask.squeeze(0).numpy()
                augmented = self.train_augmentation(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
    
        return image, mask, label


