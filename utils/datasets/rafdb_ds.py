import os

import cv2
from PIL import Image
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import albumentations as A
import torchvision
import torch
import random
from sgu24project.utils.augs.augmenters import seg_raf , seg_raftest1, seg_raftest2

def my_data_augmentation(image): 
   
    def A_Perspective(image):
        h, w, _ = image.shape
        value = 30
        # Định nghĩa các điểm nguồn và điểm đích cho biến đổi phối cảnh
        src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst_points = np.float32([[random.uniform(-value, value), random.uniform(-value, value)],
                                  [w + random.uniform(-value, value), random.uniform(-value, value)],
                                  [random.uniform(-value, value), h + random.uniform(-value, value)],
                                  [w + random.uniform(-value, value), h + random.uniform(-value, value)]])

        # Tính toán ma trận biến đổi phối cảnh
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Áp dụng biến đổi cho hình ảnh
        transformed_image = cv2.warpPerspective(image, matrix, (w, h))

        return transformed_image
    
    def A_Rotate(my_image):
        # Chọn góc quay ngẫu nhiên từ -45 đến 45 độ
        angle = np.random.uniform(-45, 45)
     
        # Quay hình ảnh
        (h, w) = my_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(my_image, M, (w, h))

        return rotated_image

    if random.random() < 0.5:
        if random.random() < 0.5:
            image = cv2.flip(image.copy(), 1) #Horizontal_flip
        if random.random() < 0.5:
            random_number = cv2.flip(image.copy(), 0) #Vertical_flip
        if random.random() < 0.5:    
            image = A_Rotate(image.copy())

        if random.random() < 0.5:
            image = A_Perspective(image.copy())

        if random.random() < 0.9:
            random_number = random.choice([0, 1, 2, 4])
            if random_number == 1:
                transform = A.Compose([A.CLAHE(p=1.0, clip_limit=2.0, tile_grid_size=(8, 8))])
            elif random_number == 2:
                transform = A.Compose([A.RandomBrightnessContrast(p=1)])
            elif random_number == 3:
                transform = A.Compose([A.RandomGamma(p=1)])
            if random_number != 0:
                augmented = transform(image=image)
                image = augmented['image']
            
        if random.random() < 0.5:
            random_number = random.choice([0, 1, 2, 3])
            if random_number == 1:
                transform = A.Compose([A.Sharpen(p=1)])
            elif random_number == 2:
                transform = A.Compose([A.Blur(blur_limit=3, p=1)])
            elif random_number == 3:
                transform = A.Compose([A.MotionBlur(blur_limit=3, p=1)])
            if random_number != 0:
                augmented = transform(image=image)
                image = augmented['image']
            
        if random.random() < 0.5:
            random_number = random.choice([0, 1, 2])
            if random_number == 1:
                transform = A.Compose([A.RandomBrightnessContrast(p=1)])
            elif random_number == 2:
                transform = A.Compose([A.HueSaturationValue(p=1)])
            if random_number != 0:
                augmented = transform(image=image)
                image = augmented['image']
    return image

class RafDataSet(Dataset):
    def __init__(self, data_type, configs,  ttau = False, len_tta = 48, use_albumentation = True):
        self.use_albumentation = use_albumentation
        self.data_type = data_type
        self.configs = configs
        self.ttau = ttau
        self.len_tta = len_tta
        self.shape = (configs["image_size"], configs["image_size"])

        df = pd.read_csv(os.path.join(self.configs["raf_path"],configs["label_path"]), sep=' ', header=None,names=['name','label'])

        if data_type == 'train':
            self.data = df[df['name'].str.startswith('train')]
        else:
            self.data = df[df['name'].str.startswith('test')]

        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:, 'label'].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {data_type} samples: {self.sample_counts}')

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.configs["raf_path"], self.configs["image_path"], f)
            self.file_paths.append(path)

        self.transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),

        ]
        )

    def __len__(self):
        return len(self.file_paths)
    
    def is_ttau(self):
        return self.ttau == True

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]
#         print(image.shape)
#         image = cv2.resize(image, self.shape)
        
        if self.data_type == "train":
            #image = seg_raf(image = image)
            image = my_data_augmentation(image.copy())
        if self.data_type == "test" and self.ttau == True:
            images1 = [seg_raftest1(image=image) for i in range(self.len_tta)]
            images2 = [seg_raftest2(image=image) for i in range(self.len_tta)]

            images = images1 + images2
            # images = [image for i in range(self._tta_size)]
            images = list(map(self.transform, images))
            label = self.label[idx]
        
            return images, label

        image = self.transform(image)
        label = self.label[idx]
        
        return image, label
