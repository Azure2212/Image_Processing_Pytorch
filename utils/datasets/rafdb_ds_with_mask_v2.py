from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
import pandas as pd
import os
import numpy as np
import cv2

def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        
        #A.PadIfNeeded(min_height=320, min_width=320, always_apply=True),
        #A.RandomCrop(height=320, width=320, always_apply=True),
        
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(384, 480),
    ]
    return A.Compose(test_transform)

class RafDataSet_Mask(Dataset):
    def __init__(self, data_type, configs, ttau=False, classes=None):
        if data_type == 'train':
            self.augmentation = get_training_augmentation()
        else:
            self.augmentation = None
        self.configs = configs
        self.shape = (configs["image_size"], configs["image_size"])

        df = pd.read_csv(os.path.join(self.configs["raf_path"], configs["label_path"]), sep=' ', header=None, names=['name', 'label'])

        if data_type == 'train':
            self.data = df[df['name'].str.startswith('train')]
        else:
            self.data = df[df['name'].str.startswith('test')]

        file_names = self.data.loc[:, 'name'].values
        self.labels = self.data.loc[:, 'label'].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        #_, self.sample_counts = np.unique(self.labels, return_counts=True)
        # print(f' distribution of {data_type} samples: {self.sample_counts}')
        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.configs["raf_path"], self.configs["image_path"], f)
            self.file_paths.append(path)
    
        self.classes = classes #['eyes_mask', 'eyebrows_mask', 'nose_mask', 'mouth_mask', 'face_mask']
    def __len__(self):
        return len(self.file_paths)
    
    def read_mask_of_image(self, masks_dir, mask_name):
        image = cv2.imread(f'{masks_dir}/{mask_name}.jpg', cv2.IMREAD_GRAYSCALE)
        image = np.where(image > 125, 255, image)
        image = np.where(image <= 125, 0, image)
        image = image / 255.0
        return image

    def get_mask(self, masks_dir):      
        
        eyes_mask = self.read_mask_of_image(masks_dir, self.classes[0])
        eyebrows_mask = self.read_mask_of_image(masks_dir, self.classes[1])
        nose_mask = self.read_mask_of_image(masks_dir, self.classes[2])
        mouth_mask = self.read_mask_of_image(masks_dir, self.classes[3])
        face_mask = self.read_mask_of_image(masks_dir, self.classes[4])

        face_is_one = (face_mask == 1.0)

        any_other_mask_is_one = (
        (eyes_mask == 1.0) |
        (eyebrows_mask == 1.0) |
        (nose_mask == 1.0) |
        (mouth_mask == 1.0)
        )
    
        # Update face_mask where conditions are met
        face_mask[face_is_one & any_other_mask_is_one] = 0.0

        masks = [eyes_mask, eyebrows_mask, nose_mask, mouth_mask, face_mask]

        masks_stack = np.stack(masks, axis=-1)

        # Tạo mặt nạ nền
        background_mask = np.all(masks_stack == 0, axis=-1).astype(np.uint8)

        # Thêm mặt nạ nền vào mảng mặt nạ
        mask = np.concatenate([masks_stack, background_mask[:, :, np.newaxis]], axis=-1).astype('float')

        return mask # background:0, eyes:1, eyebrows_mask:2, nose_mask:3, mouth_mask:4, face_mask:5
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        mask = self.get_mask(masks_dir = path.replace('/Image/aligned/', '/Image/masks/').split('.')[0]) 
        
        image = cv2.resize(image, self.shape)
        mask = cv2.resize(mask, self.shape, interpolation=cv2.INTER_NEAREST)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image.transpose(2, 0, 1), mask.transpose(2, 0, 1)
       
        return image, mask, label