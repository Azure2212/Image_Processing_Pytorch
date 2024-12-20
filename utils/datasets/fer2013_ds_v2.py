import os

import cv2
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import torchvision
import torch
from utils.augs.augmenters import seg_fer, seg_fertest1, seg_fertest2




EMOTION_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


class FERDataset(Dataset):
  def __init__(self, data_type, configs, ttau = False, len_tta = 48):
    self.data_type = data_type
    self.configs = configs
    self.ttau = ttau
    self.len_tta = len_tta

    self.shape = (configs["image_size"], configs["image_size"])

    data_path = self.configs['fer_path_train']
    if data_type == 'test':
      data_path = self.configs['fer_path_test']
          
    label_mapping = {'surprise':0, 'fear':1, 'disgust':2, 'happy':3, 'sad':4, 'angry':5, 'neutral':6}
    
    emotions = os.listdir(data_path)
    self.file_paths = []
    self.label = []
    for e in emotions:
      images = os.listdir(os.path.join(data_path, e))
      images = [os.path.join(data_path, e, i) for i in images]
      self.file_paths.extend(images)
      self.label.extend([label_mapping[e]] * len(images))

    self.transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Imagenet
            #transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2675, 0.2565, 0.2761]), # VGGface2

        ]
        )
    

  def is_ttau(self):
    return self.ttau == True

  def __len__(self):
    return len(self.file_paths)
    
  def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]
#         print(image.shape)
        image = cv2.resize(image, self.shape)
        
        if self.data_type == "train":
            image = seg_fer(image = image)
            #image = my_data_augmentation(image.copy())
        if self.data_type == "test" and self.ttau == True:
            images1 = [seg_fertest1(image=image) for i in range(self.len_tta)]
            images2 = [seg_fertest2(image=image) for i in range(self.len_tta)]

            images = images1 + images2
            # images = [image for i in range(self._tta_size)]
            images = list(map(self.transform, images))
            label = self.label[idx]
        
            return images, label

        image = self.transform(image)
        label = self.label[idx]
        
        return image, label