from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
import face_alignment
import pandas as pd
import os
import numpy as np
import random
import cv2
import torch
from sgu24project.utils.augs.face_alignment import transform, crop, get_preds_fromhm, _get_preds_fromhm, transform_np
from sgu24project.utils.augs.augmenters import make_augmentation_image_landmark_boundingbox_custom

class image_with_landmark_RafDataSet(Dataset):
    def __init__(self, data_type, configs, ttau=False, device='cpu'):
        self.device = device
        self.configs = configs
        self.data_type = data_type
        self.shape = (configs["image_size"], configs["image_size"])
        self.fa_model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu')

        df = pd.read_csv(os.path.join(self.configs["raf_path"], configs["label_path"]), sep=' ', header=None, names=['name', 'label'])

        if data_type == 'train':
            self.data = df[df['name'].str.startswith('train')]
        else:
            self.data = df[df['name'].str.startswith('test')]

        file_names = np.array(self.data.loc[:, 'name'].values)
        self.labels = np.array(self.data.loc[:, 'label'].values - 1)  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        # Initialize an empty list to store the file names
        path_current = os.path.abspath(globals().get("__file__","."))
        path_current = os.path.abspath(f"{path_current}/..")
        path_images_without_landmarks = []

        # Open the input file in read mode
        file2read = os.path.join(path_current,'rafdb_image_path_without_landmarks.txt')
        with open(file2read, 'r') as f:
            # Read each line in the file
            for line in f:
                # Strip any whitespace characters (like newline) and add to the list
                path_images_without_landmarks.append(line.strip().replace('_aligned', ''))
        
        check_in = ~np.isin(file_names, path_images_without_landmarks)
        file_names = file_names[check_in]
        self.labels = self.labels[check_in]

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

    def get_face_box_image(self, image):
        detected_faces = self.fa_model.face_detector.detect_from_image(image.copy())
        return detected_faces

    def get_landmarks(self, image, detected_faces):
        landmarks = []
        landmarks_scores = []
        if(detected_faces == None):
            print(f'path None: {path}')
            return None
        
        def get_landmarks(image, detected_faces):
            landmarks = []
            landmarks_scores = []
            for i, d in enumerate(detected_faces):
                center = np.array(
                    [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
                center[1] = center[1] - (d[3] - d[1]) * 0.12
                scale = (d[2] - d[0] + d[3] - d[1]) / fa_model.face_detector.reference_scale
                inp = crop(image, center, scale)
                inp = torch.from_numpy(inp.transpose(
                    (2, 0, 1))).float()
                inp = inp.to(device ='cpu', dtype=torch.float32)
                inp.div_(255.0).unsqueeze_(0)
                out = fa.face_alignment_net(inp).detach()
                # print(out.shape)
                # out = out.to(device='cpu', dtype=torch.float32).numpy()
                # print(out.shape)
                # pts, pts_img, scores = get_preds_fromhm(out, center, scale)
                # print(pts.shape)
                # pts, pts_img = torch.from_numpy(pts), torch.from_numpy(pts_img)
                # print(pts.shape)
                # pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
                # scores = scores.squeeze(0)
                # print(pts.shape)
                # landmarks.append(pts_img.numpy())
                # landmarks_scores.append(scores)
                return out
        
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)[:,:,::-1]
        image, detected_faces = make_augmentation_image_landmark_boundingbox_custom(image.copy(), task='resize')
        landmarks  = self.get_landmarks(image.copy(), detected_faces)
        if(landmarks == None):
            print(path)
        if self.data_type == 'train':
            image = make_augmentation_image_landmark_boundingbox_custom(image.copy())
            #landmarks = landmarks[0]

        image = self.transform(image)
        return image, landmarks