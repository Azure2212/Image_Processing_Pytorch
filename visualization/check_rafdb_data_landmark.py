import tqdm 
import matplotlib.colors as mcolors
import cv2
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sgu24project.utils.datasets.rafdb_ds_with_mask_v2 import RafDataSet_Mask
from sgu24project.utils.augs.face_alignment import transform, crop, get_preds_fromhm, _get_preds_fromhm, transform_np
import face_alignment
import argparse 
import torch
from sgu24project.utils.augs.augmenters import make_augmentation_image_landmark_boundingbox_custom
from torchvision.transforms import transforms
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--data-type', default= "train", type=str, help='type data')
parser.add_argument('--batch-size', default= 1, type=int, help='1')
parser.add_argument('--device', default= "cpu", type=str, help='gpu or cpu')
args, unknown = parser.parse_known_args()

device = args.device
fa_model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)

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
        inp = inp.to(device = device, dtype=torch.float32)
        inp.div_(255.0).unsqueeze_(0)
        out = fa_model.face_alignment_net(inp).detach()
        return out


configs = {"batch_size":1, 
          "raf_path": "/kaggle/input/rafdb-basic/rafdb_basic",
    "image_path": "Image/aligned/",
    "label_path": "EmoLabel/list_patition_label.txt",
    'removed_image_path':"/kaggle/working/sgu24project/utils/datasets/rafdb_image_path_without_landmarks.txt",
    "image_size": 256,
    "n_channels": 3,
    "n_classes": 7,
          }

configs = configs
data_type = args.data_type

shape = (configs["image_size"], configs["image_size"])
fa_model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu')

df = pd.read_csv(os.path.join(configs["raf_path"], configs["label_path"]), sep=' ', header=None, names=['name', 'label'])

if data_type == 'train':
    data = df[df['name'].str.startswith('train')]
else:
    data = df[df['name'].str.startswith('test')]

file_names = np.array(data.loc[:, 'name'].values)
labels = np.array(data.loc[:, 'label'].values - 1)  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

# Initialize an empty list to store the file names
path_images_without_landmarks = []

# Open the input file in read mode
file2read = configs['removed_image_path']
with open(file2read, 'r') as f:
    # Read each line in the file
    for line in f:
        # Strip any whitespace characters (like newline) and add to the list
        path_images_without_landmarks.append(line.strip().replace('_aligned', ''))

check_in = ~np.isin(file_names, path_images_without_landmarks)
file_names = file_names[check_in]
labels = labels[check_in]

file_paths = []
for f in file_names:
    f = f.split(".")[0]
    f = f +"_aligned.jpg"
    path = os.path.join(configs["raf_path"], configs["image_path"], f)
    file_paths.append(path)



transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            # std=[0.229, 0.224, 0.225]),
        ])

fig, ax = plt.subplots(args.batch_size,2,figsize=(12, args.batch_size *6))
for i in range(args.batch_size):
   
    path = file_paths[i]
    image = cv2.imread(path)[:,:,::-1]
    image, detected_faces = make_augmentation_image_landmark_boundingbox_custom(image.copy(), task='resize')
    feature_landmarks  = get_landmarks(image.copy(), detected_faces)
    if(feature_landmarks == None):
        print(path)
    if data_type == 'train':
        image = make_augmentation_image_landmark_boundingbox_custom(image.copy(), task='image_change')

    image = transform(image)
    
    image2show = np.transpose(image.numpy(), (1, 2, 0))
    ax[i, 0].imshow(image2show)
    ax[i, 0].set_title('image')
    ax[i, 0].axis('off')

    landmarks = []
    landmarks_scores = []
    for j, d in enumerate(detected_faces):
        center = np.array([d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
        center[1] = center[1] - (d[3] - d[1]) * 0.12
        scale = (d[2] - d[0] + d[3] - d[1]) / fa_model.face_detector.reference_scale
        feature_landmarks = feature_landmarks.to(device=device, dtype=torch.float32).numpy()
        pts, pts_img, scores = get_preds_fromhm(feature_landmarks, center, scale)
        pts, pts_img = torch.from_numpy(pts), torch.from_numpy(pts_img)
        pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
        scores = scores.squeeze(0)
        landmarks.append(pts_img.numpy())
        landmarks_scores.append(scores)

    for landmark in landmarks[0]:
        ax[i, 1].scatter(landmark[0], landmark[1], color='red', s=10)

    ax[i, 1].imshow(image2show)
    ax[i, 1].set_title('Detected LMark')
    ax[i, 1].axis('off')




