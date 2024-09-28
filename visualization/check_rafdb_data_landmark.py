import tqdm 
import matplotlib.colors as mcolors
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sgu24project.utils.datasets.rafdb_ds_with_mask_v2 import RafDataSet_Mask
from sgu24project.utils.augs.augmenters import make_augmentation_image_landmark_boundingbox_custom

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--type-data', default= "train", type=str, help='type data')

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
        return out


configs = {"batch_size":1, 
          "raf_path": "/kaggle/input/rafdb-mask-basic-15k3",
    "image_path": "rafdb_mask_basic/Image/aligned/",
    "label_path": "rafdb_mask_basic/EmoLabel/list_patition_label.txt",
    'removed_image_path':"/kaggle/working/sgu24project/utils/datasets/rafdb_image_path_without_landmarks.txt",
    "image_size": 256,
    "n_channels": 3,
    "n_classes": 7,
          }
device = device
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


image = cv2.imread(file_paths[0])[:,:,::-1]
image, detected_faces = make_augmentation_image_landmark_boundingbox_custom(image.copy(), task='resize')

image, detected_faces = make_augmentation_image_landmark_boundingbox_custom(image.copy(), task='resize')
landmarks  = get_landmarks(image.copy(), detected_faces)
if(landmarks == None):
    print(path)
if data_type == 'train':
    image = make_augmentation_image_landmark_boundingbox_custom(image.copy())


ax, fig = plt.subplots(args.batch_size,2,figsize=(12,16))
a[0].imshow(image)
a[0].set_title('image')
a[0].axis('off')

for landmark in detected_faces[0]:
    a[1].scatter(landmark[0], landmark[1], color='red', s=10)

a[1].imshow(image)
a[1].set_title('image')
a[1].axis('off')




