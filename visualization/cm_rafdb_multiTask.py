import os
import json
import random
import warnings
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
warnings.simplefilter(action="ignore", category=FutureWarning)

import random
import imgaug
import torch
import torch.multiprocessing as mp
import numpy as np
import tqdm
import torch.nn as nn
import torch.nn.functional as F

seed = 113
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from sgu24project.models.segmentation_models_pytorch.model import Resnet50UnetMultitask_v2 

from sgu24project.utils.datasets.rafdb_ds_with_mask_v2 import RafDataSet_Mask
#from sgu24project.utils.metrics.metrics import accuracy, make_batch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--rs-dir', default= "ResnetDuck_Cbam_cuaTuan", type=str, help='rs dir in kaggle')
parser.add_argument('--model-name', default= "resnet50", type=str, help='model2load')

args, unknown = parser.parse_known_args()
path_current = os.path.abspath(globals().get("__file__","."))
script_dir  = os.path.dirname(path_current)
root_path = os.path.abspath(f"{script_dir}/../")
config_path = root_path+"/configs/config_rafdb.json"  # Adjust the path as needed

configs = json.load(open(config_path))
CLASSES = ['eyes_mask', 'eyebrows_mask', 'nose_mask', 'mouth_mask', 'face_mask']

test_loader_ttau =  RafDataSet_Mask(data_type = 'test', configs = configs , classes=CLASSES)
OUT_CLASSES = len(CLASSES) + 1 
model = Resnet50UnetMultitask_v2(in_channels=3, num_seg_classes=OUT_CLASSES, num_cls_classes=7)

def make_batch(images):
    if not isinstance(images, list):
        images = [images]
    return torch.stack(images, 0)



# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=mean, std=std)
# ])
Mean = [0.229, 0.224, 0.225]
Std = [0.485, 0.456, 0.406]

def plot_confusion_matrix(model, testloader,title = "My model"):
    model.cuda()
    model.eval()
    correct = 0
    total = 0
    all_target = []
    all_output = []

    params = smp.encoders.get_preprocessing_params("resnet50")
    std = torch.tensor(params["std"]).view(1, 3, 1, 1)
    mean = torch.tensor(params["mean"]).view(1, 3, 1, 1)
    # test_set = fer2013("test", configs, tta=True, tta_size=8)
    # test_set = fer2013('test', configs, tta=False, tta_size=0)
    with torch.no_grad():
        for idx in tqdm.tqdm(range(len(testloader)), total=len(testloader), leave=False):
            images, masks, labels = testloader[idx]
            images = torch.from_numpy(images)
           
            images = make_batch(images)
            images = (images - mean) / std

            images = images.cuda(non_blocking=True)

            seg_preds, cls_preds = model(images)
            cls_preds = F.softmax(cls_preds, 1)

            # cls_preds.shape [tta_size, 7]
            cls_preds = torch.sum(cls_preds, 0)
            cls_preds = torch.argmax(cls_preds, 0)
            cls_preds = cls_preds.item()
            labels = labels.item()
            total += 1
            correct += cls_preds == labels

            all_target.append(labels)
            all_output.append(cls_preds)

    
    cf_matrix = confusion_matrix(all_target, all_output)
    cmn = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    class_names = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]
    #0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

    # Create pandas dataframe
    dataframe = pd.DataFrame(cmn, index=class_names, columns=class_names)
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(dataframe, annot=True, cbar=True,cmap="Blues",fmt=".2f")
    
    plt.title(title), plt.tight_layout()
    
    plt.ylabel("True Class", fontsize=12), 
    plt.xlabel("Predicted Class", fontsize=12)
    plt.show()

    plt.savefig("DuckAttention_imagenet_RAFDB_CM.pdf")
    plt.close()

    print(classification_report(all_target, all_output, target_names=class_names))
if __name__ == '__main__':
  plot_confusion_matrix(model, test_loader_ttau)
