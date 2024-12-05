import os
import json
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import imgaug
import torch
import torch.multiprocessing as mp
import numpy as np
import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

seed = 113
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from utils.metrics.metrics import accuracy, make_batch
from utils.datasets.fer2013_ds import FERDataset
# from models.vgg16_cbam import  VGG19_CBAM
from models.resnet_cbam import ResidualNet , cbam_resnet50
from models.vggnet import vgg16_bn, vgg19, vgg19_bn, vgg16
from models.resnet import resnet50
from models.vggnet_cbam import vgg16_cbam, vgg19_cbam
from models.vggnet_cbam_pre import vgg16_cbam_pre, vgg16_bn_cbam_pre, vgg19_bn_cbam_pre, MultiFCVGGnetCBam
from models.test_cbam import TestModel
from models.resmasking import *


from utils.visualize.show_img import show_image_dataset
from trainer.fer2013_trainer import FER2013_Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default= "resnet50", type=str, help='model2Train')
parser.add_argument('--rs-dir', default= "ResnetDuck_Cbam_cuaTuan", type=str, help='rs dir in kaggle')
parser.add_argument('--use-pretrained', default= 1, type=int, help='use pre-trained = 1')
parser.add_argument('--lr-scheduler', default= "ReduceLROnPlateau", type=str, help='learning rate scheduler you chose')

args, unknown = parser.parse_known_args()
path_current = os.path.abspath(globals().get("__file__","."))
script_dir  = os.path.dirname(path_current)
root_path = os.path.abspath(f"{script_dir}/../")



config_path = "configs/config_fer2013.json"
configs = json.load(open(config_path))
test_loader_ttau = FERDataset("test", configs, ttau = True, len_tta = 64) 

if args.model_name == 'resnet50_pytorchcv':
    print('resnet50_pytorchcv go !')
    if args.use_pretrained == 1:
        print('Pre train activated')
    model = ptcv_get_model("resnet50", pretrained=True if args.use_pretrained == 1 else False)
    model.output = nn.Linear(2048, 7)
elif args.model_name == 'resnet50_Cbam_pytorchcv':
    print('resnet50_Cbam_pytorchcv go !')
    if args.use_pretrained == 1:
        print('Pre train activated')
    model = ptcv_get_model("cbam_resnet50", pretrained=True if args.use_pretrained == 1 else False)
    model.output = nn.Linear(2048, 7)
    
state = torch.load("best_weight")
      
model.load_state_dict(state["net"])

def plot_confusion_matrix(model, testloader,title = "My model"):
    model.cuda()
    model.eval()

    correct = 0
    total = 0
    all_target = []
    all_output = []

    # test_set = fer2013("test", configs, tta=True, tta_size=8)
    # test_set = fer2013('test', configs, tta=False, tta_size=0)

    with torch.no_grad():
        for idx in tqdm.tqdm(range(len(testloader)), total=len(testloader), leave=False):
            images, labels = testloader[idx]

            images = make_batch(images)
            images = images.cuda(non_blocking=True)

            preds = model(images).cpu()
            preds = F.softmax(preds, 1)

            # preds.shape [tta_size, 7]
            preds = torch.sum(preds, 0)
            preds = torch.argmax(preds, 0)
            preds = preds.item()
            labels = labels.item()
            total += 1
            correct += preds == labels

            all_target.append(labels)
            all_output.append(preds)


    cf_matrix = confusion_matrix(all_target, all_output)
    cmn = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    
    # Create pandas dataframe
    dataframe = pd.DataFrame(cmn, index=class_names, columns=class_names)
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(dataframe, annot=True, cbar=True,cmap="Blues",fmt=".2f")
    
    plt.title(title), plt.tight_layout()
    
    plt.ylabel("True Class", fontsize=12), 
    plt.xlabel("Predicted Class", fontsize=12)
    plt.show()

    plt.savefig("checkpoints/best_fer.pdf")
    plt.close()
      
if __name__ == '__main__':
      plot_confusion_matrix(model, test_loader_ttau)
