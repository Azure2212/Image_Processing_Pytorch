import os
import json
import random
import warnings
import segmentation_models_pytorch as smp
warnings.simplefilter(action="ignore", category=FutureWarning)

import imgaug
import torch
import torch.multiprocessing as mp
import numpy as np
import tqdm
import torch.nn as nn

seed = 113
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from sgu24project.utils.datasets.rafdb_ds_with_mask import RafDataSet_Mask
from sgu24project.utils.datasets.smp_dataloader import get_training_augmentation, get_validation_augmentation, Dataset
from sgu24project.trainer.rafdb_segmentation_trainer_v2 import RAFDB_Segmentation_Trainer_v2
from sgu24project.trainer.rafdb_multitask_trainer import RAFDB_Multitask_Trainer

from sgu24project.models.Unet import UNET
from sgu24project.models.multi_task_resunet import Resnet50UnetMultitask

#from sgu24project.models.resnet_cbam_v5 import resnet50_co_cbam
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default= "resnet50", type=str, help='model2Train')
parser.add_argument('--optimizer-chose', default= "Adam", type=str, help='optimizer you chose')
parser.add_argument('--lr-scheduler', default= "CosineAnnealingLR", type=str, help='learning rate scheduler you chose')
parser.add_argument('--lr-value', default= 1e-3, type=float, help='learning rate initial')
parser.add_argument('--use-wandb', default= 0, type=int, help='use wandb = 1, not use = 0')
parser.add_argument('--load-state-dir', default= '', type=str, help='weight2load')
parser.add_argument('--isDebug', default= 0, type=int, help='debug = 1')
parser.add_argument('--num-classes', default= 5, type=int, help='num-classes')
parser.add_argument('--use-pretrained', default= 1, type=int, help='use pre-trained = 1')
parser.add_argument('--current-epoch-num', default= 0, type=int, help='epoch start')
parser.add_argument('--epoch-num', default= 20, type=int, help='total epoch')
parser.add_argument('--name-run-wandb', default= 'Resnet50', type=str, help='name to save in wandb')
args, unknown = parser.parse_known_args()

print(torch.__version__)

config_path = "sgu24project/configs/config_rafdb.json"

configs = json.load(open(config_path))
CLASSES = ['car', 'road', 'pavement', 'building', 'unlabelled']

configs["optimizer_chose"] = args.optimizer_chose
configs["lr_scheduler"] = args.lr_scheduler
configs["lr"] = args.lr_value
configs["isDebug"] = args.isDebug
configs["num_seg_classes"] = len(CLASSES)+1
configs["current_epoch_num"] = args.current_epoch_num
configs["name_run_wandb"] = args.name_run_wandb
if args.load_state_dir != '':
    configs["load_state_dir"] = args.load_state_dir
if args.epoch_num != 1:
    configs["max_epoch_num"] = args.epoch_num


CLASSES = ['car', 'road', 'pavement', 'building', 'unlabelled']
DATA_DIR = '/kaggle/working/data/CamVid'

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')
train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    augmentation=get_validation_augmentation(), 
    classes=CLASSES,
)

test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(), 
    classes=CLASSES,
)

print(f'number of classes = {args.num_classes}')
import segmentation_models_pytorch as smp



class Resnet50UnetMultitask_v2(smp.Unet):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, num_seg_classes=2, num_cls_classes=7, activation=None):
        # Initialize the parent class (smp.Unet)
        super().__init__(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, activation=None, classes=num_seg_classes)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_cls_classes)
    def forward(self, x):
        #Phase 1: Encoder
        encoder_features = self.encoder(x) 
        
        #make classification Task
        stage4_resnet50 = encoder_features[5]
        avg = self.avgpool(stage4_resnet50)
        flat = torch.flatten(avg, 1)
        cls_output = self.fc(flat)
        
        # Phase 2: Decoder
        seg_output = self.decoder(*encoder_features) 
        seg_output = self.segmentation_head(seg_output) 
        
        #seg_output = torch.randn(42, 6, 224, 224)
        return seg_output

OUT_CLASSES = len(CLASSES)
model = Resnet50UnetMultitask_v2(in_channels=3, num_seg_classes=OUT_CLASSES, num_cls_classes=7)


print(f"the number of parameter: {sum(p.numel() for p in model.parameters())}")


use_wb = True if args.use_wandb == 1 else False


  
trainer = RAFDB_Segmentation_Trainer_v2(model = model, 
                                    train_loader = train_dataset, 
                                    val_loader = valid_dataset, 
                                    test_loader = test_dataset, 
                                    configs = configs, 
                                    wb = use_wb)
trainer.Train_model()
