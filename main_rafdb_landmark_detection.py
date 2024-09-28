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

from sgu24project.utils.datasets.rafdb_ds_with_landmark_feature import image_with_landmark_RafDataSet
#from sgu24project.utils.datasets.smp_dataloader import get_training_augmentation, get_validation_augmentation, Dataset
from sgu24project.trainer.rafdb_LandmarkDetection_trainer_onlyLoss import RAFDB_Landmark_Detection_Trainer

from sgu24project.models.landmark_detection_in_unet import Landmark_Detection_in_InUnet

#from sgu24project.models.resnet_cbam_v5 import resnet50_co_cbam
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default= "resnet50", type=str, help='model2Train')
parser.add_argument('--optimizer-chose', default= "Adam", type=str, help='optimizer you chose')
parser.add_argument('--lr-scheduler', default= "ReduceLROnPlateau", type=str, help='learning rate scheduler you chose')
parser.add_argument('--lr-value', default= 1e-3, type=float, help='learning rate initial')
parser.add_argument('--use-wandb', default= 0, type=int, help='use wandb = 1, not use = 0')
parser.add_argument('--load-state-dir', default= '', type=str, help='weight2load')
parser.add_argument('--isDebug', default= 0, type=int, help='debug = 1')
parser.add_argument('--num-classes', default= 5, type=int, help='num-classes')
parser.add_argument('--use-pretrained', default= 1, type=int, help='use pre-trained = 1')
parser.add_argument('--current-epoch-num', default= 0, type=int, help='epoch start')
parser.add_argument('--epoch-num', default= 120, type=int, help='total epoch')
parser.add_argument('--name-run-wandb', default= 'Resnet50', type=str, help='name to save in wandb')
args, unknown = parser.parse_known_args()

print(torch.__version__)

config_path = "sgu24project/configs/config_rafdb.json"

configs = json.load(open(config_path))
#CLASSES = ['car', 'road', 'pavement', 'building', 'unlabelled']

configs["optimizer_chose"] = args.optimizer_chose
configs["lr_scheduler"] = args.lr_scheduler
configs["lr"] = args.lr_value
configs["isDebug"] = args.isDebug
configs["current_epoch_num"] = args.current_epoch_num
configs["name_run_wandb"] = args.name_run_wandb
if args.load_state_dir != '':
    configs["load_state_dir"] = args.load_state_dir
if args.epoch_num != 1:
    print(f'total epoch = {args.epoch_num}')
    configs["max_epoch_num"] = args.epoch_num


train_dataset = image_with_landmark_RafDataSet(data_type = 'train', configs = configs, device ='cuda')
valid_dataset = image_with_landmark_RafDataSet(data_type = 'test', configs = configs,  device ='cuda')
test_dataset = image_with_landmark_RafDataSet(data_type = 'test', configs = configs, device ='cuda')

model = Landmark_Detection_in_InUnet(in_channels=3)


print(f"the number of parameter: {sum(p.numel() for p in model.parameters())}")


use_wb = True if args.use_wandb == 1 else False


  
trainer = RAFDB_Landmark_Detection_Trainer(model = model, 
                                    train_loader = train_dataset, 
                                    val_loader = valid_dataset, 
                                    test_loader = test_dataset, 
                                    configs = configs, 
                                    wb = use_wb)
trainer.Train_model()
