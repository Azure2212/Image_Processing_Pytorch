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

from sgu24project.utils.datasets.rafdb_ds_with_mask_v2 import RafDataSet_Mask
from sgu24project.trainer.rafdb_multitask_trainer_v2 import RAFDB_Multitask_Trainer_v2

from sgu24project.models.multi_task_resunet import Resnet50UnetMultitask
import segmentation_models_pytorch as smp

#from sgu24project.models.resnet_cbam_v5 import resnet50_co_cbam
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default= "resnet50", type=str, help='model2Train')
parser.add_argument('--optimizer-chose', default= "RAdam", type=str, help='optimizer you chose')
parser.add_argument('--lr-scheduler', default= "ReduceLROnPlateau", type=str, help='learning rate scheduler you chose')
parser.add_argument('--lr-value', default= 1e-3, type=float, help='learning rate initial')
parser.add_argument('--use-wandb', default= 1, type=int, help='use wandb = 1, not use = 0')
parser.add_argument('--load-state-dir', default= '', type=str, help='weight2load')
parser.add_argument('--isDebug', default= 0, type=int, help='debug = 1')
parser.add_argument('--num-seg-classes', default= 6, type=int, help='num-classes')
parser.add_argument('--num-classes', default= 7, type=int, help='num-classes')
parser.add_argument('--use-pretrained', default= 1, type=int, help='use pre-trained = 1')
parser.add_argument('--current-epoch-num', default= 0, type=int, help='epoch start')
parser.add_argument('--epoch-num', default= 1, type=int, help='total epoch')
parser.add_argument('--name-run-wandb', default= 'Resnet50', type=str, help='name to save in wandb')
args, unknown = parser.parse_known_args()

print(torch.__version__)

config_path = "sgu24project/configs/config_rafdb.json"

configs = json.load(open(config_path))

configs["optimizer_chose"] = args.optimizer_chose
configs["lr_scheduler"] = args.lr_scheduler
configs["lr"] = args.lr_value
configs["isDebug"] = args.isDebug
configs["num_classes"] = args.num_classes
configs["num_seg_classes"]= args.num_seg_classes
configs["current_epoch_num"] = args.current_epoch_num
configs["name_run_wandb"] = args.name_run_wandb
if args.load_state_dir != '':
    configs["load_state_dir"] = args.load_state_dir
if args.epoch_num != 1:
    configs["max_epoch_num"] = args.epoch_num

CLASSES = ['eyes_mask', 'eyebrows_mask', 'nose_mask', 'mouth_mask', 'face_mask']

train_dataset = RafDataSet_Mask(data_type = 'train', configs = configs , classes=CLASSES)
valid_dataset = RafDataSet_Mask(data_type = 'test', configs = configs , classes=CLASSES)
test_dataset = RafDataSet_Mask(data_type = 'test', configs = configs , classes=CLASSES)

print(f'number of classes = {configs["num_seg_classes"]}')

OUT_CLASSES = len(CLASSES) + 1 
model = Resnet50UnetMultitask(in_channels=3, num_seg_classes=OUT_CLASSES, num_cls_classes=7)

print(f"the number of parameter: {sum(p.numel() for p in model.parameters())}")


use_wb = True if args.use_wandb == 1 else False


  
trainer = RAFDB_Segmentation_Trainer_v2(model = model, 
                                    train_loader = train_dataset, 
                                    val_loader = valid_dataset, 
                                    test_loader = test_dataset, 
                                    configs = configs, 
                                    wb = use_wb)
trainer.Train_model()