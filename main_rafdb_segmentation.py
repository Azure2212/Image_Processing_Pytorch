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
from sgu24project.trainer.rafdb_segmentation_trainer import RAFDB_Segmentation_Trainer
from sgu24project.trainer.rafdb_multitask_trainer import RAFDB_Multitask_Trainer

from sgu24project.models.Unet import UNET
from sgu24project.models.multi_task_resunet import Resnet50UnetMultitask

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
parser.add_argument('--num-classes', default= 6, type=int, help='num-classes')
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
configs["current_epoch_num"] = args.current_epoch_num
configs["name_run_wandb"] = args.name_run_wandb
if args.load_state_dir != '':
    configs["load_state_dir"] = args.load_state_dir
if args.epoch_num != 1:
    configs["max_epoch_num"] = args.epoch_num
train_loader = RafDataSet_Mask( "train", configs)
test_loader = RafDataSet_Mask("test", configs, ttau = False, len_tta = 48) 
print(f'number of classes = {args.num_classes}')
import segmentation_models_pytorch as smp

if args.model_name == 'UNET':
    print('unet Tuan code')
    model = UNET(in_channels=3, classes=args.num_classes)
elif args.model_name == 'Resnet50UnetMultitask':
    print('multi task Unet Resnet Tuan code')
    model = Resnet50UnetMultitask(num_seg_classes=6, num_cls_classes=7, activation=None)
elif args.model_name =='UNET_resnet50_imagenet':
    print('smp.unet resnet50 on imagenet in library segmentation_models_pytorch')
    model = smp.Unet(
        encoder_name="resnet50",        # Choose an encoder (backbone)
        encoder_weights="imagenet",      # Use pretrained weights for the encoder
        classes=args.num_classes,             # Number of output classes
        activation=None                  # Choose activation function
    )
else:
# Create a U-Net model with a pretrained encoder
    print('unet in library segmentation_models_pytorch')
    model = smp.Unet(
        encoder_name="resnet50",        # Choose an encoder (backbone)
        encoder_weights="imagenet",      # Use pretrained weights for the encoder
        classes=args.num_classes,             # Number of output classes
        activation=None                  # Choose activation function
    )
print(f"the number of parameter: {sum(p.numel() for p in model.parameters())}")


use_wb = True if args.use_wandb == 1 else False


if args.model_name == 'Resnet50UnetMultitask':    
    trainer = RAFDB_Multitask_Trainer(model = model, 
                                    train_loader = train_loader, 
                                    val_loader = test_loader, 
                                    test_loader = test_loader, 
                                    configs = configs, 
                                    wb = use_wb)
else:
    trainer = RAFDB_Segmentation_Trainer(model = model, 
                                    train_loader = train_loader, 
                                    val_loader = test_loader, 
                                    test_loader = test_loader, 
                                    configs = configs, 
                                    wb = use_wb)

trainer.Train_model()
