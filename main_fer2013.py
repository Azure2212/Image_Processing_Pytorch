import os
import json
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import imgaug
import torch
import torch.multiprocessing as mp
import numpy as np
import torch.nn as nn

seed = 113
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from utils.datasets.fer2013_ds_v2 import FERDataset

from trainer.fer2013_trainer import FER2013_Trainer

from pytorchcv.model_provider import get_model as ptcv_get_model

print(torch.__version__)

config_path = "sgu24project/configs/config_fer2013.json"

configs = json.load(open(config_path))

train_loader = FERDataset( "train", configs)
val_loader = FERDataset("test", configs, ttau = False, len_tta = 48) 
test_loader_ttau = FERDataset("test", configs, ttau = True, len_tta = 10) 
test_loader = FERDataset("test", configs, ttau = False, len_tta = 48)

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default= "resnet50", type=str, help='model2Train')
parser.add_argument('--optimizer-chose', default= "RAdam", type=str, help='optimizer you chose')
parser.add_argument('--lr-scheduler', default= "ReduceLROnPlateau", type=str, help='learning rate scheduler you chose')
parser.add_argument('--lr-value', default= 1e-3, type=float, help='learning rate initial')
parser.add_argument('--use-wandb', default= 1, type=int, help='use wandb = 1, not use = 0')
parser.add_argument('--load-state-dir', default= '', type=str, help='weight2load')
parser.add_argument('--isDebug', default= 0, type=int, help='debug = 1')
parser.add_argument('--use-pretrained', default= 1, type=int, help='use pre-trained = 1')
parser.add_argument('--current-epoch-num', default= 0, type=int, help='epoch start')
parser.add_argument('--name-run-wandb', default= 'Resnet50', type=str, help='name to save in wandb')
args, unknown = parser.parse_known_args()

print(torch.__version__)

config_path = "sgu24project/configs/config_rafdb.json"

configs = json.load(open(config_path))

configs["optimizer_chose"] = args.optimizer_chose
configs["lr_scheduler"] = args.lr_scheduler
configs["lr"] = args.lr_value
configs["isDebug"] = args.isDebug
configs["current_epoch_num"] = args.current_epoch_num
configs["name_run_wandb"] = args.name_run_wandb
if args.load_state_dir != '':
    configs["load_state_dir"] = args.load_state_dir

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


#trainer = FER2013_Trainer(model, train_loader, val_loader, test_loader, test_loader_ttau, configs , wb = False)
from trainer.rafdb_trainer import RAFDB_Trainer
trainer = RAFDB_Trainer(model, train_loader, test_loader, test_loader, test_loader_ttau, configs , wb = False)
# trainer.acc_on_test()
# trainer.acc_on_test_ttau()

trainer.Train_model()
