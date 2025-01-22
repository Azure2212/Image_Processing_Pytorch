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
import torch.nn as nn

import pandas as pd

seed = 113
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from utils.datasets.rafdb_ds import RafDataSet

from models.resnet_v2 import resnet50, resnet50_vggface2, resnet50_vggface2_ft, CbamBlock

from trainer.rafdb_trainer import RAFDB_Trainer

from sgu24project.models.resnet_cbam_pytorchcv.cbamresnet import cbam_resnet50


from sgu24project.models.resnet_cbam_pytorchcv.cbamresnet_duck import cbam_resnet50_duck

#from sgu24project.models.resnet_cbam_v5 import resnet50_co_cbam
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default= "resnet50", type=str, help='model2Train')
parser.add_argument('--optimizer-chose', default= "RAdam", type=str, help='optimizer you chose')
parser.add_argument('--lr-scheduler', default= "ReduceLROnPlateau", type=str, help='learning rate scheduler you chose')
parser.add_argument('--lr-value', default= 1e-3, type=float, help='learning rate initial')
parser.add_argument('--use-wandb', default= 1, type=int, help='use wandb = 1, not use = 0')
parser.add_argument('--load-weight-path', default= '', type=str, help='weight2load')
parser.add_argument('--apply-dropout', default= 0, type=int, help='apply or not')
parser.add_argument('--isDebug', default= 0, type=int, help='debug = 1')
parser.add_argument('--use-pretrained', default= 1, type=int, help='use pre-trained = 1')
parser.add_argument('--link-pretrained', default= '', type=str, help='link pretrain')
parser.add_argument('--use-cbam', default= 0, type=int, help='use cbam= 1')
parser.add_argument('--use-duck', default= 0, type=int, help='use duck = 1')
parser.add_argument('--max-epoch-num', default= 120, type=int, help='max epoch to train')
parser.add_argument('--current-epoch-num', default= 0, type=int, help='epoch start')
parser.add_argument('--name-run-wandb', default= 'Resnet50', type=str, help='name to save in wandb')
parser.add_argument('--freeze-cbam', default= 0, type=int, help='epoch start')
args, unknown = parser.parse_known_args()

print(torch.__version__)

config_path = "sgu24project/configs/config_rafdb.json"

configs = json.load(open(config_path))

configs["optimizer_chose"] = args.optimizer_chose
configs["lr_scheduler"] = 'None'
configs["lr"] = args.lr_value
configs["isDebug"] = args.isDebug
configs["current_epoch_num"] = args.current_epoch_num
configs["name_run_wandb"] = args.name_run_wandb
configs["max_epoch_num"] = args.max_epoch_num
if args.load_weight_path != '':
    configs["load_weight_path"] = args.load_weight_path

train_loader = RafDataSet( "train", configs)
test_loader_ttau = RafDataSet("test", configs, ttau = True, len_tta = 10) 
test_loader = RafDataSet("test", configs, ttau = False, len_tta = 48) 

model = None
if args.model_name == 'resnet50_cbam_duck_pytorchcv':
    print('resnet50_cbam_duck_pytorchcv_tuan_code !')
    model = cbam_resnet50_duck()
    model.output = nn.Linear(2048, 7)
elif args.model_name == 'resnet50_vggface2':
    print('resnet50 with pre-train on vggface2(trained from cratch) was chose !')
    model = resnet50_vggface2()
elif args.model_name == 'resnet50_vggface2_ft':
    print('resnet50 with pre-train on vggface2(trained on MS1M, and then fine-tuned on VGGFace2) was chose !')
    model = resnet50_vggface2_ft(pretrained = True if args.use_pretrained == 1 else False, 
                                 use_cbam = True if args.use_cbam == 1 else False, 
                                 use_duck = True if args.use_duck == 1 else False, 
                                 load_weight_path = args.load_weight_path,
                                 apply_dropout = args.apply_dropout)
    # for name, layer in model.named_children():
    #    print(f"{name}: {layer}")
    #    print('________________________________')
    
    if args.freeze_cbam == 1:
        print("go freeze")
        layers = [3, 4, 6, 3]  
        layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        
        for name, layer in model.named_children():
            if isinstance(layer, torch.nn.Module):  # Only consider actual layers
                for param in layer.parameters():
                    param.requires_grad = False

        # Unfreeze CBAM blocks as specified by the layers
        for i, layer_name in enumerate(layer_names):
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                #print(f'vo layer{i}')
                for idx in range(layers[i]):
                    # Unfreeze parameters for the CBAM block in the current layer
                    for param in layer[idx].CbamBlock.parameters():
                        param.requires_grad = True
                    # Print the name of each layer in the CBAM block
                    # for name, submodule in layer[idx].CbamBlock.named_children():
                    #     print(f"Layer Name: {name} in {layer_name}[{idx}] CbamBlock")
                    # print("_____________________")
            
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")

elif args.model_name == 'resnet50_imagenet':
    print('resnet50 with pre-train on imagenet was chose !')
    model = resnet50()
elif args.model_name == 'resnet_in_unet':
    from sgu24project.models.resnet_in_unet import Resnet50InUnet
    print('resnet50 in unet')
    model = Resnet50InUnet()
elif args.model_name == 'Resnet50_in_smp':
    from sgu24project.models.segmentation_models_pytorch.model import Resnet50_in_smp
    model = Resnet50_in_smp(in_channels=3, num_seg_classes=6, num_cls_classes=7)
    print('Resnet50_in_smp activated')
else:
    print('because of missing model chosen, resnet in pytorch library activated !')
    model = resnet50(pretrained = True if args.use_pretrained == 1 else False)
print(f"the number of parameter: {sum(p.numel() for p in model.parameters())}")


use_wb = True if args.use_wandb == 1 else False
trainer = RAFDB_Trainer(model, train_loader, test_loader, test_loader, test_loader_ttau, configs , wb = use_wb, output_csv_path = '/kaggle/working/out.csv')
model_afd_5tun, best_val_acc_previous = trainer.Train_model()

print(best_val_acc_previous)
print("----------------------------------------------")
print("Phase2 train")
configs["lr_scheduler"] = args.lr_scheduler
configs["lr"] = args.lr_value/10
for name, layer in model_afd_5tun.named_children():
    if isinstance(layer, torch.nn.Module):  # Only consider actual layers
        for param in layer.parameters():
            param.requires_grad = True
            
print(f'lr_scheduler ={configs["lr_scheduler"]}')

for name, param in model_afd_5tun.named_parameters():
    print(f"{name}: {param.requires_grad}")
# for name, layer in model_afd_5tun.named_children():
#     print(f"{name}: {layer}")
print("Training!")
trainer2 = RAFDB_Trainer(model_afd_5tun, train_loader, test_loader, test_loader, test_loader_ttau, configs , wb = use_wb, output_csv_path = '/kaggle/working/out2.csv', initial_best_val_acc = best_val_acc_previous)
model_afd_2rdTrain, best_val_acc_2rdTrain = trainer2.Train_model()