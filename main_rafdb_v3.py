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

seed = 113
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#from utils.datasets.fer2013_ds import FERDataset
from utils.datasets.rafdb_ds import RafDataSet
# from models.vgg16_cbam import  VGG19_CBAM
#from models.resnet_cbam import ResidualNet , cbam_resnet50
#from models.vggnet import vgg16_bn, vgg19, vgg19_bn, vgg16
#from models.vggnet_cbam import vgg16_cbam, vgg19_cbam
#from models.DDAnet_vgg import *
#from models.test_cbam import TestModel
#from models.resmasking import *
#from models.BamNetwork import *
#from models.New_model import *
#from models.DDAnet import *

#from utils.visualize.show_img import show_image_dataset
#from trainer.fer2013_trainer import FER2013_Trainer
from trainer.rafdb_trainer import RAFDB_Trainer
#from trainer.rafdb_trainer_fixCUDA import RAFDB_Trainer


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

train_loader = RafDataSet( "train", configs)
# val_loader = RafDataSet("val", configs)
test_loader_ttau = RafDataSet("test", configs, ttau = True, len_tta = 10) 
test_loader = RafDataSet("test", configs, ttau = False, len_tta = 48) 

# show_image_dataset(train_ds)

# model = resnet50_cbam()
# if torch.cuda.is_available():
#     model.cuda()

# n_inputs = model.classifier[6].in_features  
# model.classifier[6] = nn.Sequential(
#             nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
#             nn.Linear(256, 7))

# import torchvision
# model = vgg16_bn()
#model = DuckResnet50(use_cbam = True)
#model = resnet50()

model = None
if args.model_name == 'resnet50_cbam_duck_pytorchcv':
    print('resnet50_cbam_duck_pytorchcv_tuan_code !')
    model = cbam_resnet50_duck()
    model.output = nn.Linear(2048, 7)
elif args.model_name == 'resnet_cbam':
    print('resnet_cbam_pytorchcv_tuan_code was chose !')
    model = cbam_resnet50()
    model.output = nn.Linear(2048, 7)
elif args.model_name == 'resnet50_cbam_mix':
    model = resnet50_cbam_mix()
elif args.model_name == 'resnet50_vggface2':
    print('resnet50 with pre-train on vggface2 was chose !')
    model = resnet50_v2()
else:
    print('because of missing model chosen, resnet in pytorch library activated !')
    model = resnet50(pretrained = True if args.use_pretrained == 1 else False)
print(f"the number of parameter: {sum(p.numel() for p in model.parameters())}")
#model = cbam_resnet50(in_channels=3, num_classes= 7 )
#model = resnet50_cbam_tuan()
# model = vgg19()
# model = vgg19_bn()
# model1 = vgg16_bn(pretrained = True, batch_norm = True)
# model = vgg19_cbam(num_classes = 7)
# model = vgg16_bn_cbam_pre(num_classes=7)
# pretrained_model = torchvision.models.vgg16_bn(pretrained=True)
# model = TestModel(pretrained_model)
# model = MultiFCVGGnetCBam(model1)
# model = resmasking_dropout1()
# model = resnetvdsr_dropout1()

# model = resnet34(num_classes = 7)
use_wb = True if args.use_wandb == 1 else False
trainer = RAFDB_Trainer(model, train_loader, test_loader, test_loader, test_loader_ttau, configs , wb = use_wb)

#760c20e6aae9012f4e41e4feffa73c63d4a10fc3

# trainer.acc_on_test()
# trainer.acc_on_test_ttau()
# 
trainer.Train_model()