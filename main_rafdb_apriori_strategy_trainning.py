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

seed = 113
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from utils.datasets.fer2013_ds import FERDataset
from utils.datasets.rafdb_ds import RafDataSet
# from models.vgg16_cbam import  VGG19_CBAM
#from models.resnet_cbam import ResidualNet , cbam_resnet50

from utils.visualize.show_img import show_image_dataset
from trainer.fer2013_trainer import FER2013_Trainer
#from trainer.rafdb_trainer import RAFDB_Trainer
from trainer.apriori_strategy_trainning_on_rafdb_trainer import RAFDB_Trainer
#from trainer.rafdb_trainer_fixCUDA import RAFDB_Trainer

from sgu24project.models.resnet_cbam_pytorchcv.cbamresnet import cbam_resnet50


from sgu24project.models.resnet_cbam_pytorchcv.cbamresnet_duck import cbam_resnet50_duck

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

parser.add_argument('--evaluation-priority', default= 'val_acc', type=str, help='val_acc or val_loss or percent_average_diagnose')
args, unknown = parser.parse_known_args()

print(torch.__version__)

config_path = "sgu24project/configs/config_rafdb.json"

class Node:
    def __init__(self, name, depth):
        self.status = "TBA"
        self.depth = depth
        self.name = name
        self.best_val_acc = None
        self.best_val_loss = None
        self.confusion_matrix = None
        self.confusion_matrix_normalization = None
        self.dir_weight = None
        self.head = None
        self.tail = None

    def getPercentAverageDiagnose(self):
        return np.diag(self.confusion_matrix_normalization).sum()/7
    
    def UpdateStatistic(self, best_val_acc, best_val_loss, confusion_matrix, confusion_matrix_normalization, status):
        self.best_val_acc = best_val_acc
        self.best_val_loss = best_val_loss
        self.confusion_matrix = confusion_matrix
        self.confusion_matrix_normalization = confusion_matrix_normalization
        self.status = status
    def toPrint(self):
        print("\n--------------report----------")
        print(f"strategy name = {self.name} in depth ={self.depth}")
        print(f"best_val_acc = {self.best_val_acc}")
        print(f"best_val_loss = {self.best_val_loss}")
        print(f"normalization confusion matrix :\n{self.confusion_matrix_normalization}")
        print(f"percent average diagnose ={self.getPercentAverageDiagnose()}\n")
        print(f"status = {self.status}")
        print(f"parent_node = {self.head.name}")


#============================== SETUP model and training strategy decision tree =========================================
print('resnet50 is activated !')

#Background layer = 0 (have 1 node)
model = resnet50(pretrained = True if args.use_pretrained == 1 else False)   
status_enum = ["TBA","Eliminated","best"]

deep = {
        1: ['RAdam', 'Adam', 'AdamW', 'Adamax', 'Adagrad', 'RMSprop', 'SGD'],  #optimizer options
        2: ['ReduceLROnPlateau', 'MultiStepLR', 'ExponentialLR', 'PolynomialLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'None'], # lr schedule options
        3: ['use', 'None'],   # diversify images using_albumentation or not
       }

########Create Training Strategy Decision Tree
root = Node(name = "root", depth = 0)
root.status = status_enum[2]
root.tail = []

for optimizer in deep[1]:
    optimizer_node = Node(name = optimizer, depth = 1)
    optimizer_node.dir_weight = f"root_{optimizer}(optimizer)"
    optimizer_node.head = root
    optimizer_node.status = status_enum[0]
    optimizer_node.tail = []
    root.tail.append(optimizer_node)
    for lr_scheduler in deep[2]:
        lr_scheduler_node = Node(name = lr_scheduler, depth = 2)
        lr_scheduler_node.dir_weight = optimizer_node.dir_weight + f"_{lr_scheduler}(lr_scheduler)"
        lr_scheduler_node.status = status_enum[0]
        lr_scheduler_node.head = optimizer_node
        lr_scheduler_node.tail = []
        optimizer_node.tail.append(lr_scheduler_node)
        for use_albumentation in deep[3]:
            use_albumentation_node = Node(name = use_albumentation, depth = 3)
            use_albumentation_node.dir_weight = lr_scheduler_node.dir_weight + f"_{use_albumentation}(use_albumentation)"
            use_albumentation_node.status = status_enum[0]
            use_albumentation_node.head = lr_scheduler_node
            use_albumentation_node.tail = []
            lr_scheduler_node.tail.append(use_albumentation_node)

#============================== Running training strategy decision tree =========================================
#trainer = RAFDB_Trainer(model, train_loader, test_loader, test_loader, test_loader_ttau, configs , wb = True if args.use_wandb == 1 else False)

#trainer.Train_model()

configs = json.load(open(config_path))

configs["optimizer_chose"] = args.optimizer_chose
configs["lr_scheduler"] = args.lr_scheduler
configs["isDebug"] = args.isDebug

#============================== SETUP DATA =========================================
train_loader = RafDataSet( "train", configs)
# val_loader = RafDataSet("val", configs)
test_loader_ttau = RafDataSet("test", configs, ttau = True, len_tta = 10) 
test_loader = RafDataSet("test", configs, ttau = False, len_tta = 48) 

if args.isDebug == 1:
      print("Debug mode activated")
      configs['max_epoch_num'] = 3
    #if in debug mode, remainding 100 or less image for trainning, validation, testing model

      #========= set 140 image for Train ================
      n_train_debug = 140 if len(train_loader)> 140 else len(train_loader)
      train_loader.label = train_loader.label[: n_train_debug]
      train_loader.file_paths = train_loader.file_paths[: n_train_debug]

      #========= set 70 image for test ( = validation) ================
      n_test_debug = 70 if len(test_loader)> 70 else len(test_loader)
      test_loader.label = test_loader.label[: n_test_debug]
      test_loader.file_paths = test_loader.file_paths[: n_test_debug]
     
      #========= set 70 image for test_loader_ttau ================
      n_test_ttau_debug = 140 if len(test_loader_ttau)> 140 else len(test_loader_ttau)
      #self.test_loader_ttau.label = test_loader_ttau.label[: n_test_ttau_debug]
      #self.test_loader_ttau.file_paths = test_loader_ttau.file_paths[: n_test_ttau_debug]
      label_amount = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
      label_list = []
      img_list = []
      for a in tqdm.tqdm(range(len(test_loader_ttau) - 1, -1, -1), desc="Processing data"):
        label = test_loader_ttau.label[a]
        image = test_loader_ttau.file_paths[a]
        if label_amount[label] < n_test_ttau_debug / 7:
          img_list.append(image)
          label_list.append(label)
          label_amount[label] = label_amount[label] +1
        if len(label_list) == n_test_ttau_debug:
            break
      test_loader_ttau.file_paths = img_list
      test_loader_ttau.label = label_list
      #print(label_amount)

#====================Looking for the best optimizer(depth=1)====================
import tqdm
for index, optimizer in enumerate(tqdm.tqdm(root.tail, desc="In selecting optimizer process(deep = 1 in training strategy decision tree)")):
    print(f"============== In optimizer {index+1}/{len(root.tail)} ==============")
    configs["optimizer_chose"] = optimizer.name
    configs["lr_scheduler"] = None
    train_loader.use_albumentation = False
    use_wb = True if args.use_wandb == 1 else False
    trainer = RAFDB_Trainer(model, train_loader, test_loader, test_loader, test_loader_ttau, configs, dir_weight = optimizer.dir_weight , wb = use_wb)

    best_val_acc, best_val_loss, confusion_matrix, confusion_matrix_normalization = trainer.Train_model()
    optimizer.UpdateStatistic(best_val_acc, best_val_loss, confusion_matrix, confusion_matrix_normalization, "TBA")
    optimizer.toPrint()


#
for i in range(len(root.tail)):
    for j in range(i+1,len(root.tail)):
        if root.tail[i].best_val_acc > root.tail[j].best_val_acc:
            root.tail[i], root.tail[j] = root.tail[j], root.tail[i]

index_best_optimizer = 0
root.tail[index_best_optimizer].status = status_enum[2]


path_file_2_remove = []
for i in range(1,len(root.tail)):
    statement = False # False: current index not better index_best_lr
    if args.evaluation_priority == 'val_acc' and (root.tail[i].best_val_acc > root.tail[index_best_optimizer].best_val_acc):
        statement = True
    elif args.evaluation_priority == 'val_loss' and (root.tail[i].best_val_loss > root.tail[index_best_optimizer].best_val_loss):
        statement = True
    elif args.evaluation_priority == 'percent_average_diagnose' and (root.tail[i].getPercentAverageDiagnose() > root.tail[index_best_optimizer].getPercentAverageDiagnose()):
        statement = True

    if statement == False:
        root.tail[i].status = status_enum[1]
        path_file_2_remove.append('/kaggle/working/ResnetDuck_Cbam_cuaTuan_'+root.tail[i].dir_weight)
    else:
        root.tail[index_best_optimizer].status = status_enum[1]
        path_file_2_remove.append('/kaggle/working/ResnetDuck_Cbam_cuaTuan_'+root.tail[index_best_optimizer].dir_weight)
        index_best_optimizer = i
        root.tail[index_best_optimizer].status = status_enum[2]



print()
print(f"remove all weight{len(path_file_2_remove)} in list:{path_file_2_remove}")
for file_path in path_file_2_remove:
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} removed successfully.")
    else:
        print(f"File {file_path} does not exist.")

print("---------------------------------------------------------")
print()
print()
print("the best optimizer should be use in model is:")
for optimizer in root.tail:
    if optimizer.status == status_enum[2]:
        optimizer.toPrint()
print()
print()
print("---------------------------------------------------------")

#====================Looking for the best learning_rate scheduler(depth=2)====================
import tqdm
for index, lr_scheduler in enumerate(tqdm.tqdm(root.tail[index_best_optimizer].tail, desc="In selecting learning rate scheduler process(deep = 2 in training strategy decision tree)")):
    if lr_scheduler.name != 'None':
        print(f"============== In lr_scheduler {index+1}/{len(root.tail[index_best_optimizer].tail)} ==============")
        configs["optimizer_chose"] = root.tail[index_best_optimizer].name
        configs["lr_scheduler"] = lr_scheduler.name
        train_loader.use_albumentation = False
        use_wb = True if args.use_wandb == 1 else False
        trainer = RAFDB_Trainer(model, train_loader, test_loader, test_loader, test_loader_ttau, configs, dir_weight = lr_scheduler.dir_weight , wb = use_wb)

        best_val_acc, best_val_loss, confusion_matrix, confusion_matrix_normalization = trainer.Train_model()
        lr_scheduler.UpdateStatistic(best_val_acc, best_val_loss, confusion_matrix, confusion_matrix_normalization, "TBA")
        lr_scheduler.toPrint()


for i in range(len(root.tail[index_best_optimizer].tail)):
    for j in range(i+1,len(root.tail[index_best_optimizer].tail)):
        if root.tail[index_best_optimizer].tail[i].name == 'None' or root.tail[index_best_optimizer].tail[j].name == 'None': continue

        print(f"{root.tail[index_best_optimizer].tail[i].name} ----------------- {root.tail[index_best_optimizer].tail[j].name}")
        if root.tail[index_best_optimizer].tail[i].best_val_acc > root.tail[index_best_optimizer].tail[j].best_val_acc:
            root.tail[index_best_optimizer].tail[i], root.tail[index_best_optimizer].tail[j] = root.tail[index_best_optimizer].tail[j], root.tail[index_best_optimizer].tail[i]
print("------------------------------------------------------------------------------------")

index_best_lr = 0
root.tail[index_best_optimizer].tail[index_best_lr].status = status_enum[2]

path_file_2_remove = []
for i in range(1,len(root.tail[index_best_optimizer].tail)):
    if root.tail[index_best_optimizer].tail[i].name != 'None': 
        statement = False # False: current index not better index_best_lr
        if args.evaluation_priority == 'val_acc' and (root.tail[index_best_optimizer].tail[i].best_val_acc > root.tail[index_best_optimizer].tail[index_best_lr].best_val_acc):
            statement = True
        elif args.evaluation_priority == 'val_loss' and (root.tail[index_best_optimizer].tail[i].best_val_loss > root.tail[index_best_optimizer].tail[index_best_lr].best_val_loss):
            statement = True
        elif args.evaluation_priority == 'percent_average_diagnose' and (root.tail[index_best_optimizer].tail[i].getPercentAverageDiagnose() > root.tail[index_best_optimizer].tail[index_best_lr].getPercentAverageDiagnose()):
            statement = True

        if statement == False:
            root.tail[index_best_optimizer].tail[i].status = status_enum[1]
            path_file_2_remove.append('/kaggle/working/ResnetDuck_Cbam_cuaTuan_'+root.tail[index_best_optimizer].dir_weight+root.tail[index_best_optimizer].tail[i].dir_weight)
        else:
            root.tail[index_best_optimizer].tail[index_best_lr].status = status_enum[1]
            path_file_2_remove.append('/kaggle/working/ResnetDuck_Cbam_cuaTuan_'+root.tail[index_best_optimizer].dir_weight+root.tail[index_best_optimizer].tail[index_best_lr].dir_weight)
            index_best_lr = i
            root.tail[index_best_optimizer].tail[index_best_lr].status = status_enum[2]

    else:
        root.tail[index_best_optimizer].tail[index_best_lr].toPrint()
        statement = False # False: current index not better index_best_lr
        if args.evaluation_priority == 'val_acc' and (root.tail[index_best_optimizer].best_val_acc > root.tail[index_best_optimizer].tail[index_best_lr].best_val_acc):
            statement = True
        elif args.evaluation_priority == 'val_loss' and (root.tail[index_best_optimizer].best_val_loss > root.tail[index_best_optimizer].tail[index_best_lr].best_val_loss):
            statement = True
        elif args.evaluation_priority == 'percent_average_diagnose' and (root.tail[index_best_optimizer].getPercentAverageDiagnose() > root.tail[index_best_optimizer].tail[index_best_lr].getPercentAverageDiagnose()):
            statement = True

        if statement == False:
            root.tail[index_best_optimizer].status = status_enum[1]
            path_file_2_remove.append('/kaggle/working/ResnetDuck_Cbam_cuaTuan_'+root.tail[index_best_optimizer].dir_weight)
        else:
            root.tail[index_best_optimizer].tail[index_best_lr].status = status_enum[1]
            path_file_2_remove.append('/kaggle/working/ResnetDuck_Cbam_cuaTuan_'+root.tail[index_best_optimizer].dir_weight+root.tail[index_best_optimizer].tail[index_best_lr].dir_weight)
            index_best_lr = len(deep[2])-1
            root.tail[index_best_optimizer].status = status_enum[2]


print()
print(f"remove all weight{len(path_file_2_remove)} in list:{path_file_2_remove}")
for file_path in path_file_2_remove:
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} removed successfully.")
    else:
        print(f"File {file_path} does not exist.")

print("---------------------------------------------------------")
print()
print()
print("the best lr_scheduler should be use in model is:")
is_lr_scheduler_useful = False
for lr_scheduler in root.tail[index_best_optimizer].tail:
    if lr_scheduler.status == status_enum[2]:
        lr_scheduler.toPrint()
        is_lr_scheduler_useful = True
if is_lr_scheduler_useful == False:
    print("Lr rate schedulers arn't useful !\n the best strategy now is:\n")
    root.tail[index_best_optimizer].toPrint()
print()
print()
print("---------------------------------------------------------")


#report

# print(f"----------------------root(dep=0)----------------------")
# print(f"root name:{root.name} , depth ={root.depth}, status ={root.status}, childs-Node = {len(root.tail)}")
# print(f"\n----------------------optimizer(dep=1)----------------------")
# for i in root.tail:
#     i.toPrint()
# print(f"\n----------------------lr_scheduler(dep=2)----------------------")
# for i in root.tail[index_best_optimizer].tail:
#     i.toPrint()