import numpy as np
import datetime
import os
import traceback
import shutil

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# from utils.radam import RAdam

import tqdm
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiplicativeLR, StepLR, MultiStepLR, ConstantLR, LinearLR, PolynomialLR, CosineAnnealingLR, ChainedScheduler, ExponentialLR, SequentialLR, ReduceLROnPlateau, CyclicLR, CosineAnnealingWarmRestarts

from sgu24project.utils.metrics.metrics import accuracy, make_batch

import warnings
warnings.filterwarnings('ignore')
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Trainer(object):
    """base class for trainers"""

    def __init__(self):
        pass


class RAFDB_Multitask_Trainer(Trainer):
  def __init__(self, model, train_loader, val_loader, test_loader, configs, wb = True):

    self.train_loader = train_loader
    self.val_loader = val_loader
    self.test_loader = test_loader

    self.configs = configs

    self.batch_size = configs["batch_size"]
    self.debug = configs["debug"]
    # self.epochs = configs["epochs"]
    self.learning_rate = configs["lr"]
    self.min_lr = configs["min_lr"]
    self.num_workers = configs["num_workers"]
    self.momentum = configs["momentum"]
    self.weight_decay = configs["weight_decay"]
    self.device = torch.device(configs["device"])
    self.max_plateau_count = configs["max_plateau_count"]
    self.max_epoch_num = configs["max_epoch_num"]
    self.distributed = configs["distributed"]
    self.optimizer_chose = configs["optimizer_chose"]
    self.lr_scheduler_chose = configs["lr_scheduler"]
    self.isDebug = configs["isDebug"]
    self.name_run_wandb = configs["name_run_wandb"]
    self.wb = wb
    self.num_classes = configs["num_classes"]
    print(f'self.num_classes ={self.num_classes}')
    #self.model = model.to(self.device)'cpu'
    self.model = model.to(self.device)

# Move the model to the device
    '''try:
      model = model.to(self.device)
    except Exception as e:
      print("Error:", e)'''
    #index 0 is segmentation, index 1 is classification
    self.train_loss_list = [[],[]]
    self.train_acc_list = [[],[]]
    self.train_dice_list = []
    self.train_iou_list = []
    self.train_total_loss_list = []

    self.val_loss_list = [[],[]]
    self.val_acc_list = [[],[]]
    self.val_dice_list = []
    self.val_iou_list = []
    self.val_total_loss_list = []

    self.best_train_acc = {'segmentation':0.0, 'classification':0.0}
    self.best_train_loss = {'segmentation':0.0, 'classification':0.0}
    self.best_train_dice = []
    self.best_train_iou = []

    self.best_val_acc = {'segmentation':0.0, 'classification':0.0}
    self.best_val_loss = {'segmentation':0.0, 'classification':0.0}
    self.best_val_dice = {'segmentation':0.0, 'classification':0.0}
    self.best_val_iou = {'segmentation':0.0, 'classification':0.0}

    self.test_acc = {'segmentation':0.0, 'classification':0.0}
    self.test_acc_ttau = {'segmentation':0.0, 'classification':0.0}
    self.test_dice =0.0
    self.test_iou = 0.0
    self.plateau_count = 0
    #self.current_epoch_num = 0
    self.current_epoch_num = configs["current_epoch_num"]
    self.load_state_dir = configs["load_state_dir"]
    # Set information for training
    self.start_time = datetime.datetime.now()

    self.checkpoint_dir = "/kaggle/working/"

    '''self.checkpoint_path = os.path.join(self.checkpoint_dir, "{}_{}_{}".format
                                        (self.configs["project_name"], self.configs["model"], self.start_time.strftime("%Y%b%d_%H.%M"),))'''

    self.checkpoint_path = os.path.join(self.checkpoint_dir,"ResnetDuck_Cbam_cuaTuan")
    # load dataset

    if self.distributed == 1:
            torch.distributed.init_process_group(backend="nccl")
            self.model = nn.parallel.DistributedDataParallel(self.model)
            print("Let's use", torch.cuda.device_count(), "GPUs!")

            self.train_ds = DataLoader(
                self.train_loader,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=True,
                worker_init_fn=lambda x: np.random.seed(x),
            )
            self.val_ds = DataLoader(
                self.val_loader,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=False,
                worker_init_fn=lambda x: np.random.seed(x),
            )

            self.test_ds = DataLoader(
                self.test_loader,
                batch_size=1,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=False,
                worker_init_fn=lambda x: np.random.seed(x),
            )
    else:

      self.train_ds = DataLoader(self.train_loader,batch_size=self.batch_size,num_workers=self.num_workers,
                        pin_memory=True, shuffle=True)
      self.val_ds = DataLoader(self.val_loader, batch_size = self.batch_size, num_workers=self.num_workers,
                      pin_memory=True, shuffle=False)
      self.test_ds = DataLoader(self.test_loader, batch_size= 1,num_workers=self.num_workers,
                      pin_memory=True, shuffle=False)
    
    
    self.criterion = nn.CrossEntropyLoss().to(self.device)
    if self.optimizer_chose == "RAdam":
      print("The selected optimizer is RAdam")
      self.optimizer = torch.optim.RAdam(
        params=self.model.parameters(),
        lr=self.learning_rate,
        weight_decay=self.weight_decay,
#       amsgrad = True,
    )
    elif self.optimizer_chose == "SGD":
      print("The selected optimizer is SGD")
      self.optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
    elif self.optimizer_chose == "RMSprop":
      print("The selected optimizer is RMSprop")
      self.optimizer = torch.optim.RMSprop(
        params=model.parameters(),
        lr=self.learning_rate,
        weight_decay=self.weight_decay,
        momentum=0.9,
        alpha=0.99,
        eps=1e-8)
    elif self.optimizer_chose == "Adam":
      print("The selected optimizer is Adam")
      self.optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=self.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=self.weight_decay)
    elif self.optimizer_chose == "AdamW":
      print("The selected optimizer is AdamW")
      self.optimizer = torch.optim.AdamW(
          params=self.model.parameters(),
          lr=self.learning_rate,
          betas=(0.9, 0.999),
          eps=1e-8,
          weight_decay=self.weight_decay)
    elif self.optimizer_chose == "Adamax":
      print("The selected optimizer is Adamax")
      self.optimizer = torch.optim.Adamax(
          params=self.model.parameters(),
          lr=self.learning_rate,
          betas=(0.9, 0.999),
          eps=1e-8,
          weight_decay=self.weight_decay)
    elif self.optimizer_chose == "Adagrad":
      print("The selected optimizer is Adagrad")
      self.optimizer = torch.optim.Adagrad(
          params=self.model.parameters(),
          lr=self.learning_rate,
          lr_decay=0.001,  
          weight_decay=self.weight_decay,  
          initial_accumulator_value=0.1,  
          eps=1e-8
      )
    else:  #default ="RAdam"
      print("The selected optimizer is RAdam")
      self.optimizer = torch.optim.RAdam(
        params=self.model.parameters(),
        lr=self.learning_rate,
        weight_decay=self.weight_decay,
#       amsgrad = True,
    )
      
      
    if self.lr_scheduler_chose == "ReduceLROnPlateau":
      self.scheduler = ReduceLROnPlateau(
        self.optimizer,
        patience=self.configs["plateau_patience"],
        min_lr=self.min_lr,
        # factor = torch.exp(torch.Tensor([-0.1])),
        verbose=True,
        factor = 0.1,
      )
      print("The selected learning_rate scheduler strategy is ReduceLROnPlateau")
    elif self.lr_scheduler_chose == "MultiStepLR":
      milestones = [x for x in range(5, 120, 5)]
      self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=0.5,  verbose=True)
      print("The selected learning_rate scheduler strategy is MultiStepLR")

    elif self.lr_scheduler_chose == "ExponentialLR":
      self.scheduler = ExponentialLR(self.optimizer, gamma = 0.8,  verbose=True)
      print("The selected learning_rate scheduler strategy is ExponentialLR")

    elif self.lr_scheduler_chose == "PolynomialLR":
      self.scheduler = PolynomialLR(self.optimizer, total_iters=30, power=2,  verbose=True)
      print("The selected learning_rate scheduler strategy is PolynomialLR")

    elif self.lr_scheduler_chose == "CosineAnnealingLR":
      self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=self.min_lr,  verbose=True)
      print("The selected learning_rate scheduler strategy is CosineAnnealingLR")

    elif self.lr_scheduler_chose == "CosineAnnealingWarmRestarts":
      self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=self.min_lr,  verbose=True)
      print("The selected learning_rate scheduler strategy is CosineAnnealingWarmRestarts")

    else: #default ="ReduceLROnPlateau"
      self.lr_scheduler_chose = None
      print("No choosing Learning rate scheduler")


       
  def init_wandb(self):
    #set up wandb for training
    if self.wb == True:
      try:
        print("------------SETTING UP WANDB--------------")
        import wandb
        self.wandb = wandb
        self.wandb.login(key=self.configs["wandb_api_key"])
        print("------Wandb Init-------")

        self.wandb.init(
            project = self.configs["project_name"],
            name = self.name_run_wandb,
            config = self.configs
        )
        self.wandb.watch(self.model, self.criterion, log="all", log_freq=10)
        print()
        print("-----------------------TRAINING MODEL-----------------------")
      except:
          print("--------Can not import wandb-------")

  def compute_metrics(self, preds, labels, num_classes):
    # Convert logits to probabilities
    preds = torch.softmax(preds, dim=1)
    
    # Compute accuracy
    _, predicted = torch.max(preds, 1)
    correct = (predicted == labels).sum().item()
    total = labels.numel()
    accuracy = (correct / total) * 100.0
    
    # Compute Dice score per class
    smooth = 1e-8
    dice_scores = []
    iou_scores = []
    
    for c in range(num_classes):
        # Compute Dice score for class c
        pred_c = (predicted == c).float()
        label_c = (labels == c).float()
        intersection = (pred_c * label_c).sum()
        union = pred_c.sum() + label_c.sum()
        dice_score_c = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice_score_c)
        
        # Compute IoU for class c
        union = (pred_c + label_c).sum() - intersection
        iou_score_c = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou_score_c)
    
    # Average Dice score and IoU across all classes
    dice_score = torch.mean(torch.tensor(dice_scores))
    iou_score = torch.mean(torch.tensor(iou_scores))
    
    return accuracy, dice_score, iou_score
    # return wandb
  def step_per_train(self):
    # if self.wb == True:
    #   self.wandb.watch(model)

    self.model.train()
    seg_train_loss = 0.0
    seg_train_acc = 0.0
    seg_train_dice = 0.0
    seg_train_iou = 0.0

    cls_train_acc = 0.0
    cls_train_loss = 0.0

    train_total_loss = 0.0

    for i, (images, masks, labels) in tqdm.tqdm(
        enumerate(self.train_ds), total = len(self.train_ds), leave = True, colour = "blue", desc = f"Epoch {self.current_epoch_num}",
        bar_format="{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    ):

      # Move images to GPU before feeding them to the model, to fix error happen : Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
      self.model = self.model.cuda()
      
      images = images.cuda(non_blocking = True)
      labels = labels.cuda(non_blocking = True)
      masks = masks.long().to(device=self.device)

      # compute output, accuracy and get loss
      y_seg_pred, y_cls_pred = self.model(images)
      seg_loss = self.criterion(y_seg_pred, masks)
      cls_loss = self.criterion(y_cls_pred, labels)

      total_loss = 0.4 * seg_loss + 0.6 * cls_loss
       # Compute accuracy and dice score
      acc, dice_score, iou_score = self.compute_metrics(y_seg_pred, masks, self.num_classes)
      cls_acc = accuracy(y_cls_pred, labels)[0]

      seg_train_loss += seg_loss.item()
      seg_train_acc += acc
      seg_train_dice += dice_score
      seg_train_iou += iou_score

      cls_train_acc += cls_acc.item()
      cls_train_loss += cls_loss.item()

      train_total_loss += total_loss.item()
      
      # compute gradient and do SGD step
      self.optimizer.zero_grad()
      total_loss.backward()
      self.optimizer.step()

      # write wandb
      metric = {
          " Seg_Loss" : seg_train_loss / (i+1),
          " Seg_Accuracy" :seg_train_acc / (i+1),
          " Cls_Loss" : cls_train_loss / (i+1),
          " Cls_Accuracy" :cls_train_acc / (i+1),
          " Seg_DiceScore" :seg_train_dice / (i+1),
          " Seg_IouScore" :seg_train_iou / (i+1),
          " Total_Loss" : train_total_loss / (i+1),
          " epochs" : self.current_epoch_num,
          " Learning_rate" : get_lr(self.optimizer)
      }
      if self.wb == True and i <= len(self.train_ds):
            self.wandb.log(metric)
      if self.isDebug == 1: 
        break

    i += 1
    print(f'train_cls_acc : {cls_train_acc / i}')
    self.train_loss_list[0].append(seg_train_loss / i)
    self.train_acc_list[0].append(seg_train_acc / i)
    self.train_dice_list.append(seg_train_dice / i)
    self.train_iou_list.append(seg_train_iou / i)

    self.train_loss_list[1].append(cls_train_loss / i)
    self.train_acc_list[1].append(cls_train_acc / i)
    self.train_total_loss_list.append(train_total_loss / i)

    print(seg_train_acc / i)
    print(cls_train_acc / i)

    print(" Seg_Loss: {:.4f}".format(self.train_loss_list[0][-1])
          , ", Seg_Accuracy: {:.4f}%".format(self.train_acc_list[0][-1])
          , ", Dice_score: {:.4f}".format(self.train_dice_list[-1])
          , ", Iou_score: {:.4f}".format(self.train_iou_list[-1])
          , ", Cls_Loss: {:.4f}%".format(self.train_loss_list[1][-1])
          , ", Cls_Accuracy: {:.4f}%".format(self.train_acc_list[1][-1])
          , ", Train_total_loss: {:.4f}".format(self.train_total_loss_list[-1]))

  def step_per_val(self):
    self.model.eval()
    seg_val_loss = 0.0
    seg_val_acc = 0.0
    seg_val_dice = 0.0
    seg_val_iou = 0.0

    cls_val_acc = 0.0
    cls_val_loss = 0.0

    val_total_loss = 0.0

    with torch.no_grad():
      for i, (images, masks, labels) in tqdm.tqdm(
          enumerate(self.val_ds), total = len(self.val_ds), leave = True, colour = "green", desc = "        ",
          bar_format="{desc} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
      ):
        images = images.cuda(non_blocking = True)
        labels = labels.cuda(non_blocking = True)
        masks = masks.long().to(device=self.device)

        # compute output, accuracy and get loss
        y_seg_pred, y_cls_pred = self.model(images)
        seg_loss = self.criterion(y_seg_pred, masks)
        cls_loss = self.criterion(y_cls_pred, labels)
        total_loss = 0.4 * seg_loss + 0.6 * cls_loss
      
       # Compute accuracy and dice score
        acc, dice_score, iou_score = self.compute_metrics(y_seg_pred, masks, self.num_classes)
        cls_acc = accuracy(y_cls_pred, labels)[0]
        seg_val_loss += seg_loss.item()
        seg_val_acc += acc
        seg_val_dice += dice_score
        seg_val_iou += iou_score

        cls_val_acc += cls_acc.item()
        cls_val_loss += cls_loss.item()

        val_total_loss += total_loss.item()
        if self.isDebug == 1: 
          break
      i += 1
      self.val_loss_list[0].append(seg_val_loss / i)
      self.val_acc_list[0].append(seg_val_acc / i)
      self.val_dice_list.append(seg_val_dice / i)
      self.val_iou_list.append(seg_val_iou / i)

      self.val_loss_list[1].append(cls_val_loss / i)
      self.val_acc_list[1].append(cls_val_acc / i)

      self.val_total_loss_list.append(val_total_loss / i)


      print(seg_val_acc / i)
      print(cls_val_acc / i)


      print(" Seg_Val_Loss: {:.4f}".format(self.val_loss_list[0][-1])
            ,", Seg_Val_Accuracy: {:.4f}%".format(self.val_acc_list[0][-1])
            ,", Cls_Val_Loss: {:.4f}%".format(self.val_loss_list[1][-1])
            ,", Cls_Val_Accuracy: {:.4f}%".format(self.val_acc_list[1][-1])
            , ", Seg_Val_Dice: {:.4f}".format(self.val_dice_list[-1])
            , ", Seg_Val_Iou: {:.4f}".format(self.val_iou_list[-1])
            , ", Val_total_loss: {:.4f}".format(self.val_total_loss_list[-1]))

      # write wandb
      if self.wb == True:
        metric = {
            " Seg_Val_Loss" : self.val_loss_list[0][-1],
            " Seg_Val_Accuracy" :self.val_acc_list[0][-1],
            " Val_DiceScore" :self.val_dice_list[-1],
            " Val_IouScore" :self.val_iou_list[-1],
            " Val_Cls_Loss" : self.val_loss_list[1][-1],
            " Val_Cls_Accuracy" :self.val_acc_list[1][-1],
            " Val_total_loss": self.val_total_loss_list[-1]
            # "Learning_rate" : self.learning_rate
        }
        self.wandb.log(metric)

      

  def acc_on_test(self):
    self.model.eval()
    seg_test_loss = 0.0
    seg_test_acc = 0.0
    seg_test_dice = 0.0
    seg_test_iou = 0.0

    cls_test_loss = 0.0
    cls_test_acc = 0.0

    test_total_loss = 0.0
    with torch.no_grad():
      for i, (images, masks, labels) in tqdm.tqdm(
          enumerate(self.test_ds), total = len(self.test_ds), leave = True, colour = "green", desc = "        ",
          bar_format="{desc} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
      ):
        images = images.cuda(non_blocking = True)
        labels = labels.cuda(non_blocking = True)
        masks = masks.long().to(device=self.device)
     
        # compute output, accuracy and get loss
        with torch.cuda.amp.autocast():
          y_seg_pred, y_cls_pred = self.model(images)
          seg_loss = self.criterion(y_seg_pred, masks)
          cls_loss = self.criterion(y_cls_pred, labels)
          total_loss = 0.4 * seg_loss + 0.6 * cls_loss
      
       # Compute accuracy and dice score
        acc, dice_score, iou_score = self.compute_metrics(y_seg_pred, masks, self.num_classes)
        cls_acc = accuracy(y_cls_pred, labels)[0]
        print(f'cls_acc_test ={cls_acc}')
       
        seg_test_loss += seg_loss.item()
        seg_test_acc += acc
        seg_test_dice += dice_score
        seg_test_iou += iou_score
        
        cls_test_acc += cls_acc
        cls_test_loss += cls_loss.item()
    
        test_total_loss += total_loss.item()
        if self.isDebug == 1: 
          break
 
      i += 1
      seg_test_loss = (seg_test_loss / i)
      seg_test_acc = (seg_test_acc / i)
      seg_test_dice = (seg_test_dice / i)
      seg_test_iou = (seg_test_iou / i)
      
      cls_test_acc = cls_test_acc / i
      cls_test_loss = cls_test_loss / i 

      print("Seg_Test_Accuracy: {:.4f}, Cls_Test_Accuracy: {:.4f}, Test_Dice_score: {:.4f}, Test_IOU_score:{:.4f} ".format(seg_test_acc, cls_test_acc, seg_test_dice, seg_test_iou))
      if self.wb == True:
        self.wandb.log({
          "Seg_Test_accuracy": seg_test_acc,
          "Cls_Test_accuracy": cls_test_acc,
          "Test_diceScore": test_dice,
          "Test_iouScore": test_iou
          })
      return seg_test_acc, cls_test_acc, seg_test_dice, seg_test_iou

  def Train_model(self):
    self.init_wandb()
    #self.scheduler.step(100 - self.best_val_acc)
    try:
      if self.load_state_dir != "":
        #shutil.copy(self.load_state_dir, self.checkpoint_path)
        my_checkpoint_path = torch.load(self.checkpoint_path)
        self.model.load_state_dict(my_checkpoint_path['net'])
        self.optimizer.load_state_dict(my_checkpoint_path['optimizer'])
        print("loaded old weight successful")
      while not self.stop_train():
        self.update_epoch_num()
        self.step_per_train()
        self.step_per_val()

        self.update_state_training()
        if self.isDebug == 1:
          break

    except KeyboardInterrupt:
      traceback.print_exc()
      pass
    # Stop training
    try:
      #loading best model
      state = torch.load(self.checkpoint_path)
      self.model.load_state_dict(state["net"])
      print("----------------------Cal on Test-----------------------")
      self.test_acc['segmentation'], self.test_acc['classification'], self.test_dice, self.test_iou = self.acc_on_test()
      self.save_weights()

    except Exception as e:
      traceback.prtin_exc()
      pass

    consume_time = str(datetime.datetime.now() - self.start_time)
    print("----------------------SUMMARY-----------------------")
    print(" After {} epochs and {} plateau count, consume {}".format((self.current_epoch_num), (self.plateau_count),consume_time[:-7]))
    print(" Best Train Seg Accuracy: {:.4f}, Best Train Cls Accuracy: {:.4f}, Dice Score: {:.4f}, Best Train IOU Score:{:.4f} ".format(self.best_train_acc['segmentation'], self.best_train_acc['classification'], self.best_train_dice, self.best_train_iou))
    print(" Best Val Seg Accuracy: {:.4f}, Best Val Cls Accuracy: {:.4f}, Dice Score: {:.4f}, Best Val IOU Score:{:.4f} ".format(self.best_val_acc['segmentation'], self.best_val_acc['classification'], self.best_val_dice, self.best_val_iou))
    print(" Seg Test Accuracy: {:.4f}, Cls Test Accuracy: {:.4f}, Test Dice Score: {:.4f}, Test IOU Score:{:.4f} ".format((self.test_acc['segmentation']), (self.test_acc['classification']), (self.test_dice), (self.test_iou)))

  #set up for training (update epoch, stopping training, write logging)
  def update_epoch_num(self):
    self.current_epoch_num += 1

  def stop_train(self):
    return (
        self.plateau_count > self.max_plateau_count or
        self.current_epoch_num > self.max_epoch_num 
    )
  def update_state_training(self):
    if self.val_acc_list[0][-1] > self.best_val_acc['segmentation'] and self.val_acc_list[1][-1] > self.best_val_acc['classification']:
      self.save_weights()
      self.plateau_count = 0
      self.best_val_acc['segmentation'] = self.val_acc_list[0][-1]
      self.best_val_loss['segmentation'] = self.val_loss_list[0][-1]
      self.best_val_acc['classification'] = self.val_acc_list[1][-1]
      self.best_val_loss['classification'] = self.val_loss_list[1][-1]
      self.best_val_dice = self.val_dice_list[-1]
      self.best_val_iou = self.val_iou_list[-1]

      self.best_train_acc['segmentation'] = self.train_acc_list[0][-1]
      self.best_train_loss['segmentation']  = self.train_loss_list[0][-1]
      self.best_train_acc['classification'] = self.train_acc_list[1][-1]
      self.best_train_loss['classification']  = self.train_loss_list[1][-1]
      
      self.best_train_dice = self.train_dice_list[-1]
      self.best_train_iou = self.train_iou_list[-1]
      
    else:
      self.plateau_count += 1
# 100 - self.best_val_acc
    if self.lr_scheduler_chose == "ReduceLROnPlateau":
      last_val_seg_acc = self.val_acc_list[0][-1]
      last_val_cls_acc = self.val_acc_list[1][-1]
      combined_score = 0.6 * (100 - last_val_cls_acc) + 0.4 * (100 - last_val_seg_acc)
      self.scheduler.step(combined_score)
    else:
      self.scheduler.step()

    if self.optimizer.param_groups[0]['lr'] < self.min_lr:
      self.optimizer.param_groups[0]['lr'] = self.min_lr
#100 - self.best_val_acc

  def save_weights(self):
    state_dict = self.model.state_dict()

    state = {
        **self.configs,
        "net": state_dict,
        "best_train_seg_loss": self.best_train_loss['segmentation'],
        "best_train_seg_acc": self.best_train_acc['segmentation'],
        "seg_train_loss_list": self.train_loss_list[0],
        "seg_train_acc_list": self.train_acc_list[0],

        "best_train_cls_loss": self.best_train_loss['classification'],
        "best_train_cls_acc": self.best_train_acc['classification'],
        "cls_train_loss_list": self.train_loss_list[1],
        "cls_train_acc_list": self.train_acc_list[1],
        "best_train_dice": self.best_train_dice,
        "best_train_iou": self.best_train_iou,

        "best_val_Seg_loss": self.best_val_loss['segmentation'],
        "best_val_Seg_acc": self.best_val_acc['segmentation'],
        "seg_val_loss_list": self.val_loss_list[0],
        "seg_val_acc_list": self.val_acc_list[0],

        "best_val_Cls_loss": self.best_val_loss['classification'],
        "best_val_Cls_acc": self.best_val_acc['classification'],
        "cls_val_loss_list": self.val_loss_list[1],
        "cls_val_acc_list": self.val_acc_list[1],
        "best_val_dice": self.best_val_dice,
        "best_val_iou": self.best_val_iou,

        "seg_test_acc": self.test_acc['segmentation'],
        "cls_test_acc": self.test_acc['classification'],
        "optimizer": self.optimizer.state_dict(),
    }

    torch.save(state, self.checkpoint_path)