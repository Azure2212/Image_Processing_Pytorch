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
import torchmetrics
import segmentation_models_pytorch as smp
# from utils.radam import RAdam

import tqdm
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiplicativeLR, StepLR, MultiStepLR, ConstantLR, LinearLR, PolynomialLR, CosineAnnealingLR, ChainedScheduler, ExponentialLR, SequentialLR, ReduceLROnPlateau, CyclicLR, CosineAnnealingWarmRestarts

from sgu24project.utils.metrics.metrics import accuracy, make_batch, dice_coeff, multiclass_dice_coeff, dice_loss

import warnings
warnings.filterwarnings('ignore')
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Trainer(object):
    """base class for trainers"""

    def __init__(self):
        pass


class RAFDB_Segmentation_Trainer_v2(Trainer):
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
    self.num_seg_classes = configs["num_seg_classes"]
    print(f'self.num_seg_classes ={self.num_seg_classes}')
    #self.model = model.to(self.device)'cpu'
    self.model = model.to(self.device)

# Move the model to the device
    '''try:
      model = model.to(self.device)
    except Exception as e:
      print("Error:", e)'''

    self.train_loss_list = []
    self.train_acc_list = []
    self.train_dice_list = []
    self.train_iou_list = []

    self.val_loss_list = []
    self.val_acc_list = []
    self.val_dice_list = []
    self.val_iou_list = []

    self.best_train_acc = 0.0
    self.best_train_loss = 0.0
    self.best_train_dice = 0.0
    self.best_train_iou = 0.0

    self.best_val_acc = 0.0
    self.best_val_loss = 0.0
    self.best_val_dice = 0.0
    self.best_val_iou = 0.0

    self.test_acc = 0.0
    self.test_acc_ttau = 0.0
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
    
    
    self.criterion = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

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

  # def compute_metrics(self, pred_mask, mask, num_classes):
  #   tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="multiclass", num_classes=num_classes)
  #   per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
  #   dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
  #   return per_image_iou, dataset_iou


  def compute_metrics(self, y_pred, y_true, num_classes):
    """
    Compute Dice score and IoU score for a batch of images.

    Args:
        y_pred (torch.Tensor): The predicted output from the model, with shape (batch_size, num_classes, height, width).
        y_true (torch.Tensor): The ground truth masks, with shape (batch_size, height, width).
        num_classes (int): The number of segmentation classes.

    Returns:
        dice_scores (list of floats): The Dice score for each class.
        iou_scores (list of floats): The IoU score for each class.
    """
    epsilon = 1e-6
    # Convert predictions to class indices
    y_pred = torch.argmax(y_pred, dim=1)  # Shape: (batch_size, height, width)
  
    dice_scores = []
    iou_scores = []
    for i in range(num_classes):
        # Create binary masks for the i-th class
        pred_mask = (y_pred == i).float()
        true_mask = (y_true[:, i] == 1).float() 
        
        pred_mask_flat = torch.flatten(pred_mask)
        true_mask_flat = torch.flatten(true_mask)
        
        # Compute Dice score for the i-th class
        intersection = torch.sum(pred_mask_flat * true_mask_flat)
        pred_and_true = torch.sum(pred_mask_flat) + torch.sum(true_mask_flat)
        dice_score = (2. * intersection) / (pred_and_true + epsilon)

        # Compute IoU score for the i-th class
        union = torch.sum(pred_mask_flat + true_mask_flat) - intersection
        iou_score = intersection / (union + 1e-6)  # Add a small constant to avoid division by zero

        dice_scores.append(dice_score.item())
        iou_scores.append(iou_score.item())

    dice_score = torch.mean(torch.tensor(dice_scores)) * 100.0
    iou_score = torch.mean(torch.tensor(iou_scores)) * 100.0
    
    return dice_score, iou_score

    
    # return wandb
  def step_per_train(self):
    # if self.wb == True:
    #   self.wandb.watch(model)

    self.model.train()
    train_loss = 0.0
    train_dice = 0.0
    train_iou = 0.0

    for i, (images, masks) in tqdm.tqdm(
        enumerate(self.train_ds), total = len(self.train_ds), leave = True, colour = "blue", desc = f"Epoch {self.current_epoch_num}",
        bar_format="{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    ):

      # Move images to GPU before feeding them to the model, to fix error happen : Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
      self.model = self.model.float().cuda()
      images = images.to(dtype=torch.float).cuda()
      masks = masks.to(dtype=torch.float).cuda()

      # compute output, accuracy and get loss
      
      y_pred = self.model(images)
      
      if self.num_seg_classes == 1:
        #loss = self.criterion(y_pred.squeeze(1), masks.float())
        loss = dice_loss(F.sigmoid(y_pred.squeeze(1)), masks.float(), multiclass=False)
      else:
        #loss = self.criterion(y_pred, masks)
        loss = dice_loss(
            F.softmax(y_pred, dim=1).float(),
            F.one_hot(masks.argmax(dim=1), self.num_seg_classes).permute(0, 3, 1, 2).float(),
            multiclass=True
          )
      
      # Compute accuracy and dice score
      #dice_score, iou_score = self.compute_metrics(y_pred, masks, self.num_seg_classes)
      if self.num_seg_classes == 1:
        assert masks.min() >= 0 and masks.max() <= 1, 'True mask indices should be in [0, 1]'
        y_pred = (F.sigmoid(y_pred) > 0.5).float()
        # compute the Dice score
        dice_score += dice_coeff(y_pred, masks, reduce_batch_first=False)
      else:
        assert masks.min() >= 0 and masks.max() < self.num_seg_classes, 'True mask indices should be in [0, n_classes['
        # convert to one-hot format
        masks = F.one_hot(masks, self.num_seg_classes).permute(0, 3, 1, 2).float()
        y_pred = F.one_hot(y_pred.argmax(dim=1), self.num_seg_classes).permute(0, 3, 1, 2).float()
        # compute the Dice score, ignoring background
        dice_score += multiclass_dice_coeff(y_pred[:, 1:], masks[:, 1:], reduce_batch_first=False)

      train_loss += loss.item()
      train_dice += dice_score.item()
      train_iou += iou_score.item()

      # compute gradient and do SGD step
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      # write wandb
      metric = {
          " Loss" : train_loss / (i+1),
          " DiceScore" :train_dice / (i+1),
          " IouScore" :train_iou / (i+1),
          " epochs" : self.current_epoch_num,
          " Learning_rate" : get_lr(self.optimizer)
      }
      if self.wb == True and i <= len(self.train_ds):
            self.wandb.log(metric)
      if self.isDebug == 1: 
        break

    i += 1
    self.train_loss_list.append(train_loss / i)
    self.train_dice_list.append(train_dice / i)
    self.train_iou_list.append(train_iou / i)


    print(" Loss: {:.4f}".format(self.train_loss_list[-1])
          , ", Dice_score: {:.4f}%".format(self.train_dice_list[-1])
          , ", Iou_score: {:.4f}%".format(self.train_iou_list[-1]))

  def step_per_val(self):
    self.model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0

    with torch.no_grad():
      for i, (images, masks) in tqdm.tqdm(
          enumerate(self.val_ds), total = len(self.val_ds), leave = True, colour = "green", desc = "        ",
          bar_format="{desc} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
      ):
        self.model = self.model.float().cuda()
        images = images.to(dtype=torch.float).cuda()
        masks = masks.to(dtype=torch.float).cuda()
        # compute output, accuracy and get loss
    
        y_pred = self.model(images)
      
      if self.num_seg_classes == 1:
        #loss = self.criterion(y_pred.squeeze(1), masks.float())
        loss = dice_loss(F.sigmoid(y_pred.squeeze(1)), masks.float(), multiclass=False)
      else:
        #loss = self.criterion(y_pred, masks)
        loss = dice_loss(
            F.softmax(y_pred, dim=1).float(),
            F.one_hot(masks, self.num_seg_classes).permute(0, 3, 1, 2).float(),
            multiclass=True
          )
      
      # Compute accuracy and dice score
      #dice_score, iou_score = self.compute_metrics(y_pred, masks, self.num_seg_classes)
      if self.num_seg_classes == 1:
        assert masks.min() >= 0 and masks.max() <= 1, 'True mask indices should be in [0, 1]'
        y_pred = (F.sigmoid(y_pred) > 0.5).float()
        # compute the Dice score
        dice_score += dice_coeff(y_pred, masks, reduce_batch_first=False)
      else:
        assert masks.min() >= 0 and masks.max() < self.num_seg_classes, 'True mask indices should be in [0, n_classes['
        # convert to one-hot format
        masks = F.one_hot(masks, self.num_seg_classes).permute(0, 3, 1, 2).float()
        y_pred = F.one_hot(y_pred.argmax(dim=1), self.num_seg_classes).permute(0, 3, 1, 2).float()
        # compute the Dice score, ignoring background
        dice_score += multiclass_dice_coeff(y_pred[:, 1:], masks[:, 1:], reduce_batch_first=False)

        val_loss += loss.item()
        val_dice += dice_score.item()
        val_iou += iou_score.item()

        #if self.isDebug == 1: 
         # break 
      i += 1
      self.val_loss_list.append(val_loss / i)
      self.val_dice_list.append(val_dice / i)
      self.val_iou_list.append(iou_score / i)

      print(" Val_Loss: {:.4f}".format(self.val_loss_list[-1])
            , ", Val_Dice: {:.4f}%".format(self.val_dice_list[-1])
            , ", Val_Iou: {:.4f}%".format(self.val_iou_list[-1]))

      # write wandb
      if self.wb == True:
        metric = {
            " Val_Loss" : self.val_loss_list[-1],
            " Val_DiceScore" :self.val_dice_list[-1],
            " Val_IouScore" :self.val_iou_list[-1],
        }
        self.wandb.log(metric)

      

  def acc_on_test(self):
    self.model.eval()
    test_loss = 0.0
    test_dice = 0.0
    test_iou = 0.0

    with torch.no_grad():
      for i, (images, masks) in tqdm.tqdm(
          enumerate(self.test_ds), total = len(self.test_ds), leave = True, colour = "green", desc = "        ",
          bar_format="{desc} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
      ):
        self.model = self.model.float().cuda()
        images = images.to(dtype=torch.float).cuda()
        masks = masks.to(dtype=torch.float).cuda()
    
        y_pred = self.model(images)
        after_argmax = torch.argmax(masks, dim=1)
        loss = self.criterion(y_pred, after_argmax)
      
       # Compute accuracy and dice score
        dice_score, iou_score = self.compute_metrics(y_pred, masks, self.num_seg_classes)

        test_loss += loss.item()
        test_dice += dice_score.item()
        test_iou += iou_score.item()

        if self.isDebug == 1: 
          break

      i += 1
      test_loss = (test_loss / i)
      test_dice = (test_dice / i)
      test_iou = (test_iou / i)

      print("Test_Loss: {:.4f}, Test_Dice_score: {:.4f}, Test_IOU_score:{:.4f} ".format(test_loss, test_dice, test_iou))
      if self.wb == True:
        self.wandb.log({
          "Test_diceScore": test_dice,
          "Test_iouScore": test_iou
          })
      return test_loss, test_dice, test_iou

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
      self.test_loss, self.test_dice, self.test_iou = self.acc_on_test()
      self.save_weights()

    except Exception as e:
      traceback.prtin_exc()
      pass

    consume_time = str(datetime.datetime.now() - self.start_time)
    print("----------------------SUMMARY-----------------------")
    print(" After {} epochs and {} plateau count, consume {}".format((self.current_epoch_num), (self.plateau_count),consume_time[:-7]))
    print(" Best Train Loss: {:.4f}, Best Train Dice Score: {:.4f}, Best Train IOU Score:{:.4f} ".format(self.best_train_loss, self.best_train_dice, self.best_train_iou))
    print(" Best Val Loss: {:.4f}, Best Val Dice Score: {:.4f}, Best Val IOU Score:{:.4f} ".format(self.best_val_loss, self.best_val_dice, self.best_val_iou))
    print(" Test Loss: {:.4f}, Test Dice Score: {:.4f}, Test IOU Score:{:.4f} ".format((self.test_loss), (self.test_dice), (self.test_iou)))

  #set up for training (update epoch, stopping training, write logging)
  def update_epoch_num(self):
    self.current_epoch_num += 1

  def stop_train(self):
    return (
        self.plateau_count > self.max_plateau_count or
        self.current_epoch_num > self.max_epoch_num 
    )
  
  def update_state_training(self):
    if self.val_iou_list[-1] > self.best_val_acc:
      self.save_weights()
      self.plateau_count = 0

      self.best_val_loss = self.val_loss_list[-1]
      self.best_val_dice = self.val_dice_list[-1]
      self.best_val_iou = self.val_iou_list[-1]

      self.best_train_loss = self.train_loss_list[-1]
      self.best_train_dice = self.train_dice_list[-1]
      self.best_train_iou = self.train_iou_list[-1]
      
      self.best_val_acc = self.val_iou_list[-1]
    else:
      self.plateau_count += 1
# 100 - self.best_val_acc
    if self.lr_scheduler_chose == "ReduceLROnPlateau":
      self.scheduler.step(100 - self.val_iou_list[-1])
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
        "best_train_loss": self.best_train_loss,
        "train_loss_list": self.train_loss_list,
        "train_acc_list": self.train_acc_list,
        "best_train_dice": self.best_train_dice,
        "best_train_iou": self.best_train_iou,

        "best_val_loss": self.best_val_loss,
        "val_loss_list": self.val_loss_list,
        "val_acc_list": self.val_acc_list,
        "best_val_dice": self.best_val_dice,
        "best_val_iou": self.best_val_iou,
        "optimizer": self.optimizer.state_dict(),
    }

    torch.save(state, self.checkpoint_path)