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


class RAFDB_Segmentation_Trainer(Trainer):
  def __init__(self, model, train_loader, val_loader, test_loader,test_loader_ttau, configs, wb = True):

    self.train_loader = train_loader
    self.val_loader = val_loader
    self.test_loader = test_loader
    self.test_loader_ttau = test_loader_ttau


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
    self.val_loss_list = []
    self.val_acc_list = []
    self.val_dice_list = []
    self.best_train_acc = 0.0
    self.best_val_acc = 0.0
    self.best_train_loss = 0.0
    self.best_val_loss = 0.0
    self.best_train_dice = 0.0
    self.best_val_dice = 0.0
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
    if self.isDebug == 1:
      print("Debug mode activated")

      self.max_epoch_num = 60

      n_train_debug = 100 if len(train_loader)> 100 else len(train_loader)
      self.train_loader.label = train_loader.label[: n_train_debug]
      self.train_loader.file_paths = train_loader.file_paths[: n_train_debug]

      n_val_debug = 100 if len(val_loader)> 100 else len(val_loader)
      self.val_loader.label = val_loader.label[: n_val_debug]
      self.val_loader.file_paths = val_loader.file_paths[: n_val_debug]

      n_test_debug = 100 if len(test_loader)> 100 else len(test_loader)
      self.test_loader.label = test_loader.label[: n_test_debug]
      self.test_loader.file_paths = test_loader.file_paths[: n_test_debug]

      n_test_ttau_debug = 100 if len(test_loader_ttau)> 100 else len(test_loader_ttau)
      self.test_loader_ttau.label = test_loader_ttau.label[: n_test_ttau_debug]
      self.test_loader_ttau.file_paths = test_loader_ttau.file_paths[: n_test_ttau_debug]

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

  def compute_metrics(preds, labels, num_classes):
    # Convert logits to probabilities
    preds = torch.softmax(preds, dim=1)
    
    # Compute accuracy
    _, predicted = torch.max(preds, 1)
    correct = (predicted == labels).sum().item()
    total = labels.numel()
    accuracy = correct / total
    
    # Compute Dice score per class
    smooth = 1e-8
    dice_scores = []
    for c in range(num_classes):
        # Compute Dice score for class c
        pred_c = (predicted == c).float()
        label_c = (labels == c).float()
        intersection = (pred_c * label_c).sum()
        union = pred_c.sum() + label_c.sum()
        dice_score_c = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice_score_c)
    
    # Average Dice score across all classes
    dice_score = torch.mean(torch.tensor(dice_scores))
    
    return accuracy, dice_score
    # return wandb
  def step_per_train(self):
    # if self.wb == True:
    #   self.wandb.watch(model)

    self.model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_dice = 0.0

    for i, (images, masks, labels) in tqdm.tqdm(
        enumerate(self.train_ds), total = len(self.train_ds), leave = True, colour = "blue", desc = f"Epoch {self.current_epoch_num}",
        bar_format="{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    ):

      # Move images to GPU before feeding them to the model, to fix error happen : Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
      self.model = self.model.cuda()
      
      images = images.to(device=self.device)
      masks = masks.long().to(device=self.device)

      # compute output, accuracy and get loss
      with torch.cuda.amp.autocast():
        y_pred = self.model(images)
        loss = self.criterion(y_pred, masks)
      
       # Compute accuracy and dice score
      print(f'num classes = {self.num_classes}')
      print(f'num y_pred = {y_pred.shape}')
      print(f'num masks = {masks.shape}')
      acc, dice_score = self.compute_metrics(y_pred, masks, self.num_classes)
      

      train_loss += loss.item()
      train_acc += acc
      train_dice += dice_score

      # compute gradient and do SGD step
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      # write wandb
      metric = {
          " Loss" : train_loss / (i+1),
          " Accuracy" :train_acc / (i+1),
          " DiceScore" :train_dice / (i+1),
          " epochs" : self.current_epoch_num,
          " Learning_rate" : get_lr(self.optimizer)
      }
      if self.wb == True and i <= len(self.train_ds):
            self.wandb.log(metric)

      
    i += 1
    self.train_loss_list.append(train_loss / i)
    self.train_acc_list.append(train_acc / i)
    self.train_dice_list.append(train_dice / i)


    print(" Loss: {:.4f}".format(self.train_loss_list[-1]), ", Accuracy: {:.2f}%".format(self.train_acc_list[-1]), ", Dice_score: {:.3f}%".format(self.train_dice_list[-1]))

  def step_per_val(self):
    self.model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_dice = 0.0

    with torch.no_grad():
      for i, (images, masks, labels) in tqdm.tqdm(
          enumerate(self.val_ds), total = len(self.val_ds), leave = True, colour = "green", desc = "        ",
          bar_format="{desc} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
      ):
        images = images.to(device=self.device)
        masks = masks.long().to(device=self.device)

        # compute output, accuracy and get loss
        with torch.cuda.amp.autocast():
          y_pred = self.model(images)
          loss = self.criterion(y_pred, masks)
      
       # Compute accuracy and dice score
        acc, dice_score = self.compute_metrics(y_pred, masks, self.num_classes)

        val_loss += loss.item()
        val_acc += acc.item
        val_dice += dice_score
      i += 1
      self.val_loss_list.append(val_loss / i)
      self.val_acc_list.append(val_acc / i)
      self.val_dice_list.append(val_dice / i)

      print(" Val_Loss: {:.4f}".format(self.val_loss_list[-1]),", Val_Accuracy: {:.2f}%".format(self.val_acc_list[-1]), ", Val_Dice: {:.2f}%".format(self.val_dice_list[-1]))

      # write wandb
      if self.wb == True:
        metric = {
            " Val_Loss" : self.val_loss_list[-1],
            " Val_Accuracy" :self.val_acc_list[-1],
            " Val_DiceScore" :self.val_dice_list[-1],
            # "Learning_rate" : self.learning_rate
        }
        self.wandb.log(metric)

      

  def acc_on_test(self):
    self.model.eval()
    test_loss = 0.0
    test_acc = 0.0
    test_dice = 0.0

    with torch.no_grad():
      for i, (images, masks, labels) in tqdm.tqdm(
          enumerate(self.test_ds), total = len(self.test_ds), leave = True, colour = "green", desc = "        ",
          bar_format="{desc} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
      ):
        images = images.to(device=self.device)
        masks = masks.long().to(device=self.device)

        # compute output, accuracy and get loss
        with torch.cuda.amp.autocast():
          y_pred = self.model(images)
          loss = self.criterion(y_pred, masks)
      
       # Compute accuracy and dice score
        acc, dice_score = self.compute_metrics(y_pred, masks, self.num_classes)

        test_loss += loss.item()
        test_acc += acc
        test_dice += dice_score

        # print(i)
      i += 1
      test_loss = (test_loss / i)
      test_acc = (test_acc / i)
      test_dice = (test_dice / i)

      print("Accuracy on Test_ds: {:.3f}".format(test_acc))
      if self.wb == True:
        self.wandb.log({
          "Test_accuracy": test_acc,
          "Test_diceScore": test_dice
          })
      return test_acc, test_dice

  def acc_on_test_ttau(self):
    self.model.eval()
    test_acc = 0.0
    test_dice = 0.0

    # Create accumulators for metrics
    total_batches = len(self.test_loader_ttau)

    with torch.no_grad():
        for idx in tqdm.tqdm(
            range(total_batches), total=total_batches, leave=False,
            desc="Evaluating"
        ):
            images, masks, labels = self.test_loader_ttau[idx]
            masks = torch.LongTensor([masks]).cuda(non_blocking=True)

            images = make_batch(images)
            images = images.cuda(non_blocking=True)

            y_pred = self.model(images)
            y_pred = F.softmax(y_pred, dim=1)

            y_pred = torch.sum(y_pred, dim=0)
            y_pred = torch.unsqueeze(y_pred, 0)

            # Compute accuracy and Dice score
            acc, dice_score = self.compute_metrics(y_pred, masks, self.num_classes)

            test_acc += acc
            test_dice += dice_score

    # Average metrics
    test_acc /= total_batches
    test_dice /= total_batches

    print(f"Accuracy on Test_ds with TTAU: {test_acc:.3f}")
    print(f"Dice Score on Test_ds with TTAU: {test_dice:.3f}")

    # Log metrics to wandb
    if self.wb:
        self.wandb.log({
            "Testta_accuracy": test_acc,
            "Testta_diceScore": test_dice
        })

    return test_acc, test_dice

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

    except KeyboardInterrupt:
      traceback.print_exc()
      pass
    # Stop training
    try:
      #loading best model
      state = torch.load(self.checkpoint_path)
      self.model.load_state_dict(state["net"])
      print("----------------------Cal on Test-----------------------")
      self.test_acc = self.acc_on_test()
      self.test_acc_ttau = self.acc_on_test_ttau()
      self.save_weights()

    except Exception as e:
      traceback.prtin_exc()
      pass

    consume_time = str(datetime.datetime.now() - self.start_time)
    print("----------------------SUMMARY-----------------------")
    print(" After {} epochs and {} plateau count, consume {}".format((self.current_epoch_num), (self.plateau_count),consume_time[:-7]))
    print(" Best Accuracy on Train: {:.3f} ".format(self.best_train_acc))
    print(" Best Accuracy on Val: {:.3f} ".format(self.best_val_acc))
    print(" Best Accuracy on Test: {:.3f} ".format(self.test_acc))
    print(" Best Accuracy on Test with tta: {:.3f} ".format(self.test_acc_ttau))

  #set up for training (update epoch, stopping training, write logging)
  def update_epoch_num(self):
    self.current_epoch_num += 1

  def stop_train(self):
    return (
        self.plateau_count > self.max_plateau_count or
        self.current_epoch_num > self.max_epoch_num
    )
  
  def update_state_training(self):
    if self.val_acc_list[-1] > self.best_val_acc:
      self.save_weights()
      self.plateau_count = 0
      self.best_val_acc = self.val_acc_list[-1]
      self.best_val_loss = self.val_loss_list[-1]
      self.best_train_acc = self.train_acc_list[-1]
      self.best_train_loss = self.train_loss_list[-1]
      self.best_train_dice = self.train_dice_list[-1]
      self.best_train_dice = self.val_dice_list[-1]
    else:
      self.plateau_count += 1
# 100 - self.best_val_acc
    if self.lr_scheduler_chose == "ReduceLROnPlateau":
      self.scheduler.step(100 - self.val_acc_list[-1])
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
        "best_val_loss": self.best_val_loss,
        "best_val_acc": self.best_val_acc,
        "best_train_loss": self.best_train_loss,
        "best_train_acc": self.best_train_acc,
        "train_loss_list": self.train_loss_list,
        "val_loss_list": self.val_loss_list,
        "train_acc_list": self.train_acc_list,
        "val_acc_list": self.val_acc_list,
        "test_acc": self.test_acc,
        "best_train_dice": self.best_train_dice,
        "best_val_dice": self.best_val_dice,
        "optimizer": self.optimizer.state_dict(),
    }

    torch.save(state, self.checkpoint_path)