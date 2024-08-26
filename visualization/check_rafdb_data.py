import tqdm 
import matplotlib.colors as mcolors
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sgu24project.utils.datasets.rafdb_ds_with_mask import RafDataSet_Mask

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--type-data', default= "train", type=str, help='type data')
parser.add_argument('--batch-size', default= 42, type=int, help='batch_size')
parser.add_argument('--use-shuffle', default= 1, type=int, help='1 is use, 0 is not')
parser.add_argument('--use-albumentation', default= 1, type=int, help='1 is use, 0 is not')
args, unknown = parser.parse_known_args()
configs = {
          "raf_path": "/kaggle/input/rafdb-mask-basic-15k3",
    "image_path": "rafdb_mask_basic/Image/aligned/",
    "label_path": "rafdb_mask_basic/EmoLabel/list_patition_label.txt",
    "image_size": 224,
          }

train_loader = RafDataSet_Mask( "train", configs, use_albumentation = True if args.albumentation == 1 else False)
total_image = len(train_loader)
train_ds = DataLoader(
                train_loader,
                batch_size=args.batch_size,
                pin_memory=True,
                shuffle=True if args.use_shuffle == 1 else False,
                worker_init_fn=lambda x: np.random.seed(x),
            )
print(len(train_ds))

stop = 0 
for i, (images, masks, labels) in tqdm.tqdm(
        enumerate(train_ds), total=len(train_ds), leave=True, colour="blue", desc=f"Epoch {0}",
        bar_format="{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    ):
    batch_size = len(images)  # or len(labels), assuming they have the same length
    # Optionally, you can ccperform your training steps here
    fig, ax = plt.subplots(batch_size,2,figsize=(10, batch_size * 2))
    # Example: print batch size
    print(f"list all distinct value in all masks = {masks.view(-1).unique()}")

    for idx in range(batch_size):
        ax[idx, 0].imshow(images[idx].permute(1,2,0))
        ax[idx, 0].set_title(f'image({images[idx].shape})')
        ax[idx, 0].axis('off')

        ax[idx, 1].imshow(masks[idx], cmap = mcolors.ListedColormap(['#440154', 'blue', 'green', 'red', 'yellow', 'orange']))
        ax[idx, 1].set_title('one-hot-mask')
        ax[idx, 1].axis('off')
        #print(masks[0].tolist())
    break
