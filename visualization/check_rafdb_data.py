import tqdm 
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sgu24project.utils.datasets.rafdb_ds_with_mask import RafDataSet_Mask
configs = {
          "raf_path": "/kaggle/input/rafdb-mask-basic-15k3",
    "image_path": "rafdb_mask_basic/Image/aligned/",
    "label_path": "rafdb_mask_basic/EmoLabel/list_patition_label.txt",
    "image_size": 224,
          }

train_loader = RafDataSet_Mask( "train", configs, use_albumentation = True)
train_loader = RafDataSet_Mask("test", configs, ttau = False, len_tta = 48) 
total_image = len(train_loader)
batch_size = 42
train_ds = DataLoader(
                train_loader,
                batch_size=batch_size,
                pin_memory=True,
                shuffle=False,
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
    fig, ax = plt.subplots(batch_size,2,figsize=(6,6))
    # Example: print batch size
    print(f"list all distinct value in all masks = {masks.view(-1).unique()}")

    for idx in range(batch_size):
        ax[idx, 0].imshow(images[idx].permute(1,2,0))
        ax[idx, 0].set_title(f'image({images[idx].shape})')
        ax[idx, 0].axis('off')

        ax[idx, 1].imshow(masks[idx], cmap = mcolors.ListedColormap(['#440154', 'blue', 'green', 'red', 'yellow', 'orange']))
        ax[idx, 1].set_title('one-hot-mask')
        ax[idx, 1].axis('off')
        print(masks[0].max().item())
        #print(masks[0].tolist())
    break
