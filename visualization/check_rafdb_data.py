import tqdm 
import matplotlib.colors as mcolors
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sgu24project.utils.datasets.rafdb_ds_with_mask_v2 import RafDataSet_Mask

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
CLASSES = ['eyes_mask', 'eyebrows_mask', 'nose_mask', 'mouth_mask', 'face_mask']
data_loader = RafDataSet_Mask(data_type = args.type_data, configs = configs , classes=CLASSES)
test_ds = DataLoader(
                data_loader,
                batch_size=args.batch_size*5,
                pin_memory=True,
                shuffle=True if args.use_shuffle == 1 else False,
                worker_init_fn=lambda x: np.random.seed(x),
            )
stop = 0 
for i, (images, masks) in tqdm.tqdm(
        enumerate(test_ds), total=len(test_ds), leave=True, colour="blue", desc=f"Batch {0}",
        bar_format="{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    ):
    batch_size = len(images)  # or len(labels), assuming they have the same length
    # Optionally, you can ccperform your training steps here
    fig, ax = plt.subplots(args.batch_size,7,figsize=(10, batch_size * 2))
    # Example: print batch size
    print(f"list all distinct value in all masks = {masks.view(-1).unique()}")

    for idx in range(len(images[i])):
        if(len(np.unique(masks[idx][4])) == 1):
            continue
        ax[stop, 0].imshow(images[idx].permute(1,2,0))
        ax[stop, 0].set_title(f'image')
        ax[stop, 0].axis('off')

        ax[stop, 1].imshow(masks[idx][0])
        ax[stop, 1].set_title('eyes_mask')
        ax[stop, 1].axis('off')

        ax[stop, 2].imshow(masks[idx][1])
        ax[stop, 2].set_title('eyebrows_mask')
        ax[stop, 2].axis('off')

        ax[stop, 3].imshow(masks[idx][2])
        ax[stop, 3].set_title('nose_mask')
        ax[stop, 3].axis('off')

        ax[stop, 4].imshow(masks[idx][3])
        ax[stop, 4].set_title('mouth_mask')
        ax[stop, 4].axis('off')

        ax[stop, 5].imshow(masks[idx][4])
        ax[stop, 5].set_title('face_mask')
        ax[stop, 5].axis('off')

        ax[stop, 6].imshow(masks[idx][5])
        ax[stop, 6].set_title('background')
        ax[stop, 6].axis('off')
        
        stop = stop + 1
        #print(masks[0].tolist())
        if stop == args.batch_size:
            break

    if stop == args.batch_size:
        break

plt.tight_layout()
plt.show()