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
                batch_size=len(data_loader),
                pin_memory=True,
                shuffle=True if args.use_shuffle == 1 else False,
                worker_init_fn=lambda x: np.random.seed(x),
            )

for i, (images, masks) in tqdm.tqdm(
        enumerate(test_ds), total=len(test_ds), leave=True, colour="blue", desc=f"Batch {0}",
        bar_format="{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    ):
    # Optionally, you can ccperform your training steps here
    fig, ax = plt.subplots(batch_size,7,figsize=(10, batch_size * 2))
    # Example: print batch size
    print(f"list all distinct value in all masks = {masks.view(-1).unique()}")
    count = 0
    print(len(images))
    for idx in range(len(images)):
        print(count)
        #if(len(np.unique(masks[idx][4])) == 1):
            #continue
        print(len(np.unique(masks[idx][4])))
        count = count + 1
        ax[idx, 0].imshow(images[idx].permute(1,2,0))
        ax[idx, 0].set_title(f'image')
        ax[idx, 0].axis('off')

        ax[idx, 1].imshow(masks[idx][0])
        ax[idx, 1].set_title('eyes_mask')
        ax[idx, 1].axis('off')

        ax[idx, 2].imshow(masks[idx][1])
        ax[idx, 2].set_title('eyebrows_mask')
        ax[idx, 2].axis('off')

        ax[idx, 3].imshow(masks[idx][2])
        ax[idx, 3].set_title('nose_mask')
        ax[idx, 3].axis('off')

        ax[idx, 4].imshow(masks[idx][3])
        ax[idx, 4].set_title('mouth_mask')
        ax[idx, 4].axis('off')

        ax[idx, 5].imshow(masks[idx][4])
        ax[idx, 5].set_title('face_mask')
        ax[idx, 5].axis('off')

        ax[idx, 6].imshow(masks[idx][5])
        ax[idx, 6].set_title('background')
        ax[idx, 6].axis('off')
        
        #print(masks[0].tolist())
        if count == args.batch_size:
            break
plt.tight_layout()
plt.show()

