from imgaug import augmenters as iaa
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
#
seg_fer = iaa.Sometimes(
        0.5,
	iaa.Sequential([iaa.Fliplr(p=0.6),iaa.Affine(rotate=(-30, 30))]),
        iaa.Sequential([iaa.Affine(scale=(1.0, 1.1))]),

)
seg_fertest2 = iaa.Sequential([iaa.Affine(scale=(1.0, 1.1))])
seg_fertest1 = iaa.Sequential([iaa.Fliplr(p=0.5),iaa.Affine(rotate=(-30, 30))])


#augmentaion for Image-net
seg_raf = iaa.Sometimes(
        0.5,
	iaa.Sequential([iaa.Fliplr(p=0.5), iaa.Affine(rotate=(-25, 25))]),
        iaa.Sequential([iaa.RemoveSaturation(1),iaa.Affine(scale=(1.0, 1.05)) ])
)
seg_raftest2 = iaa.Sequential([iaa.RemoveSaturation(1),iaa.Affine(scale=(1.0, 1.05))])
seg_raftest1 = iaa.Sequential([iaa.Fliplr(p=0.5), iaa.Affine(rotate=(-25, 25))])



# Define transformations for image
image_transform = A.Compose([
    A.Resize(width=224, height=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Define transformations for mask (no normalization)
mask_transform = A.Compose([
    A.Resize(width=224, height=224, always_apply=True),  # Ensure mask is resized
    ToTensorV2()
])

# Combine transformations into one with additional_targets
class CustomTransform:
    def __init__(self):
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __call__(self, image, mask=None):
        # Apply image transformations
        transformed_image = self.image_transform(image=image)['image']
        
        if mask is not None:
            # Apply mask transformations if mask is provided
            transformed_mask = self.mask_transform(image=mask)['image']
            return {'image': transformed_image, 'mask': transformed_mask}
        return {'image': transformed_image}