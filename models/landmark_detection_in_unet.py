import segmentation_models_pytorch as smp
import torch.nn as nn
import torch

class Landmark_Detection_in_InUnet(smp.Unet):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", num_seg_classes=68, activation=None):
        # Initialize the parent class (smp.Unet)
        super().__init__(encoder_name=encoder_name, encoder_weights=encoder_weights, activation=None, classes=num_seg_classes)

    def forward(self, x):
        #Phase 1: Encoder
        encoder_features = self.encoder(x) 

        #Phase 2: Decoder
        seg_output = self.decoder(*encoder_features) 

        #Output
        seg_output = self.segmentation_head(seg_output) 
        reduce_size = int(seg_output.shape[2]/4)
        
        landmark_output = nn.functional.interpolate(seg_output, size=(reduce_size, reduce_size), mode='bilinear', align_corners=False)
        
        return landmark_output

# # Example usage
# model = Resnet50UnetMultitask(num_seg_classes=8, num_cls_classes=7, activation=None)
# #print(model)
# input_tensor = torch.randn(16, 3, 224, 224)  # Batch of 1 image, 3 channels (RGB), 256x256 size
# seg_output, cls_output = model(input_tensor)
# print(seg_output.shape)  # Should be (1, 2, 256, 256) for 2 classes
# print(cls_output.shape)
