import segmentation_models_pytorch as smp
import torch.nn as nn

class CustomUNet(smp.Unet):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", num_seg_classes=6, num_cls_classes=7, activation=None):
        # Initialize the parent class (smp.Unet)
        super().__init__(encoder_name=encoder_name, encoder_weights=encoder_weights, activation=None, classes=num_seg_classes)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_cls_classes)
    def forward(self, x):
        #Phase 1: Encoder
        encoder_features = self.encoder(x) 
        
        #make classification Task
        stage4_resnet50 = encoder_features[5]
        avg = self.avgpool(stage4_resnet50)
        flat = torch.flatten(avg, 1)
        cls_output = self.fc(flat)
        
        # Phase 2: Decoder
        seg_output = self.decoder(*encoder_features) 
        seg_output = self.segmentation_head(seg_output) 
        
        return seg_output, cls_output

# # Example usage
# model = CustomUNet(num_seg_classes=8, num_cls_classes=7, activation=None)
# #print(model)
# input_tensor = torch.randn(16, 3, 224, 224)  # Batch of 1 image, 3 channels (RGB), 256x256 size
# seg_output, cls_output = model(input_tensor)
# print(seg_output.shape)  # Should be (1, 2, 256, 256) for 2 classes
# print(cls_output.shape)
