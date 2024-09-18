import segmentation_models_pytorch as smp
import torch.nn as nn
import torch

class Resnet50UnetMultitask_v2(smp.Unet):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, num_seg_classes=2, num_cls_classes=7, activation=None):
        # Initialize the parent class (smp.Unet)
        super().__init__(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, activation=None, classes=num_seg_classes)
        
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
        
        #seg_output = torch.randn(42, 6, 224, 224)
        return seg_output, cls_output

class Resnet50_in_smp(smp.Unet):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, num_seg_classes=2, num_cls_classes=7, activation=None):
        # Initialize the parent class (smp.Unet)
        super().__init__(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, activation=None, classes=num_seg_classes)
        
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
        
        #seg_output = torch.randn(42, 6, 224, 224)
        return cls_output