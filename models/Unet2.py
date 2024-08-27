import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class UNET(nn.Module):
    
    def __init__(self, in_channels, classes):
        super(UNET, self).__init__()
        
        # Load a pretrained ResNet model
        resnet = models.resnet34(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove the final fully connected layers
        
        # Define decoder layers
        self.up_trans1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = self.__double_conv(1024, 512)
        
        self.up_trans2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = self.__double_conv(512, 256)
        
        self.up_trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = self.__double_conv(256, 128)
        
        self.up_trans4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = self.__double_conv(128, 64)
        
        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)

    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv

    def forward(self, x):
        # Encoder
        x1 = self.encoder[0:4](x)  # Conv1_x
        x2 = self.encoder[4:6](x1)  # MaxPool2d and Conv2_x
        x3 = self.encoder[6:8](x2)  # MaxPool2d and Conv3_x
        x4 = self.encoder[8:10](x3)  # MaxPool2d and Conv4_x
        x5 = self.encoder[10:12](x4)  # MaxPool2d and Conv5_x
        
        # Decoder
        x = self.up_trans1(x5)
        x = F.interpolate(x, size=x4.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x4, x], dim=1)
        x = self.up_conv1(x)
        
        x = self.up_trans2(x)
        x = F.interpolate(x, size=x3.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x3, x], dim=1)
        x = self.up_conv2(x)
        
        x = self.up_trans3(x)
        x = F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x2, x], dim=1)
        x = self.up_conv3(x)
        
        x = self.up_trans4(x)
        x = F.interpolate(x, size=x1.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x1, x], dim=1)
        x = self.up_conv4(x)
        
        x = self.final_conv(x)
        
        return x
