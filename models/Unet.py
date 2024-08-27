import torchvision.models as models
import torch.nn as nn
import torch

class UNET(nn.Module):
    def __init__(self, in_channels, classes):
        super(UNET, self).__init__()
        self.encoder = models.resnet34(pretrained=True)
        self.encoder_layers = list(self.encoder.children())
        self.encoder_conv1 = nn.Sequential(*self.encoder_layers[:4])
        self.encoder_conv2 = self.encoder_layers[4]
        self.encoder_conv3 = self.encoder_layers[5]
        self.encoder_conv4 = self.encoder_layers[6]
        self.encoder_conv5 = self.encoder_layers[7]
        
        self.up_trans = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        ])
        
        self.double_conv_ups = nn.ModuleList([
            self.__double_conv(512, 256),
            self.__double_conv(256, 128),
            self.__double_conv(128, 64)
        ])
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
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
        enc1 = self.encoder_conv1(x)
        enc2 = self.encoder_conv2(enc1)
        enc3 = self.encoder_conv3(enc2)
        enc4 = self.encoder_conv4(enc3)
        enc5 = self.encoder_conv5(enc4)

        x = self.up_trans[0](enc5)
        x = self.__double_conv(512, 256)(torch.cat([x, enc4], dim=1))

        x = self.up_trans[1](x)
        x = self.__double_conv(256, 128)(torch.cat([x, enc3], dim=1))

        x = self.up_trans[2](x)
        x = self.__double_conv(128, 64)(torch.cat([x, enc2], dim=1))

        x = self.final_conv(x)
        return x
