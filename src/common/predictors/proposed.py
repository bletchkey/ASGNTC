import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.utils.toroidal import ToroidalConv2d
from configs.constants import *

class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()

        conv_layer = ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))

        self.block = nn.Sequential(
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            conv_layer,
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            conv_layer,
        )

    def forward(self, x):
        return x + self.block(x)

class AttentionBlock(nn.Module):
    def __init__(self, num_hidden):
        super(AttentionBlock, self).__init__()

        conv_layer = ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))

        self.block = nn.Sequential(
            conv_layer,
            nn.BatchNorm2d(num_hidden),
        )

        self. psi_conv = nn.Conv2d(num_hidden, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        g1  = self.block(x)
        x1  = self.block(x)
        psi = F.relu(g1 + x1)
        psi = self.psi_conv(psi)
        psi = self.sigmoid(psi)
        return x * psi

class UNetResNetAttention(nn.Module):
    def __init__(self, num_hidden):
        super(UNetResNetAttention, self).__init__()

        self.num_hidden = num_hidden

        self.enc1    = ToroidalConv2d(nn.Conv2d(GRID_NUM_CHANNELS, num_hidden, kernel_size=3, stride=1, padding=0))
        self.enc_bn1 = nn.BatchNorm2d(num_hidden)
        self.res1    = ResBlock(num_hidden)
        self.pool1   = nn.MaxPool2d(2)

        self.enc2    = ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))
        self.enc_bn2 = nn.BatchNorm2d(num_hidden)
        self.res2    = ResBlock(num_hidden)
        self.pool2   = nn.MaxPool2d(2)

        self.enc3    = ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))
        self.enc_bn3 = nn.BatchNorm2d(num_hidden)
        self.res3    = ResBlock(num_hidden)
        self.pool3   = nn.MaxPool2d(2)

        self.enc4    = ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))
        self.enc_bn4 = nn.BatchNorm2d(num_hidden)
        self.res4    = ResBlock(num_hidden)
        self.att4    = AttentionBlock(num_hidden)

        self.up3  = ToroidalConv2d(nn.ConvTranspose2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))
        self.dec3 = ResBlock(num_hidden)
        self.att3 = AttentionBlock(num_hidden)

        self.up2  = ToroidalConv2d(nn.ConvTranspose2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))
        self.dec2 = ResBlock(num_hidden)
        self.att2 = AttentionBlock(num_hidden)

        self.up1  = ToroidalConv2d(nn.ConvTranspose2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))
        self.dec1 = ResBlock(num_hidden)
        self.att1 = AttentionBlock(num_hidden)

        self.final   = nn.Conv2d(num_hidden, GRID_NUM_CHANNELS, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.enc_bn1(self.enc1(x)))
        x1 = self.res1(x1)
        x2 = self.pool1(x1)

        x2 = F.relu(self.enc_bn2(self.enc2(x2)))
        x2 = self.res2(x2)
        x3 = self.pool2(x2)

        x3 = F.relu(self.enc_bn3(self.enc3(x3)))
        x3 = self.res3(x3)
        x4 = self.pool3(x3)

        x4 = F.relu(self.enc_bn4(self.enc4(x4)))
        x4 = self.res4(x4)
        x4 = self.att4(x4)

        # Decoder
        x = self.up3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        x = self.att3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        x = self.att2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        x = self.att1(x)

        x = self.final(x)
        return self.sigmoid(x)

    def name(self):
        str_network = "UNetResNetAttention_"
        str_network += f"{self.num_hidden}hidden"
        return str_network

