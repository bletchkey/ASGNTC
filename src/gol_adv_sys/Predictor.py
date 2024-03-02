import torch
import torch.nn as nn

from .utils import constants as constants


def add_toroidal_padding(x):
    x = torch.cat([x[:, :, -1:], x, x[:, :, :1]], dim=2)
    x = torch.cat([x[:, :, :, -1:], x, x[:, :, :, :1]], dim=3)

    return x


class block(nn.Module):
    def __init__(self, channels) -> None:
        super(block, self).__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 0)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        x = self.bn(x)
        x = self.relu(x)
        x = self._pad_conv(x, self.conv)

        x = self.bn(x)
        x = self.relu(x)
        x = self._pad_conv(x, self.conv)

        x += identity

        return x

    def _pad_conv(self, x, f):
        x = add_toroidal_padding(x)
        x = f(x)

        return x


class ResNetConstantChannels(nn.Module):
    def __init__(self, block, layers, channels):
        super(ResNetConstantChannels, self).__init__()
        self.channels = channels
        self.n_layers = len(layers)

        self.in_conv = nn.Conv2d(constants.nc, channels, kernel_size=3, padding=0)
        self.out_conv = nn.Conv2d(channels, constants.nc, kernel_size=1, padding=0)

        for i in range(self.n_layers):
            setattr(self, f"layer{i}", self._make_layer(block, layers[i], self.channels))

    def forward(self, x):

        x = self.in_conv(add_toroidal_padding(x))

        for i in range(self.n_layers):
            x = getattr(self, f"layer{i}")(x)

        x = self.out_conv(x)

        return x

    def _make_layer(self, block, num_residual_blocks, channels):
        layers = []

        for _ in range(num_residual_blocks):
            layers.append(block(channels))
            layers.append(block(channels))

        return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self) -> None:
        super(UNet, self).__init__()

        # Encoder (Downsampling)
        self.enc_conv1 = nn.Conv2d(constants.nc, constants.npf, kernel_size=3, padding=0)
        self.enc_conv2 = nn.Conv2d(constants.npf, constants.npf*2, kernel_size=3, padding=0)

        self.avgpool = nn.AvgPool2d(2, 2)
        self.maxpool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(constants.npf*2, constants.npf*4, kernel_size=3, padding=0)

        # Decoder (Upsampling)
        self.up_conv1 = nn.ConvTranspose2d(constants.npf*4, constants.npf*2, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(constants.npf*4, constants.npf*2, kernel_size=3, padding=0)
        self.up_conv2 = nn.ConvTranspose2d(constants.npf*2, constants.npf, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv2d(constants.npf*2, constants.npf, kernel_size=3, padding=0)

        # Output Convolution
        self.output_conv = nn.Conv2d(constants.npf, constants.nc, kernel_size=1, padding = 0)

        # Activation Function
        self.relu = nn.ReLU()

    def forward(self, x):

        for _ in range(10):
            x = self._u_structure(x)

        return x

    def _pad_conv(self, x, f):

        x = add_toroidal_padding(x)
        x = f(x)

        return x

    def _u_structure(self, x):

        # Encoder
        x = self.relu(x)

        enc_1 = self._pad_conv(x, self.enc_conv1)
        x = self.avgpool(enc_1)
        x = self.relu(x)

        enc_2 = self._pad_conv(x, self.enc_conv2)
        x = self.avgpool(enc_2)
        x = self.relu(x)

        # Bottleneck
        x = self._pad_conv(x, self.bottleneck_conv)

        # Decoder
        dec_1 = self.up_conv1(x)
        x = torch.cat((dec_1, enc_2), dim=1)
        x = self.relu(x)
        x = self._pad_conv(x, self.dec_conv1)

        dec_2 = self.up_conv2(x)
        x = torch.cat((dec_2, enc_1), dim=1)
        x = self.relu(x)
        x = self._pad_conv(x, self.dec_conv2)

        # Output
        out = self.output_conv(x)

        return out


class VGGLike(nn.Module):
    def __init__(self) -> None:
        super(VGGLike, self).__init__()

        self.conv1 = nn.Conv2d(constants.nc, constants.npf, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(constants.npf, constants.npf*2, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(constants.npf*2, constants.npf*4, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(constants.npf*4, constants.npf*8, kernel_size=3, padding=0)
        self.conv5 = nn.Conv2d(constants.npf*8, constants.npf*16, kernel_size=3, padding=0)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x)

        x = self._pad_conv(x, self.conv1)
        x = self.relu(x)

        x = self._pad_conv(x, self.conv2)
        x = self.relu(x)

        x = self._pad_conv(x, self.conv3)
        x = self.relu(x)

        x = self._pad_conv(x, self.conv4)
        x = self.relu(x)

        x = self._pad_conv(x, self.conv5)
        x = self.relu(x)

        return x

    def _pad_conv(self, x, f):
        x = add_toroidal_padding(x)
        x = f


def Predictor_ResNet():
    return ResNetConstantChannels(block, [2, 2, 2, 2], constants.grid_size)

def Predictor_UNet():
    return UNet()

def Predictor_VGGLike():
    return VGGLike()

