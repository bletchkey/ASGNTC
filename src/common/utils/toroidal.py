import torch
import torch.nn as nn

class ToroidalConv2d(nn.Module):
    def __init__(self, conv2d, padding=1):
        super().__init__()
        self.conv2d  = conv2d
        self.padding = padding

    def forward(self, x):
        x = self.__add_toroidal_padding(x)
        x = self.conv2d(x)
        return x


    def __add_toroidal_padding(self, x):
        if self.padding <= 0:
            return x

        B, C, H, W = x.shape
        pad = self.padding

        # Create an empty tensor with the new shape
        padded_x = torch.empty((B, C, H + 2*pad, W + 2*pad), device=x.device, dtype=x.dtype)

        # Center
        padded_x[:, :, pad:pad+H, pad:pad+W] = x

        # Left and right
        padded_x[:, :, pad:pad+H, :pad] = x[:, :, :, -pad:]
        padded_x[:, :, pad:pad+H, pad+W:] = x[:, :, :, :pad]

        # Top and bottom
        padded_x[:, :, :pad, :] = padded_x[:, :, H:H+pad, :]
        padded_x[:, :, pad+H:, :] = padded_x[:, :, pad:2*pad, :]

        return padded_x

