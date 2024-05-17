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
        if self.padding  <= 0:
            return x

        x = torch.cat([x[:, :, -self.padding :], x, x[:, :, :self.padding ]], dim=2)
        x = torch.cat([x[:, :, :, -self.padding :], x, x[:, :, :, :self.padding ]], dim=3)
        return x

