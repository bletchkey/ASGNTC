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

    # TODO: This works but it's not the most efficient way to do it
    def __add_toroidal_padding(self, x):
        if self.padding  <= 0:
            return x

        x = torch.cat([x[:, :, -self.padding :], x, x[:, :, :self.padding ]], dim=2)
        x = torch.cat([x[:, :, :, -self.padding :], x, x[:, :, :, :self.padding ]], dim=3)

        return x

    # def __add_toroidal_padding(self, x):
    #     if self.padding <= 0:
    #         return x

    #     B, C, H, W = x.shape
    #     pad        = self.padding

    #     # Create an empty tensor with the new shape
    #     padded_x = torch.empty((B, C, H + 2*pad, W + 2*pad), device=x.device, dtype=x.dtype)

    #     # Center
    #     padded_x[:, :, pad:pad+H, pad:pad+W] = x

    #     # Wrap edges
    #     # Horizontal wrap (left-right)
    #     padded_x[:, :, pad:pad+H, :pad]   = x[:, :, :, -pad:]  # left wrap
    #     padded_x[:, :, pad:pad+H, pad+W:] = x[:, :, :, :pad]  # right wrap

    #     # Vertical wrap (top-bottom) with horizontal edges already wrapped
    #     padded_x[:, :, :pad, :]   = padded_x[:, :, -2*pad:-pad, :]  # top wrap
    #     padded_x[:, :, pad+H:, :] = padded_x[:, :, pad:2*pad, :]  # bottom wrap

    #     return padded_x

