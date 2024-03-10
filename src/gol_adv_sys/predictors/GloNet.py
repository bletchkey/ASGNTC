import torch
import torch.nn as nn

from ..utils.helper_functions import add_toroidal_padding
from ..utils import constants as constants


class GloNet(nn.Module):

    def __init__(self):
        super(GloNet, self).__init__()


    def _pad_conv(self, x, f):
        x = add_toroidal_padding(x)
        x = f(x)

        return x

    def forward(self, x):

        return x
