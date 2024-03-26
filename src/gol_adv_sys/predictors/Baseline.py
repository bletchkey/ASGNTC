import torch
import torch.nn as nn

from src.gol_adv_sys.predictors.utils.toroidal import toroidal_Conv2d
from config.constants import *

class Baseline(nn.Module):
    """
    Baseline model

    """
    def __init__(self) -> None:
        super(Baseline, self).__init__()

        n_layers = 20

        self.in_conv = nn.Conv2d(N_CHANNELS, 32, kernel_size=3, stride=1, padding=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0) for _ in range(n_layers)
        ])
        self.bn_layers = nn.ModuleList([nn.BatchNorm2d(32) for _ in range(n_layers)])

        self.out_conv = nn.Conv2d(32, N_CHANNELS, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()


    def forward(self, x):
        # Stem - First convolution and identity
        identity = x = toroidal_Conv2d(x, self.in_conv, padding=1)

        # Stage 1 - Applying convolutions with skip connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.bn_layers[1:]), 1):
            x = bn(x)
            x = self.relu(x)
            x = toroidal_Conv2d(x, conv, padding=1)

            # Skip connection and identity update every N layers (e.g., every 2 layers)
            if i % 2 == 0:
                x = x + identity  # Element-wise addition without modifying identity in-place
                identity = x # Update identity to the latest output for the next skip connection

        # Final output convolution
        x = self.out_conv(x)
        return x

