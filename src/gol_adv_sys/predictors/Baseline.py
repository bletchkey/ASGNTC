import torch
import torch.nn as nn

from src.gol_adv_sys.predictors.utils.toroidal import toroidal_Conv2d
from src.gol_adv_sys.utils import constants as constants


class Baseline(nn.Module):
    """
    Works fine on medium metric, but not on hard metric.
    """
    def __init__(self) -> None:
        super(Baseline, self).__init__()

        # First block
        self.conv1 = nn.Conv2d(constants.nc, constants.npf, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(constants.npf, constants.npf, kernel_size=3, padding=0)

        # Second block
        self.conv3 = nn.Conv2d(constants.npf, constants.npf*2, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(constants.npf*2, constants.npf*2, kernel_size=3, padding=0)

        # Third block
        self.conv5 = nn.Conv2d(constants.npf*2, constants.npf*4, kernel_size=3, padding=0)
        self.conv6 = nn.Conv2d(constants.npf*4, constants.npf*4, kernel_size=3, padding=0)

        # Output Convolution
        self.output_conv = nn.Conv2d(constants.npf*4, constants.nc, kernel_size=1, padding=0)

        # Activation Function
        self.relu = nn.ReLU()

    def forward(self, x):

        x = toroidal_Conv2d(x, self.conv1)
        x = self.relu(x)
        x = toroidal_Conv2d(x, self.conv2)
        x = self.relu(x)

        x = toroidal_Conv2d(x, self.conv3)
        x = self.relu(x)
        x = toroidal_Conv2d(x, self.conv4)
        x = self.relu(x)

        x = toroidal_Conv2d(x, self.conv5)
        x = self.relu(x)
        x = toroidal_Conv2d(x, self.conv6)
        x = self.relu(x)
        x = toroidal_Conv2d(x, self.conv6)
        x = self.relu(x)

        x = self.output_conv(x)

        return x


class Baseline_v2(nn.Module):
    """
    It doesn't work neither on medium nor on hard metric.
    """
    def __init__(self) -> None:
        super(Baseline_v2, self).__init__()

        self.convs = nn.ModuleList([nn.Conv2d(24, 24, kernel_size=3, padding=0) for _ in range(20)])

        self.in_conv = nn.Conv2d(constants.nc, 24, kernel_size=3, padding=0)
        self.out_conv = nn.Conv2d(24, constants.nc, kernel_size=1, padding=0)

        self.relu = nn.ReLU()


    def forward(self, x):

        # stem
        x = toroidal_Conv2d(x, self.in_conv)
        x = self.relu(x)

        # stage 1
        for conv in self.convs:
            x = toroidal_Conv2d(x, conv)
            x = self.relu(x)

        x = self.out_conv(x)

        return x


class Baseline_v3(nn.Module):
    """
    Not tested properly yet
    """
    def __init__(self) -> None:
        super(Baseline_v3, self).__init__()

        # First step
        self.in_conv = nn.Conv2d(constants.nc, 96, kernel_size=3, padding=0)

        # Second step
        self.conv_1x1 = nn.Conv2d(16, 256, kernel_size=1, padding=0)
        self.conv_3x3 = nn.Conv2d(16, 128, kernel_size=3, padding=0)
        self.conv_5x5 = nn.Conv2d(16, 64, kernel_size=5, padding=0)
        self.conv_7x7 = nn.Conv2d(16, 32, kernel_size=7, padding=0)

        # Third step
        self.out_conv = nn.Conv2d(512, constants.nc, kernel_size=1, padding=0)

        #Activation Function
        self.relu = nn.ReLU()


    def forward(self, x):

        for _ in range(2):
            x = toroidal_Conv2d(x, self.in_conv)
            x = self.relu(x)

            parts = [x[:, i:i+16, :, :] for i in range(0, 64, 16)]
            parts.append(x[:, 64:, :, :])

            x_1x1 = self.conv_1x1(parts[0])
            x_1x1 = self.relu(x_1x1)

            x_3x3 = toroidal_Conv2d(parts[1], self.conv_3x3)
            x_3x3 = self.relu(x_3x3)

            x_5x5 = toroidal_Conv2d(parts[2], self.conv_5x5)
            x_5x5 = self.relu(x_5x5)

            x_7x7 = toroidal_Conv2d(parts[3], self.conv_7x7)
            x_7x7 = self.relu(x_7x7)

            x = torch.cat([x_1x1, x_3x3, x_5x5, x_7x7, parts[4]], dim=1)

            x = self.out_conv(x)
            x = self.relu(x)

        return x

class Baseline_v4(nn.Module):
    """
    Not tested properly yet.
    """
    def __init__(self) -> None:
        super(Baseline_v4, self).__init__()

        self.in_conv = nn.Conv2d(constants.nc, 32, kernel_size=3, stride=1, padding=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0) for _ in range(40)
        ])
        self.bn_layers = nn.ModuleList([nn.BatchNorm2d(32) for _ in range(40)])
        self.out_conv = nn.Conv2d(32, constants.nc, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()


    def forward(self, x):

        # Stem - First convolution and identity
        identity = x = toroidal_Conv2d(x, self.in_conv, padding=1)

        # Stage 1 - Applying convolutions with skip connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.bn_layers)):
            out = toroidal_Conv2d(x, conv, padding=1)
            out = bn(out)
            out = self.relu(out)

            # Skip connection and identity update every N layers (e.g., every 2 layers)
            if (i + 2) % 1 == 0:
                out = out + identity  # Element-wise addition without modifying identity in-place
                identity = out # Update identity to the latest output for the next skip connection

            x = out

        # Final output convolution
        x = self.out_conv(x)
        return x


