import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.constants import *
from src.common.utils.toroidal import ToroidalConv2d

def sample_gumbel(shape, device, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape, device = device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.size(), device = logits.device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y

    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will be a probabilistic distribution.
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = logits.size(-1)
        # Straight through.
        y_hard = (y == y.max(dim=-1, keepdim=True)[0]).float()
        y = (y_hard - y).detach() + y
    return y

class BinaryGenerator(nn.Module):
    def __init__(self,
                 topology,
                 num_hidden,
                 temperature=1.0):
        super(BinaryGenerator, self).__init__()

        self.temperature = temperature
        self.num_hidden  = num_hidden
        self.topology    = topology

        self.conv1  = nn.Sequential(
            *(
                [ToroidalConv2d(nn.Conv2d(NUM_CHANNELS_GRID, num_hidden, kernel_size=3, stride=1, padding=0))]
                if self.topology == TOPOLOGY_TOROIDAL
                else [nn.Conv2d(NUM_CHANNELS_GRID, num_hidden, kernel_size=3, stride=1, padding=1)]
            )
        )

        self.conv2  = nn.Sequential(
            *(
                [ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))]
                if self.topology == TOPOLOGY_TOROIDAL
                else [nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1)]
            )
        )

        self.conv3  = nn.Sequential(
            *(
                [ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=0))]
                if self.topology == TOPOLOGY_TOROIDAL
                else [nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1)]
            )
        )

        self.conv4 = nn.Conv2d(num_hidden, 2, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, -0.1)  # Slight negative bias to discourage ones

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        logits = self.conv4(x)  # (batch_size, 2, grid_size, grid_size)

        # Apply Gumbel-Softmax to each grid cell
        logits = logits.permute(0, 2, 3, 1).contiguous()  # (batch_size, grid_size, grid_size, 2)
        logits = logits.view(-1, 2)  # (batch_size * grid_size * grid_size, 2)
        y = gumbel_softmax(logits, self.temperature, hard=True)
        y = y[:, 1]  # Take the binary output
        y = y.view(batch_size, 1, GRID_SIZE, GRID_SIZE)  # Reshape to (batch_size, 1, grid_size, grid_size)
        return y

    def name(self):
        str_network = "BinaryGen_"
        str_network += f"{self.num_hidden}hidden_"
        str_network += f"{self.topology}"

        return str_network

