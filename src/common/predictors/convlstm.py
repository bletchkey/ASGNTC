import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.utils.toroidal import ToroidalConv2d
from configs.constants import *


class ConvLSTMPred(nn.Module):
    def __init__(self,
                 topology: str,
                 num_hidden: int):
        super(ConvLSTMPred, self).__init__()

        self.topology   = topology
        self.num_hidden = num_hidden

        self.conv1 = nn.Sequential(
            *(
                [ToroidalConv2d(nn.Conv2d(NUM_CHANNELS_GRID, num_hidden, kernel_size=3, stride=1, padding=0))]
                if self.topology == TOPOLOGY_TOROIDAL
                else [nn.Conv2d(NUM_CHANNELS_GRID, num_hidden, kernel_size=3, stride=1, padding=1)]
            )
        )

        self.conv_lstm = nn.ConvLSTM(input_channels=num_hidden,
                                     hidden_channels=[num_hidden],
                                     kernel_size=(3, 3),
                                     num_layers=1,
                                     batch_first=True,
                                     bias=True,
                                     return_all_layers=False)

        self.conv2 = nn.Conv2d(num_hidden, NUM_CHANNELS_GRID, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        # x: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape

        # Initial processing
        processed_frames = []
        for t in range(seq_len):
            frame = F.relu(self.conv1(x[:, t]))
            processed_frames.append(frame.unsqueeze(1))
        processed_frames = torch.cat(processed_frames, dim=1)  # Reshape to (batch, seq_len, channels, H, W)

        # ConvLSTM processing
        lstm_out, _ = self.conv_lstm(processed_frames)
        last_hidden_state = lstm_out[0][0][:, -1, :, :, :]  # Taking the last time step output from the last layer

        # Final prediction of the next frame
        predicted_frame = F.relu(self.conv2(last_hidden_state))
        return predicted_frame

    def name(self):
        str_network = "ConvLSTM_"
        str_network += f"{self.num_hidden}hidden_"
        str_network += f"{self.topology}topology"
        return str_network

