import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.utils.toroidal import ToroidalConv2d
from configs.constants import *


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    ConvLSTM module that can stack multiple ConvLSTM layers.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True):
        super(ConvLSTM, self).__init__()

        self.batch_first = batch_first
        self.conv_lstm_layers = nn.ModuleList()

        for i in range(num_layers):
            input_dim = input_dim if i == 0 else hidden_dim[i-1]
            self.conv_lstm_layers.append(ConvLSTMCell(input_dim, hidden_dim[i], kernel_size, bias))

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)  # Convert to (seq_len, batch, channels, height, width)

        current_input = input_tensor
        for layer in self.conv_lstm_layers:
            output_inner = []
            for t in range(current_input.size(0)):
                h, c = hidden_state[layer] if hidden_state is not None else (None, None)
                if t == 0 and h is None:
                    h, c = layer.init_hidden(current_input.size(1), (current_input.size(3), current_input.size(4)))
                h, c = layer(current_input[t, :, :, :, :], (h, c))
                output_inner.append(h.unsqueeze(0))
            current_input = torch.cat(output_inner, dim=0)
            hidden_state[layer] = (h, c)

        return current_input, hidden_state

    def init_hidden(self, batch_size, image_size):
        init_states = []
        for layer in self.conv_lstm_layers:
            init_states.append(layer.init_hidden(batch_size, image_size))
        return init_states


class ConvLSTMPred(nn.Module):
    def __init__(self,
                 topology: str,
                 num_hidden: int):
        super(ConvLSTMPred, self).__init__()

        self.topology   = topology
        self.num_hidden = num_hidden

        # self.conv1 = nn.Sequential(
        #     *(
        #         [ToroidalConv2d(nn.Conv2d(NUM_CHANNELS_GRID, num_hidden, kernel_size=3, stride=1, padding=0))]
        #         if self.topology == TOPOLOGY_TOROIDAL
        #         else [nn.Conv2d(NUM_CHANNELS_GRID, num_hidden, kernel_size=3, stride=1, padding=1)]
        #     )
        # )

        self.conv1 = nn.Conv2d(NUM_CHANNELS_GRID, num_hidden, kernel_size=3, stride=1, padding=1)

        self.conv_lstm = ConvLSTM(input_dim=1,
                                  hidden_dim=[num_hidden],
                                  kernel_size=3,
                                  num_layers=1,
                                  batch_first=True,
                                  bias=True)

        self.conv2 = nn.Conv2d(num_hidden, NUM_CHANNELS_GRID, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        # x: (batch_size, sequence_length, channels, height, width)
        batch_size, channels, height, width = x.shape
        seq_len = 1

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

