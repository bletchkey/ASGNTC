import torch
import torch.nn as nn
import torch.nn.functional as F


from src.common.utils.simulation_functions import *
from src.gol_adv_sys.utils.helpers         import get_initial_config

from src.common.utils.toroidal import ToroidalConv2d
from configs.constants import *


def get_framesteps(generated, n_framesteps, topology, device):
    """
    Get the framesteps from the generated initial configuration

    """
    initial = get_initial_config(generated, INIT_CONFIG_INTIAL_THRESHOLD)

    sim_configs = basic_simulation_config(initial,
                                          topology,
                                          n_framesteps,
                                          device)

    # stack the configurations
    sim_configs = torch.stack(sim_configs, dim=1)

    return sim_configs


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, num_input, num_hidden, kernel_size, bias=True):
        super(SpatioTemporalLSTMCell, self).__init__()
        self.num_hidden = num_hidden

        self.conv_xh = nn.Conv2d(num_input + num_hidden, num_hidden * 4, kernel_size, padding=kernel_size//2, bias=bias)
        self.conv_hh = nn.Conv2d(num_hidden * 2, num_hidden * 4, kernel_size, padding=kernel_size//2, bias=bias)

        self.conv_spatial = ToroidalConv2d(nn.Conv2d(num_hidden, num_hidden, kernel_size, padding=0, bias=bias),
                                           padding=kernel_size//2)

    def forward(self, x, h, c, spatial_c):
        combined   = torch.cat([x, h], dim=1)  # Concat along channel axis
        conv_xh    = self.conv_xh(combined)
        i, f, o, g = torch.split(conv_xh, self.num_hidden, dim=1)

        combined_h = torch.cat([h, spatial_c], dim=1)
        conv_hh = self.conv_hh(combined_h)
        i_spatial, f_spatial, o_spatial, g_spatial = torch.split(conv_hh, self.num_hidden, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        i_spatial = torch.sigmoid(i_spatial)
        f_spatial = torch.sigmoid(f_spatial)
        o_spatial = torch.sigmoid(o_spatial)
        g_spatial = torch.tanh(g_spatial)

        new_c = f * c + i * g
        new_spatial_c = f_spatial * spatial_c + i_spatial * g_spatial

        new_h = o * torch.tanh(new_c) + o_spatial * torch.tanh(new_spatial_c)

        return new_h, new_c, new_spatial_c


class SpatioTemporalLSTMLayer(nn.Module):
    def __init__(self, num_input, num_hidden, kernel_size, num_layers):
        super(SpatioTemporalLSTMLayer, self).__init__()

        self.num_layers = num_layers
        self.st_lstm_cells = nn.ModuleList([SpatioTemporalLSTMCell(num_input if i == 0 else num_hidden,
                                                                   num_hidden, kernel_size) for i in range(num_layers)])

    def forward(self, x, hidden_states):
        seq_len = x.size(1)
        for t in range(seq_len):
            input_tensor = x[:, t, :, :, :]
            new_hidden_states = []
            for i, st_lstm_cell in enumerate(self.st_lstm_cells):
                h, c, spatial_c = hidden_states[i]
                h, c, spatial_c = st_lstm_cell(input_tensor, h, c, spatial_c)
                new_hidden_states.append((h, c, spatial_c))
                input_tensor = h
            hidden_states = new_hidden_states
        return input_tensor, hidden_states


class SpatioTemporal(nn.Module):
    def __init__(self,
                 topology,
                 num_input,
                 num_hidden,
                 num_layers,
                 framesteps):
        super(SpatioTemporal, self).__init__()

        self.framesteps = framesteps
        self.topology   = topology
        self.num_hidden = num_hidden

        self.st_lstm_layer = SpatioTemporalLSTMLayer(num_input, num_hidden, kernel_size=3, num_layers=num_layers)
        self.conv_last     = nn.Conv2d(num_hidden, num_input, kernel_size=3, padding=0)

    def forward(self, x):
        steps = get_framesteps(x, self.framesteps, self.topology, x.device)

        batch_size, channels, height, width = x.size()
        hidden_states = self.init_hidden(x)
        predictions = []

        for t in range(steps):
            x_t = x[:, t, :, :, :]  # Select the t-th time step
            x_t, hidden_states = self.st_lstm_layer(x_t.unsqueeze(1), hidden_states)  # Add time dimension back for LSTM
            x_t = self.conv_last(x_t.squeeze(1))  # Remove time dimension after LSTM
            predictions.append(x_t)

        return torch.stack(predictions, dim=1)

    def init_hidden(self, x):
        batch_size, steps, channels, height, width = x.size()
        hidden_states = []
        for i in range(self.st_lstm_layer.num_layers):
            hidden_states.append((torch.zeros(batch_size, self.st_lstm_layer.st_lstm_cells[0].num_hidden, height, width).to(x.device),
                                  torch.zeros(batch_size, self.st_lstm_layer.st_lstm_cells[0].num_hidden, height, width).to(x.device),
                                  torch.zeros(batch_size, self.st_lstm_layer.st_lstm_cells[0].num_hidden, height, width).to(x.device)))
        return hidden_states

    def name(self):
        str_network = "LifeMotionPredictor_"
        str_network += f"{self.num_hidden}hidden_"
        str_network += f"{self.steps}framesteps_"
        str_network += f"{self.topology}topology"
        return str_network

