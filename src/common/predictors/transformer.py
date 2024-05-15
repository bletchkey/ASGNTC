import torch
import torch.nn as nn

from src.common.utils.helpers import toroidal_Conv2d
from configs.constants import *


class GoLTransformer(nn.Module):
    def __init__(self, board_size, num_layers, num_heads, d_model, dropout=0.1):
        super(GoLTransformer, self).__init__()
        self.board_size = board_size  # Assuming board_size is 32 in your case
        self.d_model = d_model
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=num_heads,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          dropout=dropout)
        self.fc_in = nn.Linear(board_size * board_size, d_model)  # 32*32 in your case
        self.fc_out = nn.Linear(d_model, board_size * board_size)  # Projecting back to board space

    def forward(self, src):
        # Assuming src is of shape [batch_size, 1, board_size, board_size]
        # Flatten board to 1D vector
        src = src.view(-1, self.board_size * self.board_size)  # Now src is [batch_size, 1024]

        # Embedding the input
        src = self.fc_in(src)  # [batch_size, d_model]

        # Transformer expects [seq_len, batch_size, d_model], seq_len is 1 here
        src = src.unsqueeze(0)  # [1, batch_size, d_model]

        # Dummy target for transformer's decoder, not used for prediction here
        tgt = torch.zeros_like(src)

        # Passing the source through the transformer
        output = self.transformer(src, tgt)

        # Projecting the output back to board space
        output = self.fc_out(output.squeeze(0))  # Back to [batch_size, board_size * board_size]

        # Applying sigmoid to get a probability for each cell
        output = torch.sigmoid(output)

        # Reshaping output to board shape
        output = output.view(-1, 1, self.board_size, self.board_size)  # Back to [batch_size, 1, 32, 32]

        return output

